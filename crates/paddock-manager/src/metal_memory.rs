//! Unified memory accounting. Metal allocations and CPU prefix caches consume
//! the SAME physical RAM. Never present process-local currentAllocatedSize as
//! machine-wide GPU usage. Availability includes pageable file-backed caches,
//! not inactive anonymous pages or the uncompressed contents of the compressor.
use crate::routes::AppState;

#[cfg(any(target_os = "macos", test))]
const HEADROOM: u64 = 1 << 30;

#[cfg(any(target_os = "macos", test))]
#[derive(Default)]
struct MemoryPages {
    free: u64,
    speculative: u64,
    purgeable: u64,
    file_backed: u64,
    wired: u64,
    compressor: u64,
}

/// An advisory reclaimable-memory estimate, not a guarantee of allocation or
/// zero-cost reclamation (dirty file pages can require writeback). XNU's
/// host_statistics64 reports pageable external pages, excluding wired pages;
/// its speculative file pages also appear in free_count. Count that overlap
/// once. Never add inactive_count: it includes anonymous application memory.
/// See Apple's xnu/osfmk/kern/host.c and osfmk/mach/vm_statistics.h.
#[cfg(any(target_os = "macos", test))]
fn available_bytes(pages: MemoryPages, page_size: u64, physical: u64) -> u64 {
    let reclaimable = pages
        .free
        .saturating_add(pages.file_backed.saturating_sub(pages.speculative))
        .saturating_add(pages.purgeable)
        .saturating_mul(page_size);
    // Samples can straddle VM transitions. Never count wired/compressor memory
    // as reclaimable even if individual counters momentarily disagree.
    let unavailable = pages
        .wired
        .saturating_add(pages.compressor)
        .saturating_mul(page_size);
    reclaimable
        .min(physical.saturating_sub(unavailable))
        .saturating_sub(HEADROOM)
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub name: String,
    pub physical: u64,
    pub limit: u64,
    pub available: u64,
}

#[cfg(target_os = "macos")]
pub fn physical_bytes() -> Option<u64> {
    let mut value = 0u64;
    let mut size = std::mem::size_of_val(&value);
    // SAFETY: correctly sized writable output; a read-only, NUL-terminated key.
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && size == 8 && value > 0).then_some(value)
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc exposes mach_task_self through the SDK's legacy binding.
pub fn sample() -> Option<Snapshot> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    use objc2_metal::MTLDevice;
    let device = objc2_metal::MTLCreateSystemDefaultDevice()?;
    if !device.hasUnifiedMemory() {
        return None;
    }
    let physical = physical_bytes()?;
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: host_statistics64 receives the SDK struct and its integer count.
    // Release the send right from mach_host_self on every path.
    let result = unsafe {
        let host = libc::mach_host_self();
        let result = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &mut count,
        );
        mach_port_deallocate(libc::mach_task_self(), host);
        result
    };
    if result != libc::KERN_SUCCESS {
        return None;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return None;
    }
    let available = available_bytes(
        MemoryPages {
            free: stats.free_count.into(),
            speculative: stats.speculative_count.into(),
            purgeable: stats.purgeable_count.into(),
            file_backed: stats.external_page_count.into(),
            wired: stats.wire_count.into(),
            compressor: stats.compressor_page_count.into(),
        },
        page as u64,
        physical,
    );
    // The runner grants from the same ceiling (paddock_admin::metal_budget).
    // HEADROOM stays on `available` only (VM-sample uncertainty), never as a
    // second cut of the GPU cap.
    let recommended = device.recommendedMaxWorkingSetSize();
    Some(Snapshot {
        name: device.name().to_string(),
        physical,
        limit: paddock_admin::metal_budget::gpu_ceiling(recommended)
            .min(physical.saturating_sub(HEADROOM)),
        available,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample() -> Option<Snapshot> {
    None
}

/// Two independent limits apply and the tighter one wins.
/// - GPU cap: `limit` is shared by every runner, and live allocations consume
///   it, so the whole device reservation is charged against it.
/// - RAM: live allocations are already absent from `available`, so only the
///   unallocated part of the reservation and the host RAM caches are charged.
pub fn residual(
    snapshot: &Snapshot,
    device_reserved: u64,
    device_used: u64,
    host_reserved: u64,
) -> u64 {
    let gpu_room = snapshot.limit.saturating_sub(device_reserved);
    let ram_room = snapshot
        .available
        .saturating_sub(device_reserved.saturating_sub(device_used))
        .saturating_sub(host_reserved);
    gpu_room.min(ram_room)
}

pub async fn available(state: &AppState, freeing: Option<u16>) -> Option<(Snapshot, u64)> {
    let mut snapshot = sample()?;
    let views = state.supervisor.list().await;
    let recon = state.recon.borrow().clone();
    let mut reserved = 0u64;
    let mut used = 0u64;
    let mut host = 0u64;
    for port in views
        .iter()
        .map(|v| v.port)
        .chain(state.supervisor.spawning_ports())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let live = recon
            .as_ref()
            .as_ref()
            .and_then(|r| r.runners.iter().find(|r| r.port == port))
            .and_then(|r| r.self_mem)
            .unwrap_or(0);
        if Some(port) == freeing {
            snapshot.available = snapshot
                .available
                .saturating_add(live)
                .min(snapshot.physical);
            continue;
        }
        // A deferred save is not the configuration the running process loaded.
        // Ask its own baseline (which survives manager restarts) before trusting
        // a smaller saved ceiling. Unknown/drifted runners reserve the entire
        // backend ceiling until applied; never resell their live headroom.
        let client = paddock_admin::client::AdminClient::new(port);
        let status =
            tokio::time::timeout(std::time::Duration::from_secs(1), client.config_status())
                .await
                .ok()
                .and_then(Result::ok);
        // A budget is a ceiling, not a reservation after confirmed unload.
        // Fresh PID-checked state outranks a stale reconciliation ledger.
        // Every subsequent cold load rechecks physical memory under the
        // device-wide runner load gate; unknown/older runners stay reserved.
        if status.as_ref().is_some_and(|s| {
            views
                .iter()
                .any(|v| v.port == port && s.residency_released(v.pid))
        }) {
            continue;
        }
        let dynamic = status.as_ref().and_then(|s| {
            views
                .iter()
                .find(|v| v.port == port)
                .and_then(|v| s.residency_reservation(v.pid))
        });
        let matches = status.is_some_and(|s| {
            s.restart_required == Some(false)
                && views.iter().any(|v| v.port == port && v.pid == s.pid)
        });
        let budget = dynamic.unwrap_or_else(|| {
            reservation(
                &snapshot,
                state.supervisor.config_vram_budget(port),
                matches,
            )
        });
        tracing::debug!(
            port,
            reservation = budget,
            live,
            dynamic = dynamic.is_some(),
            baseline_matches = matches,
            fell_back_to_limit = dynamic.is_none() && budget == snapshot.limit,
            "Metal memory: runner reservation"
        );
        reserved = reserved.saturating_add(budget.max(live));
        used = used.saturating_add(live);
        if let Ok(spec) = state
            .supervisor
            .spec_from_config_file(&state.supervisor.server_config_path(port))
            && let Some(kv) = spec.kv_offload.filter(|kv| kv.enabled)
        {
            host = host.saturating_add((kv.ram_gb.max(0.0) * (1u64 << 30) as f64) as u64);
        }
    }
    let free = residual(&snapshot, reserved, used, host);
    tracing::debug!(
        limit = snapshot.limit,
        physical = snapshot.physical,
        available = snapshot.available,
        reserved,
        used,
        host,
        gpu_room = snapshot.limit.saturating_sub(reserved),
        ram_room = snapshot
            .available
            .saturating_sub(reserved.saturating_sub(used))
            .saturating_sub(host),
        free,
        "Metal memory: residual"
    );
    Some((snapshot, free))
}

fn reservation(snapshot: &Snapshot, budget_mib: Option<u64>, baseline_matches: bool) -> u64 {
    budget_mib
        .filter(|_| baseline_matches)
        .and_then(|m| m.checked_mul(1 << 20))
        .unwrap_or(snapshot.limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_checkpoint_cache_does_not_prevent_saved_budget_restart() {
        // Regression: a 32 GiB saved grant passed a cold start on a 128 GiB
        // machine, but failed after 16 GiB moved from free to model-file cache.
        let before = MemoryPages {
            free: 42,
            file_backed: 45,
            ..Default::default()
        };
        let after = MemoryPages {
            free: 26,
            file_backed: 61,
            ..Default::default()
        };
        let cold = available_bytes(before, 1 << 30, 128 << 30);
        let warm = available_bytes(after, 1 << 30, 128 << 30);
        assert_eq!(cold, warm);
        let s = Snapshot {
            name: "test".into(),
            physical: 128 << 30,
            limit: 97 << 30,
            available: warm,
        };
        assert!(residual(&s, 0, 0, 0) >= 32 << 30);
        // Cache reclaimability must not allow reselling another runner's
        // unallocated reservation, or CPU offload-cache reservations.
        assert!(residual(&s, 64 << 30, 0, 0) < 32 << 30);
        assert!(residual(&s, 0, 0, 64 << 30) < 32 << 30);
    }

    #[test]
    fn speculative_file_pages_are_counted_exactly_once() {
        let pages = MemoryPages {
            free: 8,
            speculative: 4,
            file_backed: 24,
            purgeable: 2,
            ..Default::default()
        };
        assert_eq!(available_bytes(pages, 1 << 30, 64 << 30), 29 << 30);
    }

    #[test]
    fn anonymous_and_compressed_memory_is_not_available_capacity() {
        let pages = MemoryPages {
            free: 2,
            file_backed: 2,
            wired: 4,
            compressor: 8,
            ..Default::default()
        };
        // The rest is anonymous memory, not free just because it is pageable.
        assert_eq!(available_bytes(pages, 1 << 30, 128 << 30), 3 << 30);
    }

    #[test]
    fn inconsistent_samples_saturate_without_spending_os_headroom() {
        assert_eq!(available_bytes(MemoryPages::default(), 16384, 8 << 30), 0);
        assert_eq!(
            available_bytes(
                MemoryPages {
                    free: u64::MAX,
                    file_backed: u64::MAX,
                    purgeable: u64::MAX,
                    wired: 4,
                    compressor: 2,
                    ..Default::default()
                },
                1 << 30,
                8 << 30,
            ),
            1 << 30,
        );
        assert_eq!(
            available_bytes(
                MemoryPages {
                    free: 1,
                    speculative: 10,
                    file_backed: 4,
                    ..Default::default()
                },
                1 << 30,
                8 << 30,
            ),
            0,
        );
    }
    #[test]
    fn reservations_share_ram_without_double_charging_live_gpu_bytes() {
        let s = Snapshot {
            name: "test".into(),
            physical: 100,
            limit: 90,
            available: 60,
        };
        assert_eq!(residual(&s, 40, 30, 0), 50);
        assert_eq!(residual(&s, 40, 30, 15), 35);
        assert_eq!(residual(&s, 100, 30, 0), 0);
        assert_eq!(residual(&s, 10, 10, 100), 0);
        // 10 GiB live is already missing from `available`, but still consumes
        // the shared GPU cap, so the cap binds (22 - 10), not `available`.
        let mac = Snapshot {
            name: "test".into(),
            physical: 32 << 30,
            limit: 22 << 30,
            available: 21 << 30,
        };
        assert_eq!(residual(&mac, 10 << 30, 10 << 30, 0), 12 << 30);
        // Host RAM caches lower RAM room but never the GPU cap.
        assert_eq!(residual(&mac, 0, 0, 15 << 30), 6 << 30);
        // Unallocated reservation counts against both limits.
        assert_eq!(residual(&mac, 10 << 30, 4 << 30, 0), 12 << 30);
        assert_eq!(reservation(&s, Some(1), true), 1 << 20);
        assert_eq!(reservation(&s, Some(1), false), s.limit);
        assert_eq!(reservation(&s, None, true), s.limit);
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn apple_silicon_reports_real_memory() {
        use objc2_metal::MTLDevice;
        let s = sample().unwrap();
        let recommended = objc2_metal::MTLCreateSystemDefaultDevice()
            .expect("Metal device")
            .recommendedMaxWorkingSetSize();
        assert!(s.physical > 1 << 30 && s.limit <= s.physical && s.available <= s.physical);
        assert_eq!(
            s.limit,
            paddock_admin::metal_budget::gpu_ceiling(recommended)
                .min(s.physical.saturating_sub(HEADROOM))
        );
    }
}
