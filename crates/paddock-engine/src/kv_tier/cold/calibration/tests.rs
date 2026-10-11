use super::super::{ColdCache, ColdConfig, CostModel, Done, Job, Payload};
use super::*;
use std::{path::PathBuf, sync::mpsc};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "paddock-cost-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn config(&self) -> ColdConfig {
        ColdConfig {
            ram_bytes: 4 << 20,
            disk: Some((self.0.clone(), 8 << 20)),
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        // Only this unique test-owned directory, after worker/cache teardown.
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn measured() -> CostModel {
    let mut cost = CostModel::new();
    cost.observe_prefill(1024, 1_000_000.);
    cost
}
fn wait(cache: &mut ColdCache) {
    let start = Instant::now();
    loop {
        cache.pump();
        if cache
            .calibration
            .as_ref()
            .is_none_or(|c| c.in_flight.is_none())
            && !cache.writes_pending()
            && cache.stats().open_tickets == 0
        {
            return;
        }
        assert!(start.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn metadata_is_device_matched_bounded_and_atomic() {
    let dir = Temp::new();
    let c = measured().calibration().unwrap();
    assert_eq!(read(&dir.0, "metal-device"), None);
    save(&dir.0, "metal-device", c).unwrap();
    assert_eq!(read(&dir.0, "metal-device"), Some(c));
    assert_eq!(read(&dir.0, "different-device"), None);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(dir.0.join(FILE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let text = c.encode("metal-device");
    for invalid in [
        text[..text.len() / 2].to_owned(),
        "x".repeat(MAX_BYTES as usize + 1),
        text.replace("prefill_tpus 0.001024", "prefill_tpus NaN"),
        text.replace("prefill_tpus 0.001024", "prefill_tpus inf"),
        text.replace("prefill_tpus 0.001024", "prefill_tpus 0"),
    ] {
        assert_ne!(invalid, text);
        std::fs::write(dir.0.join(FILE), invalid).unwrap();
        assert_eq!(read(&dir.0, "metal-device"), None);
    }
    save(&dir.0, "metal-device", c).unwrap();
    assert_eq!(read(&dir.0, "metal-device"), Some(c));
    assert_eq!(
        std::fs::read_dir(&dir.0).unwrap().count(),
        1,
        "no abandoned temporary files"
    );
}

#[cfg(unix)]
#[test]
fn metadata_never_follows_an_existing_symlink() {
    let dir = Temp::new();
    let outside = dir.0.join("not-metadata");
    std::fs::write(&outside, b"keep").unwrap();
    std::os::unix::fs::symlink(&outside, dir.0.join(FILE)).unwrap();
    assert_eq!(read(&dir.0, "metal"), None);
    save(&dir.0, "metal", measured().calibration().unwrap()).unwrap();
    assert_eq!(std::fs::read(&outside).unwrap(), b"keep");
    assert!(
        !std::fs::symlink_metadata(dir.0.join(FILE))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[test]
fn changes_coalesce_and_failed_writes_back_off_without_losing_latest() {
    let dir = Temp::new();
    let mut state = State::load(&dir.0, "metal");
    let now = Instant::now();
    let mut cost = measured();
    let first = cost.calibration().unwrap();
    state.observe(first);
    assert_eq!(state.next(now), Some(first));
    state.submitted(now, first);
    assert_eq!(state.shutdown_value(), None, "already queued");
    for us in [2_000_000., 3_000_000., 4_000_000.] {
        cost.observe_prefill(1024, us);
        state.observe(cost.calibration().unwrap());
    }
    let latest = cost.calibration().unwrap();
    assert_eq!(
        state.next(now + SAVE_EVERY),
        None,
        "one in-flight write maximum"
    );
    assert_eq!(state.shutdown_value(), Some(latest));
    state.complete(first, false);
    assert_eq!(
        state.next(now + SAVE_EVERY - Duration::from_millis(1)),
        None
    );
    assert_eq!(state.next(now + SAVE_EVERY), Some(latest));
    state.submitted(now + SAVE_EVERY, latest);
    state.complete(latest, true);
    assert_eq!(
        state.next(now + SAVE_EVERY * 2),
        None,
        "unchanged rates are not rewritten"
    );
    assert_eq!(state.shutdown_value(), None);
}

#[test]
fn full_queue_never_blocks_and_retries_after_capacity_returns() {
    let dir = Temp::new();
    let mut cache = ColdCache::open(
        ColdConfig {
            ram_bytes: 4 << 20,
            disk: None,
        },
        1 << 20,
    )
    .unwrap();
    cache.calibration = Some(State::load(&dir.0, "metal"));
    let (tx, jobs) = mpsc::sync_channel(1);
    let (done, rx) = mpsc::channel();
    cache.rx = rx;
    let cost = measured();
    let c = cost.calibration().unwrap();
    assert!(tx.try_send(Job::Calibrate(c)).is_ok());
    cache.tx = Some(tx);
    cache.observe_cost(&cost);
    assert!(cache.calibration.as_ref().unwrap().in_flight.is_none());
    assert!(matches!(jobs.try_recv(), Ok(Job::Calibrate(_))));
    cache.pump();
    assert_eq!(cache.calibration.as_ref().unwrap().in_flight, Some(c));
    assert!(matches!(jobs.try_recv(), Ok(Job::Calibrate(v)) if v == c));
    assert!(done.send(Done::Calibrated(c, true)).is_ok());
    cache.pump();
    assert!(cache.calibration.as_ref().unwrap().in_flight.is_none());
    assert!(jobs.try_recv().is_err());
    assert!(!cache.stats().tripped);
}

#[test]
fn restart_uses_measured_seed_and_first_live_prefill_replaces_it() {
    let dir = Temp::new();
    let mut cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "metal").unwrap();
    assert_eq!(cache.calibration(), None);
    let mut cost = measured();
    cache.observe_cost(&cost);
    wait(&mut cache);
    assert_eq!(read(&dir.0, "metal"), cost.calibration());
    // A short run still saves its latest coalesced measurement at shutdown.
    cost.observe_prefill(1024, 2_000_000.);
    cache.observe_cost(&cost);
    let expected = cost.calibration().unwrap();
    drop(cache);
    let cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "metal").unwrap();
    assert_eq!(cache.calibration(), Some(expected));
    let mut fresh = CostModel::new();
    fresh.adopt_calibration(cache.calibration().unwrap());
    assert!(
        fresh.calibration().is_none(),
        "restored seed is not a live measurement"
    );
    fresh.observe_prefill(1024, 500_000.);
    assert_eq!(fresh.calibration().unwrap().prefill_tpus, 1024. / 500_000.);
    drop(cache);
    let cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "another GPU").unwrap();
    assert_eq!(cache.calibration(), None);
}

#[test]
fn metadata_failure_does_not_disable_payload_caching_or_restore() {
    let dir = Temp::new();
    let mut cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "metal").unwrap();
    std::fs::create_dir(dir.0.join(FILE)).unwrap(); // Atomic rename must fail.
    cache.observe_cost(&measured());
    let reservation = cache.reserve(4096).unwrap();
    cache.put([19; 32], Payload::new(vec![42; 4096], reservation));
    wait(&mut cache);
    assert!(!cache.stats().tripped);
    assert_eq!(cache.stats().io_failures, 0, "not a payload I/O failure");
    assert_eq!(cache.calibration.as_ref().unwrap().saved, None);
    drop(cache);
    let mut cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "metal").unwrap();
    assert_eq!(cache.calibration(), None);
    assert!(cache.load([19; 32]));
    wait(&mut cache);
    assert_eq!(cache.get(&[19; 32]).unwrap().bytes(), &[42; 4096]);
    assert!(!cache.stats().tripped);
}

#[test]
fn ram_only_and_unmeasured_costs_never_write_metadata() {
    let dir = Temp::new();
    let mut cache = ColdCache::open_calibrated(
        ColdConfig {
            ram_bytes: 4 << 20,
            disk: None,
        },
        1 << 20,
        "metal",
    )
    .unwrap();
    cache.observe_cost(&measured());
    assert!(cache.calibration.is_none());
    drop(cache);
    let mut cache = ColdCache::open_calibrated(dir.config(), 1 << 20, "metal").unwrap();
    cache.observe_cost(&CostModel::new());
    assert!(cache.calibration.as_ref().unwrap().in_flight.is_none());
    drop(cache);
    assert!(!dir.0.join(FILE).exists());
}
