//! The overlap scheduler's admission knobs and gate diagnostics - what
//! decides whether a mixed tick (decode riders aboard a prefill span) runs
//! on the decode pipe beside the span or fuses into one unified tick.

use super::Slot;
use crate::generator::Generator;

/// Live-REQUEST floor (slots holding a request, chunking or decoding) for
/// the two side-stream decode paths - the OVERLAP block (span-only prefill
/// ticks on the main lane, decoders pumped on the pipe lane) and the
/// classic pure-decode pipe. Below it the decode rows ride the unified
/// prefill pass and ticks run one at a time. 0 = the paths engage at any
/// width (the 188-SM die, where both were measured to win). The qwen35
/// default stack sets 9 on dies under 128 SMs: on GB10 (LPDDR5X, 48 SMs) a
/// decode tick is ~100 ms of weight streaming, so at 8 slots the two lanes
/// just time-slice the same bandwidth - 1024x1024 c8 TTFT p50 2698 -> 2124
/// ms and 128x128 c1 223 -> 136 ms with the paths off, throughput flat or
/// up - while from 16 concurrent up the overlap wins the long cohort
/// (1024x1024 c16 128.9 vs 126 tok/s, TTFT p50 7.3 vs 10 s; c32 205 vs
/// 199-202, 12.4 vs 14-16 s). It is counted in requests, not decoders: the
/// cohort's loss is in its EARLY ticks, so gating on decoders (a floor of
/// 9 or 16 decoders) still lost c16/c32 - the overlap has to own the cohort
/// from its first decoder, which a request count gives. PADDOCK_PIPE_MIN_LIVE
/// overrides.
pub(super) fn pipe_min_live() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_PIPE_MIN_LIVE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// PADDOCK_OVERLAP_OVER_SPEC=1: overlap outranks spec-first on a tick the
/// pipe can take. The spec-in-mixed block gives each rider ONE round per
/// span, so a rider's token gap is the whole span (1.4 s on a 27B at the
/// 4096-row span); the pipe pumps decode ticks for the span's whole life
/// and the slot re-warms its drafter on the next spec tick. Measured on
/// Qwen3.8-27B (12 Claude-Code-shaped sessions, 2 slots): riders got 3-5
/// pipe ticks per span instead of one round, each pipe tick ~250 ms beside
/// the saturating pass, and the fat span stretched. A latency trade; see
/// the A/B in the commit that added it.
pub(super) fn spec_first_after_overlap(spec_mixed_first: bool, overlap_core: bool) -> bool {
    spec_mixed_first
        && !(overlap_core && paddock_models::dev_var_os!("PADDOCK_OVERLAP_OVER_SPEC").is_some())
}

/// Why a mixed tick with riders fused instead of overlapping (debug: one
/// line per such tick), every gate's value and each rider's plannability.
pub(super) fn log_fused_mixed_tick(
    dec: &[(usize, u32, u32)],
    slots: &[Option<Slot>],
    generator: &dyn Generator,
    unified_ok: bool,
    spec_mixed_first: bool,
    live_slots: usize,
    trunc_pipe_ok: bool,
) {
    let rows: Vec<(bool, bool, bool, bool, bool)> = dec
        .iter()
        .map(|&(k, _, _)| {
            let s = slots[k].as_ref().expect("live decode row");
            (
                s.constraint.is_none(),
                s.logprobs.is_none(),
                !s.sampler.in_reasoning(),
                s.sampler.is_device_plannable(),
                s.sampler.is_trunc_plannable(),
            )
        })
        .collect();
    tracing::debug!(
        unified_ok,
        spec_mixed_first,
        supports_overlap = generator.supports_overlap(),
        live_slots,
        pipe_min_live = pipe_min_live(),
        pool_free = ?generator.pool_free_blocks(),
        trunc_pipe_ok,
        ?rows,
        "overlap gate: fused mixed tick (no_constraint, no_logprobs, not_reasoning, plannable, trunc_plannable)"
    );
}

/// The pipe loop's exit and why (debug).
pub(super) fn log_pipe_exit(
    generator: &dyn Generator,
    ovl_ticks: usize,
    low_headroom: bool,
    all_dead: bool,
    near_end: bool,
) {
    tracing::debug!(
        ovl_ticks,
        span_done = generator.unified_span_done(),
        low_headroom,
        all_dead,
        near_end,
        pool_free = ?generator.pool_free_blocks(),
        "overlap: pipe loop exit"
    );
}
