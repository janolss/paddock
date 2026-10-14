//! A tick that fails after its prefill span launched must leave no span in
//! flight. The overlapped spec tick used to return the round's error with the
//! span still marked in flight: the service failed the tick's slots, and its
//! next launch hit the "unified span already in flight" assert - one failed
//! round took the engine thread down. `unified_span_abandon` is the error
//! exit those ticks now take; this drives it on a real span and checks that
//! the slot's chunk can then be dropped and the slot serves again.
//!
//! Heavy GPU test: PADDOCK_HEAVY_TESTS=1, --release. `QWEN35_SMALL_GGUF`
//! names the file; defaults to LightOnOCR-3 0.8B's language model (qwen35).

mod common;

use paddock_engine::gpu_model::gpt_oss::GpuModelError;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

const SMALL: &[&str] = &[
    "LightOnOCR-3-0.8B-GGUF/LightOnOCR-3-0.8B-Q8_0.gguf",
    "LightOnOCR-3-0.8B-Q8_0.gguf",
];

#[test]
fn a_failed_tick_leaves_no_span_in_flight() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN35_SMALL_GGUF", SMALL) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load");
    m.enable_batch(2).expect("enable_batch");
    // several spans' worth at the budget below, so the first leaves work queued
    let prompt: Vec<u32> = (0..1500u32).map(|i| 1000 + i % 500).collect();

    m.prefill_begin(0, prompt.clone()).expect("queue");
    assert!(
        m.unified_span_launch(256, &[]).expect("launch"),
        "a queued prompt launches a span"
    );
    assert!(!m.prefill_abort(0), "a span in flight holds its chunks");
    let e = m.unified_span_abandon(GpuModelError::PoolExhausted);
    assert!(
        !matches!(e, GpuModelError::PoolExhausted),
        "exhaustion after the launch must not read as retryable: {e}"
    );
    assert!(m.unified_span_done(), "no span left in flight");
    assert!(m.prefill_abort(0), "the failed slot's chunk can be dropped");

    // the slot serves again: a new prompt queues and runs to its finish
    m.prefill_begin(0, prompt)
        .expect("the slot takes a new prompt");
    let mut finished = Vec::new();
    for _ in 0..64 {
        if !m.unified_span_launch(256, &[]).expect("launch") {
            break;
        }
        finished.extend(m.unified_span_finish().expect("finish"));
        if !finished.is_empty() {
            break;
        }
    }
    assert!(
        finished.iter().any(|(k, ..)| *k == 0),
        "the new prompt finishes"
    );
}
