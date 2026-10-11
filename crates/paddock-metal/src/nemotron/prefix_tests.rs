use super::*;
use paddock_engine::generator::Generator;
use std::{path::Path, time::Instant};

fn run_decode(model: &mut Nemotron) -> Vec<Vec<f32>> {
    (0..16).map(|i| model.forward(8000 + i).unwrap()).collect()
}

#[test]
#[ignore = "requires PADDOCK_NEMOTRON_TEST_MODEL; exact full-model retained-state checks"]
fn user_turn_and_interleaved_prefixes_preserve_all_logits() {
    let path = std::env::var("PADDOCK_NEMOTRON_TEST_MODEL").unwrap();
    for slots in [1, 4] {
        let mut model = Nemotron::load(Path::new(&path), 4096, slots, None).unwrap();
        let original: Vec<u32> = (1000..2537).collect();
        let mut branch = original[..1536].to_vec();
        branch.extend(9000..9041);
        model.prefill(0, &original).unwrap();
        let expected = model.prefill(0, &branch).unwrap();
        assert_eq!(model.take_prefill_reused(0), 1536);
        let continuation = run_decode(&mut model);
        let allocated = model.device.allocated_bytes();
        let cache = model.cache_bytes;
        // Same allocated model; the test-only legacy arm elects one snapshot
        // and no anchors. It is not a user setting or a looser arithmetic path.
        for legacy in [true, false, true, false] {
            model.reset();
            while model.evict_prefix() {}
            model.single_snapshot = legacy;
            model.prefill(0, &original).unwrap();
            model.anchor_at(0, &original, original.len() - 1);
            let mut tool = original.clone();
            let started = Instant::now();
            for turn in 0..8 {
                tool.extend(3000 + turn * 128..3128 + turn * 128);
                model.reset();
                model.prefill(0, &tool).unwrap();
            }
            let loop_ms = started.elapsed().as_secs_f64() * 1000.;
            model.reset();
            let started = Instant::now();
            let logits = model.prefill(0, &branch).unwrap();
            let ms = started.elapsed().as_secs_f64() * 1000.;
            let reused = model.take_prefill_reused(0);
            if legacy {
                assert_eq!(reused, 0);
            } else {
                assert_eq!(reused, 1536);
                assert_eq!(logits, expected, "all suffix logits after the tool loop");
                assert_eq!(
                    run_decode(&mut model),
                    continuation,
                    "all 16 continuation logit vectors"
                );
            }
            assert_eq!(model.device.allocated_bytes(), allocated);
            println!(
                "NEMOTRON_TURN slots={slots} legacy={legacy} reused={reused} prefill_ms={ms:.3} loop_ms={loop_ms:.3} allocated={allocated} cache_bytes={cache}"
            );
        }
        model.single_snapshot = false;
        model.reset();
        while model.evict_prefix() {}
        if slots == 4 {
            let prompts: Vec<Vec<u32>> = (0..4)
                .map(|s| (4000 + s * 1000..4513 + s * 1000).collect())
                .collect();
            let mut reference = Vec::new();
            for (s, p) in prompts.iter().enumerate() {
                reference.push(model.prefill(s, p).unwrap());
                model.anchor_at(s, p, p.len() - 1);
            }
            let mut decode = Vec::new();
            for i in 0..16 {
                decode.push(model.forward_batch(&[9000 + i; 4], &[513 + i; 4]).unwrap());
            }
            model.release_inactive_slots(&[]);
            for (s, p) in prompts.iter().enumerate() {
                assert_eq!(
                    model.prefill(s, p).unwrap(),
                    reference[s],
                    "all restored suffix logits slot {s}"
                );
                assert_eq!(model.take_prefill_reused(s), 512);
            }
            for i in 0..16 {
                assert_eq!(
                    model.forward_batch(&[9000 + i; 4], &[513 + i; 4]).unwrap(),
                    decode[i as usize],
                    "all c4 decode logits, step {i}"
                );
            }
            // Cancel one in-progress request, then release every live slot:
            // complete immutable checkpoints survive; no partial one publishes.
            let mut pending = prompts[0].clone();
            pending.extend(13000..13512);
            model.prefill_begin(0, pending).unwrap();
            model.forward_mixed(&[], 127).unwrap();
            assert!(model.prefill_abort(0));
            model.release_inactive_slots(&[]);
            for (s, p) in prompts.iter().enumerate() {
                model.prefill(s, p).unwrap();
                assert_eq!(model.take_prefill_reused(s), 512);
            }
            println!("NEMOTRON_PREFIX c=4 all_logits_steps=16 cancellation=true");
        }
        model.reset();
        while model.evict_prefix() {}
        assert_eq!(
            model.pool.free_blocks(),
            model.page_stride * slots * (1 + PREFIX_PER_SLOT)
        );
        assert_eq!(model.device.allocated_bytes(), allocated);
    }
}

#[test]
#[ignore = "requires PADDOCK_NEMOTRON_TEST_MODEL; GPU-only complete-state copy comparison"]
fn complete_state_blit_matches_compute_copy_and_preserves_guards() {
    let path = std::env::var("PADDOCK_NEMOTRON_TEST_MODEL").unwrap();
    let mut model = Nemotron::load(Path::new(&path), 1024, 1, None).unwrap();
    model.prefill(0, &(1000..1513).collect::<Vec<_>>()).unwrap();
    model.copy_state_kernel(1, Some(0)).unwrap();
    model.copy_state(2, Some(0)).unwrap();
    for layer in &model.layers {
        if let Mixer::Mamba(m) = &layer.mixer {
            for (buffer, n) in [(&m.state, STATE), (&m.window, WINDOW)] {
                // SAFETY: both full-state copies have completed their fences.
                unsafe {
                    assert_eq!(buffer.read_f32(n, n), buffer.read_f32(n * 2, n));
                    assert_eq!(buffer.read_f32(0, n), buffer.read_f32(n * 2, n));
                }
            }
        }
    }
    let mut compute = Vec::new();
    let mut blit = Vec::new();
    for round in 0..32 {
        for is_blit in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let dst = 1 + round % 2;
            let start = Instant::now();
            if is_blit {
                model.copy_state(dst, Some(0)).unwrap();
                blit.push(start.elapsed().as_secs_f64() * 1e6);
            } else {
                model.copy_state_kernel(dst, Some(0)).unwrap();
                compute.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    compute.sort_by(f64::total_cmp);
    blit.sort_by(f64::total_cmp);
    println!(
        "NEMOTRON_COPY compute_median_us={:.3} blit_median_us={:.3} bytes={}",
        compute[16],
        blit[16],
        23 * (STATE + WINDOW) * 4
    );
    assert!(
        model.copy_state(3, Some(0)).is_err(),
        "reject out-of-range before submitting any copy"
    );
}
