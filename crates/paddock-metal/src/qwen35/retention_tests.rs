use super::*;
use paddock_engine::generator::Generator;
use std::{path::Path, time::Instant};

// Keep failures readable: a whole Qwen vocabulary can exceed a million log
// characters. This remains an exact comparison, not a numerical tolerance.
fn exact_logits(actual: &[f32], expected: &[f32], stage: &str) {
    assert_eq!(actual.len(), expected.len(), "{stage}: logit count");
    let different = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    let max_delta = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(different, 0, "{stage}: max_delta={max_delta}");
}

fn recurrent_snapshot(model: &Qwen35, boundary: usize) -> Vec<f32> {
    let index = model
        .cache
        .iter()
        .position(|c| c.history.len() == boundary)
        .unwrap();
    let slot = model.slots.len() + index;
    let mut values = Vec::new();
    for layer in 0..model.geometry.linear_layers() {
        for (buffer, size) in [
            (&model.state, model.geometry.state()),
            (&model.conv, model.geometry.conv() * 3),
        ] {
            // SAFETY: synchronous prefill/decode fenced every GPU submission.
            values.extend(unsafe {
                buffer.read_f32((layer * model.state_slots + slot) * size, size)
            });
        }
    }
    values
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL; full-model user-turn retention"]
fn user_turn_anchor_survives_tool_loop_with_exact_continuation() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
    for slots in [1, 4] {
        check_tool_loop(Path::new(&path), slots);
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL; cache-independent recurrent arithmetic"]
fn reserved_cache_cannot_change_prefill_arithmetic() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 3072, 1, None).unwrap();
    let prompt: Vec<u32> = (1000..2577).collect();
    let reference = model.forward_prefill(0, &prompt).unwrap();
    let continuation = model.forward(10000).unwrap();
    model.reset();
    while model.evict_checkpoint() {}
    // Every destination can belong to an asynchronous cold restore. No
    // optional capture may steal it, or alter the live recurrent contraction.
    for checkpoint in &mut model.cache {
        checkpoint.reserved = true;
    }
    let logits = model.forward_prefill(0, &prompt).unwrap();
    exact_logits(&logits, &reference, "all snapshot destinations reserved");
    assert!(
        model
            .cache
            .iter()
            .all(|c| c.reserved && c.history.is_empty())
    );
    exact_logits(
        &model.forward(10000).unwrap(),
        &continuation,
        "reserved continuation",
    );
}

fn check_tool_loop(path: &Path, slots: usize) {
    let mut model = Qwen35::load(path, 3072, slots, None).unwrap();
    let original: Vec<u32> = (1000..2537).collect();
    let boundary = 1536;
    // A later user turn re-renders the conversation from the first reply on.
    // All tokens through the retained user prompt still match.
    let mut branch = original[..boundary].to_vec();
    branch.extend(9000..9041);
    let decode = |model: &mut Qwen35| {
        (0..16)
            .map(|i| model.forward(10000 + i).unwrap())
            .collect::<Vec<_>>()
    };

    let original_logits = model.forward_prefill(0, &original).unwrap();
    let snapshot = recurrent_snapshot(&model, boundary);
    let reference = model.forward_prefill(0, &branch).unwrap();
    assert_eq!(model.take_prefill_reused(0), boundary);
    let reference_decode = decode(&mut model);
    let mut measurements = Vec::new();
    // Alternate the prior policy (no user-turn signal) and the new policy.
    // Same model, same tool prompts, no competing endpoint or checkpoint files.
    for protected in [false, true, false, true] {
        model.reset();
        while model.evict_checkpoint() {}
        let repeated = model.forward_prefill(0, &original).unwrap();
        exact_logits(&repeated, &original_logits, "repeated cold prompt");
        exact_logits(
            &recurrent_snapshot(&model, boundary),
            &snapshot,
            "repeated cold checkpoint",
        );
        let allocated = model.device.allocated_bytes();
        if protected {
            model.anchor_at(0, &original, original.len() - 1);
            assert!(
                model
                    .cache
                    .iter()
                    .any(|c| c.anchor && c.history.len() == boundary)
            );
            assert_eq!(model.device.allocated_bytes(), allocated);
        }
        let mut tool_prompt = original.clone();
        let tool_started = Instant::now();
        for turn in 0..8 {
            tool_prompt.extend((3000 + turn * 64)..(3064 + turn * 64));
            model.reset();
            model.forward_prefill(0, &tool_prompt).unwrap();
            assert!(model.take_prefill_reused(0) >= boundary);
            if protected {
                assert!(
                    model
                        .cache
                        .iter()
                        .any(|c| c.anchor && c.history.len() == boundary)
                );
            }
        }
        let tool_ms = tool_started.elapsed().as_secs_f64() * 1000.;
        if protected {
            exact_logits(
                &recurrent_snapshot(&model, boundary),
                &snapshot,
                "checkpoint after tool loop",
            );
        }
        assert_eq!(
            model.device.allocated_bytes(),
            allocated,
            "no extra GPU capacity"
        );
        model.reset();
        let started = Instant::now();
        let logits = model.forward_prefill(0, &branch).unwrap();
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        let reused = model.take_prefill_reused(0);
        if protected {
            assert_eq!(reused, boundary);
            exact_logits(
                &logits,
                &reference,
                "every suffix logit after the tool loop",
            );
            for (step, (actual, expected)) in
                decode(&mut model).iter().zip(&reference_decode).enumerate()
            {
                exact_logits(actual, expected, &format!("continuation step {step}"));
            }
        } else {
            assert_eq!(
                reused, 0,
                "the prior retention policy lost the user-turn prefix"
            );
        }
        println!(
            "METAL_USER_TURN slots={slots} anchored={protected} reused={reused} ttft_ms={elapsed:.3} tool_loop_ms={tool_ms:.3} gpu_bytes={allocated}"
        );
        measurements.push((protected, elapsed));
    }
    // Timing is diagnostic, not a fragile CI threshold. Correctness and
    // actual avoided work are the assertions above.
    println!("METAL_USER_TURN_TIMES slots={slots} {measurements:?}");
    if slots == 4 {
        model.reset();
        while model.evict_checkpoint() {}
        let mut prompts: Vec<Vec<u32>> = (0..4)
            .map(|s| (4000 + s * 1000..4257 + s * 1000).collect())
            .collect();
        for (s, prompt) in prompts.iter().enumerate() {
            model.forward_prefill(s, prompt).unwrap();
            model.anchor_at(s, prompt, prompt.len() - 1);
        }
        for turn in 0..3 {
            for (s, prompt) in prompts.iter_mut().enumerate() {
                let prior_len = prompt.len();
                prompt.extend(12000 + turn * 64..12064 + turn * 64);
                model.forward_prefill(s, prompt).unwrap();
                assert!(
                    model.take_prefill_reused(s) >= prior_len - 33,
                    "anchors cannot crowd out another active conversation's trailing cuts"
                );
            }
        }
        println!("METAL_USER_TURN_INTERLEAVED slots=4 rounds=3 retained_all_trailing_cuts=true");
    }
}
