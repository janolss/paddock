use super::*;

#[test]
fn mlx_workspace_covers_cold_arena_and_slot_bounded_decode() {
    let sizes = Scratch::sizes_for_rows(64, 32768 / 16, MLX_CHUNK, true);
    assert_eq!(
        sizes.iter().sum::<usize>() + WIDTH * 128 * 2 + 4,
        1_224_280_584
    );
    for batch in [1, 4, 64] {
        let sizes = Scratch::sizes_for_rows(batch, 2048, MLX_CHUNK, true);
        assert_eq!(sizes[24], batch * 32 * SPLITS * 130 * 4);
        assert!(sizes[11] >= MLX_CHUNK * 6 * WIDTH * 2);
    }
    assert_eq!(crate::schedule::row_cap(3, MLX_CHUNK), 128);
    assert_eq!(crate::schedule::row_cap(0, MLX_CHUNK), 2048);
    for decodes in [1usize, 3, 4, 64] {
        assert_eq!(mixed_row_cap(true, decodes, MLX_CHUNK) - decodes, 512);
        assert_eq!(mixed_row_cap(false, decodes, MLX_CHUNK), 128);
        for capacity in [0, 1, 64, 128, 512, 2048] {
            assert!(mixed_row_cap(true, decodes, capacity) <= capacity);
        }
    }
    assert_eq!(mixed_row_cap(true, 0, MLX_CHUNK), MLX_CHUNK);
    assert_eq!(mixed_row_cap(true, usize::MAX, MLX_CHUNK), MLX_CHUNK);
}
use paddock_engine::generator::Generator;

#[test]
#[ignore = "requires Apple10 GPU; paged head-128 register attention bounds"]
fn mlx_register_attention_causal_pages_and_ragged_tails() {
    let device = MetalDevice::new(None).unwrap();
    if !device.tensor_accelerated() {
        return;
    }
    let (heads, kv_heads, hd) = (4usize, 2usize, 128usize);
    for (prefix, rows) in [
        (0usize, 1usize),
        (0, 17),
        (0, 33),
        (15, 17),
        (31, 33),
        (511, 17),
    ] {
        let context = prefix + rows;
        let stride = context.div_ceil(16);
        let pages = (0..stride * 2)
            .map(|i| (stride * 2 - 1 - i) as u32)
            .collect::<Vec<_>>();
        let sample = |i: usize, seed: usize| {
            half::bf16::from_f32(((i * seed % 137) as f32 - 68.) / 64.).to_f32()
        };
        let queries = (0..2 * rows * heads * hd)
            .map(|i| sample(i, 17))
            .collect::<Vec<_>>();
        let keys = (0..2 * stride * 16 * kv_heads * hd)
            .map(|i| sample(i, 31))
            .collect::<Vec<_>>();
        let values = (0..keys.len()).map(|i| sample(i, 47)).collect::<Vec<_>>();
        let upload_bf = |values: &[f32]| {
            device
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        let upload_u32 = |values: &[u32]| {
            device
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        let q = upload_bf(&queries);
        let k = upload_bf(&keys);
        let v = upload_bf(&values);
        let page_buffer = upload_u32(&pages);
        let metadata = (0..2)
            .flat_map(|s| (0..rows).flat_map(move |r| [s, (prefix + r) as u32]))
            .collect::<Vec<_>>();
        let meta = upload_u32(&metadata);
        let tiles = (0..2)
            .flat_map(|s| {
                (0..rows)
                    .step_by(32)
                    .flat_map(move |r| [(s * rows + r) as u32, (rows - r).min(32) as u32])
            })
            .collect::<Vec<_>>();
        let tile_buffer = upload_u32(&tiles);
        let output = device
            .upload(&vec![0xff; (queries.len() + 32) * 4])
            .unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "nemo_mlx_prefill_nax",
            &[&q, &k, &v, &meta, &page_buffer, &output, &tile_buffer],
            &[heads as u32, kv_heads as u32, stride as u32],
            [heads, tiles.len() / 2, 1],
            64,
        );
        cmd.finish().unwrap();
        let actual = unsafe { output.read_f32(0, queries.len() + 32) };
        assert!(
            actual[queries.len()..]
                .iter()
                .all(|v| v.to_bits() == u32::MAX)
        );
        // Independent scalar softmax is a test oracle only. Runtime never
        // reads GPU activations back or substitutes reference results.
        for slot in 0..2 {
            for row in 0..rows {
                for head in 0..heads {
                    let qbase = ((slot * rows + row) * heads + head) * hd;
                    let kv_at = |pos: usize| {
                        ((pages[slot * stride + pos / 16] as usize * 16 + pos % 16) * kv_heads
                            + head / (heads / kv_heads))
                            * hd
                    };
                    let scores = (0..=prefix + row)
                        .map(|pos| {
                            let at = kv_at(pos);
                            (0..hd)
                                .map(|d| queries[qbase + d] as f64 * keys[at + d] as f64)
                                .sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let high = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probs = scores.iter().map(|v| (v - high).exp()).collect::<Vec<_>>();
                    let den: f64 = probs.iter().sum();
                    for d in 0..hd {
                        let expected = probs
                            .iter()
                            .enumerate()
                            .map(|(pos, p)| p * values[kv_at(pos) + d] as f64)
                            .sum::<f64>()
                            / den;
                        let got = actual[qbase + d] as f64;
                        assert!(
                            got.is_finite() && (got - expected).abs() < 0.0025,
                            "prefix={prefix} rows={rows} slot={slot} row={row} head={head} d={d}: {got} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint; complete four-conversation cohort rotation"]
fn mlx_new_cohorts_retain_all_four_current_prefixes() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 2048, 4, None).unwrap();
    let allocated = model.device.allocated_bytes();
    for (cohort, length) in [513, 1025, 1537].into_iter().enumerate() {
        model.release_inactive_slots(&[]);
        let prompts = (0..4)
            .map(|slot| {
                (0..length)
                    .map(|i| 100 + (cohort * 10000 + slot * 2000 + i) as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (slot, prompt) in prompts.iter().enumerate() {
            model.prefill_begin(slot, prompt.clone()).unwrap();
        }
        while !model.pending.is_empty() {
            let (_, complete) = model.forward_mixed(&[], 509).unwrap();
            for (slot, logits, _) in complete {
                assert!(logits.iter().all(|v| v.is_finite()));
                model.anchor_at(slot, &prompts[slot], length - 1);
            }
        }
        model.release_inactive_slots(&[]);
        assert_eq!(model.prefixes.iter().filter(|c| c.anchor).count(), 4);
        for (slot, prompt) in prompts.iter().enumerate() {
            let mut next = prompt.clone();
            next.push(9000);
            assert_eq!(
                model.prepare(slot, &next).unwrap(),
                length,
                "cohort {cohort}, slot {slot}"
            );
            assert_eq!(
                model.prefixes.iter().filter(|c| c.anchor).count(),
                4,
                "warm tool turns must preserve idle peers' user-turn cuts"
            );
        }
        assert_eq!(model.device.allocated_bytes(), allocated);
    }
    model.reset();
    while model.evict_prefix() {}
    assert_eq!(model.pool.free_blocks(), model.pool.capacity() as usize);
}

#[test]
#[ignore = "requires pinned MLX checkpoint; partial-page COW and complete-state restore"]
fn mlx_partial_prefix_preserves_divergent_continuations() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 2048, 4, None).unwrap();
    let allocated = model.device.allocated_bytes();
    for length in [17, 33, 127, 513] {
        model.reset();
        while model.evict_prefix() {}
        let prompt = (0..length)
            .map(|i| 100 + (i % 41) as u32)
            .collect::<Vec<_>>();
        model.prefill(0, &prompt).unwrap();
        let index = model
            .prefixes
            .iter()
            .position(|c| c.history == prompt)
            .unwrap();
        let page = *model.prefixes[index].table.blocks().last().unwrap();
        let page_bits = |model: &Nemotron| {
            model
                .layers
                .iter()
                .filter_map(|l| match &l.mixer {
                    Mixer::Attention(a) => Some(a),
                    _ => None,
                })
                .flat_map(|a| [&a.keys, &a.values])
                .flat_map(|b| {
                    // GPU fences complete before snapshots are inspected. Read
                    // BF16 storage as opaque pairs, not interpreted F32 numbers.
                    unsafe { b.read_f32(page as usize * BLOCK_TOKENS * 128, BLOCK_TOKENS * 128) }
                        .into_iter()
                        .map(f32::to_bits)
                })
                .collect::<Vec<_>>()
        };
        let immutable = page_bits(&model);
        let mut expected = Vec::new();
        for step in 0..8 {
            expected.push(
                model
                    .execute(&[(0, 8000 + step, length as u32 + step)], &[0])
                    .unwrap(),
            );
        }
        assert_ne!(model.slots[0].table.blocks()[length / BLOCK_TOKENS], page);
        assert_eq!(
            page_bits(&model),
            immutable,
            "live append changed retained KV"
        );
        // Two restores share the same partial page, then diverge. Neither may
        // mutate the prefix or the other continuation's newly private page.
        for slot in [1, 2] {
            let mut full = prompt.clone();
            full.push(8000);
            assert_eq!(model.prepare(slot, &full).unwrap(), length);
        }
        for step in 0..8 {
            let logits = model
                .execute(
                    &[
                        (1, 8000 + step, length as u32 + step),
                        (2, 9000 + step, length as u32 + step),
                    ],
                    &[0, 1],
                )
                .unwrap();
            assert_eq!(
                &logits[..VOCAB],
                expected[step as usize],
                "restored step {step}"
            );
        }
        assert_eq!(
            page_bits(&model),
            immutable,
            "divergent restore changed retained KV"
        );
        assert_eq!(model.device.allocated_bytes(), allocated);
    }
    model.reset();
    while model.evict_prefix() {}
    assert_eq!(model.pool.free_blocks(), model.pool.capacity() as usize);
}

#[test]
#[ignore = "requires independent MLX operation capture; arithmetic diagnostic"]
fn mlx_projection_reference_arithmetic() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixtures = std::env::var("PADDOCK_NEMOTRON_MLX_REFERENCE").unwrap();
    let model = Nemotron::load(std::path::Path::new(&path), 4096, 1, None).unwrap();
    let Mixer::Mamba(w) = &model.layers[0].mixer else {
        unreachable!()
    };
    for i in 0..3 {
        let f = paddock_models::safetensors::SafetensorsFile::open(
            &std::path::Path::new(&fixtures).join(format!("case-{i}.safetensors")),
        )
        .unwrap();
        for (stage, x_name, y_name, weight) in [
            ("input", "norm", "proj", &w.input),
            ("output", "yn", "delta", &w.output),
        ] {
            let (_, x) = f.bytes(x_name).unwrap();
            let (_, y) = f.bytes(y_name).unwrap();
            let rows = x.len() / 4 / weight.k;
            let input = model.device.upload(x).unwrap();
            let output = model.device.alloc(y.len()).unwrap();
            let expected = y
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let mut candidates = vec![
                ("nemo_mlx_project_exact", 32, 16, 1, 0),
                ("nemo_mlx_project", 32, 16, 1, 0),
                ("nemo_mlx_vector4", 4, 4, 1, 0),
            ];
            if stage == "output" {
                for parts in [2, 4, 8, 16, 32, 64] {
                    for round in [0, 1] {
                        candidates.push(("nemo_mlx_splitk_reference", 32, 16, parts, round));
                    }
                }
            }
            for (name, columns, tile, split, round) in candidates {
                let mut parts = (512 / (weight.n.div_ceil(32) * rows.div_ceil(32)))
                    .max(1)
                    .min(weight.k / 64);
                while parts > 1 && !weight.k.is_multiple_of(parts * 64) {
                    parts -= 1;
                }
                if name == "nemo_mlx_splitk_reference" {
                    parts = split;
                }
                let cmd = model.device.begin().unwrap();
                cmd.dispatch(
                    name,
                    &[&weight.buffer, &input, &output],
                    &[
                        weight.k as u32,
                        weight.n as u32,
                        rows as u32,
                        mlx::AFFINE8,
                        1,
                        parts as u32,
                        round,
                    ],
                    [weight.n.div_ceil(columns), rows.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                let actual = unsafe { output.read_f32(0, expected.len()) };
                assert!(actual.iter().all(|v| v.is_finite()));
                let different = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "NEMO_ARITH case={i} stage={stage} rows={rows} parts={parts} round={round} kernel={name} different={different}/{} max_abs={error}",
                    actual.len()
                );
            }
        }
    }
}

#[test]
#[ignore = "requires checkpoint and captured SDK briefing; profiling is not a serving result"]
fn mlx_captured_long_prompt_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixture = std::env::var("PADDOCK_NEMOTRON_PROMPT_FILE").unwrap();
    let batch: usize = std::env::var("PADDOCK_NEMOTRON_COST_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!([1, 4].contains(&batch));
    let tokenizer =
        paddock_tokenizer::GgufTokenizer::from_hf_dir(std::path::Path::new(&path)).unwrap();
    let tokens = tokenizer
        .encode(&std::fs::read_to_string(fixture).unwrap())
        .unwrap();
    assert!((4096..=32768).contains(&tokens.len()));
    let mut model = Nemotron::load(std::path::Path::new(&path), 32768, batch, None).unwrap();
    let chunks = std::env::var("PADDOCK_NEMOTRON_COST_CHUNK").map_or_else(
        |_| vec![512, 1024, 2048],
        |v| vec![v.parse::<usize>().unwrap()],
    );
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for chunk in chunks {
        assert!([512, 1024, 2048].contains(&chunk));
        model.reset();
        while model.evict_prefix() {}
        model.scratch =
            Scratch::new_for_rows(&model.device, batch, model.page_stride, chunk, true).unwrap();
        let used = model.device.allocated_bytes();
        let start = std::time::Instant::now();
        let step = chunk / batch;
        let mut results = Vec::new();
        for first in (0..tokens.len()).step_by(step) {
            let end = (first + step).min(tokens.len());
            let rows = (0..batch)
                .flat_map(|s| {
                    let tokens = &tokens;
                    (first..end).map(move |i| (s, tokens[i], i as u32))
                })
                .collect::<Vec<_>>();
            let out = if end == tokens.len() {
                (1..=batch)
                    .map(|s| s * (end - first) - 1)
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            let chunk = std::time::Instant::now();
            let logits = model.execute(&rows, &out).unwrap();
            assert!(logits.iter().all(|v| v.is_finite()));
            if !out.is_empty() {
                results.push(logits);
            }
            eprintln!(
                "NEMO_LONG c={batch} first={first} end={end} wall_ms={:.3} gpu_ms={:.3}",
                chunk.elapsed().as_secs_f64() * 1000.,
                model.last_gpu_seconds * 1000.
            );
        }
        eprintln!(
            "NEMO_LONG total c={batch} chunk={chunk} tokens={} seconds={:.3}",
            tokens.len(),
            start.elapsed().as_secs_f64()
        );
        for layer in &model.layers {
            if let Mixer::Mamba(m) = &layer.mixer {
                results.push(unsafe { m.state.read_f32(0, batch * STATE) });
                results.push(unsafe { m.window.read_f32(0, batch * WINDOW) });
            }
        }
        if let Some(expected) = &reference {
            assert!(
                results
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk={chunk}: full logits/carried-state differs"
            );
        } else {
            reference = Some(results);
        }
        assert_eq!(model.device.allocated_bytes(), used);
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint; unrelated admissions must not change a live decode"]
fn mlx_live_decode_is_independent_of_prompt_admission() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let used = model.device.allocated_bytes();
    for admitted in [1, 17, 65, 129, 257] {
        let mut run = |mixed: bool| {
            model.reset();
            while model.evict_prefix() {}
            let prompt = (0..33).map(|i| (0, 100 + i * 17, i)).collect::<Vec<_>>();
            model.execute(&prompt, &[32]).unwrap();
            let mut results = Vec::new();
            for step in 0..4 {
                // Exercise a warm decode both before and after the prompt in
                // the packed row order. No special "first row" exemption.
                let mut rows = vec![(0, 180 + step, 33 + step)];
                if mixed {
                    let other = (0..admitted)
                        .map(|i| (1, 300 + (i * 11 + step * 7) % 1000, step * admitted + i));
                    rows.extend(other);
                    if step % 2 != 0 {
                        rows.rotate_left(1);
                    }
                }
                let output = rows.iter().position(|r| r.0 == 0).unwrap();
                results.push(model.execute(&rows, &[output]).unwrap());
                for layer in &model.layers {
                    if let Mixer::Mamba(m) = &layer.mixer {
                        results.push(unsafe { m.state.read_f32(0, STATE) });
                        results.push(unsafe { m.window.read_f32(0, WINDOW) });
                    }
                }
            }
            results
        };
        let expected = run(false);
        let actual = run(true);
        for (stage, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "admitted={admitted}, stage={stage}: live logits/state changed"
            );
        }
        assert_eq!(model.device.allocated_bytes(), used);
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint; bit-identical pre-optimization GPU witness"]
fn mlx_optimized_graph_preserves_reference_logits_and_state() {
    // Preserve the historical arithmetic witness as a diagnostic. The default
    // BF16 graph is qualified separately against the independent MLX capture.
    crate::projection::with_affine_legacy(mlx_legacy_graph_preserves_reference);
}
fn mlx_legacy_graph_preserves_reference() {
    mlx_graph_witness(false);
}
#[test]
#[ignore = "real checkpoint: packed tensor experts must preserve all logits and carried state"]
fn mlx_tensor_experts_preserve_graph() {
    mlx_graph_witness(true);
}
fn mlx_graph_witness(packed: bool) {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let used = model.device.allocated_bytes();
    for batch in [1, 4] {
        for &length in if packed {
            &[32, 33, 128, 129, 257, 511, 512][..]
        } else {
            &[33, 128][..]
        } {
            let rows = (0..batch)
                .flat_map(|s| {
                    (0..length).map(move |i| (s, 100 + ((i * 17 + s * 29) % 1000) as u32, i as u32))
                })
                .collect::<Vec<_>>();
            let out = (1..=batch).map(|i| i * length - 1).collect::<Vec<_>>();
            let mut run = || {
                model.reset();
                while model.evict_prefix() {}
                let mut results = vec![model.execute(&rows, &out).unwrap()];
                for i in 0..8 {
                    let decodes = (0..batch)
                        .map(|s| (s, 180 + i as u32, (length + i) as u32))
                        .collect::<Vec<_>>();
                    results.push(
                        model
                            .execute(&decodes, &(0..batch).collect::<Vec<_>>())
                            .unwrap(),
                    );
                }
                for layer in &model.layers {
                    if let Mixer::Mamba(m) = &layer.mixer {
                        // execute() has completed its GPU fence.
                        results.push(unsafe { m.state.read_f32(0, batch * STATE) });
                        results.push(unsafe { m.window.read_f32(0, batch * WINDOW) });
                    }
                }
                results
            };
            let expected = if packed {
                crate::projection::with_unpacked_experts(&mut run)
            } else {
                crate::projection::with_affine_reference(&mut run)
            };
            let actual = run();
            assert!(
                actual
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "full logits and carried state: c={batch}, length={length}"
            );
            assert_eq!(model.device.allocated_bytes(), used);
        }
    }
}

#[test]
#[ignore = "requires Metal GPU; affine8 contraction and guarded-tail regression"]
fn mlx_projection_tiles_and_reused_vectors() {
    let d = MetalDevice::new(None).unwrap();
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    for k in [64usize, 256, 2688, 4096] {
        let n = 33usize;
        let mut bytes = (0..k * n)
            .map(|i| ((i * 29 + i / k * 7) % 256) as u8)
            .collect::<Vec<_>>();
        bytes.extend(
            (0..k * n / 64)
                .flat_map(|i| half::bf16::from_f32((i % 7 + 1) as f32 / 4096.).to_le_bytes()),
        );
        bytes.extend((0..k * n / 64).flat_map(|i| {
            half::bf16::from_f32(-((i % 7 + 1) as f32) * 127. / 4096.).to_le_bytes()
        }));
        let weights = d.upload(&bytes).unwrap();
        for rows in [1usize, 2, 4, 7, 17, 31, 65] {
            let values = (0..rows * k)
                .map(|i| bf(((i * 17 % 997) as f32 - 498.) / 139.))
                .collect::<Vec<_>>();
            let input = d
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let packed = d.upload(&vec![0xff; rows * k * 2 + 64]).unwrap();
            let count = rows * n + 32;
            let guard = vec![-12345f32; count];
            let mut outputs = Vec::new();
            for (name, columns, tile, bf_input) in if rows < 16 {
                vec![
                    ("dg_affine_vector", 4, 1, false),
                    ("nemo_mlx_vector1", 4, 1, false),
                    ("nemo_mlx_vector4", 4, 4, false),
                    ("nemo_mlx_vector1_reuse", 8, 1, false),
                    ("nemo_mlx_vector4_reuse", 8, 4, false),
                ]
            } else {
                vec![
                    ("dg_project", 32, 16, false),
                    ("nemo_mlx_project_exact", 32, 16, false),
                    ("nemo_mlx_project_exact_wide", 64, 32, false),
                    ("nemo_mlx_project_exact_mixed", 64, 32, false),
                    ("nemo_mlx_project_exact_large", 64, 64, false),
                    ("nemo_mlx_project_cooperative", 64, 32, false),
                    ("nemo_mlx_project", 32, 16, false),
                    ("nemo_mlx_project_wide", 64, 32, false),
                    ("nemo_mlx_project_reuse_baseline", 64, 64, true),
                    ("nemo_mlx_project_reuse", 64, 64, true),
                    ("nemo_mlx_project_reuse128", 64, 128, true),
                ]
            } {
                let output = d
                    .upload(
                        &guard
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let cmd = d.begin().unwrap();
                if bf_input {
                    cmd.dispatch(
                        "mlx_input_compact",
                        &[&input, &packed],
                        &[k as u32, rows as u32],
                        [(k * rows).div_ceil(256), 1, 1],
                        256,
                    );
                }
                cmd.dispatch(
                    name,
                    &[&weights, if bf_input { &packed } else { &input }, &output],
                    &[k as u32, n as u32, rows as u32, mlx::AFFINE8, 1],
                    [n.div_ceil(columns), rows.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                let actual = unsafe { output.read_f32(0, count) }.to_vec();
                assert_eq!(
                    &actual[rows * n..],
                    &guard[rows * n..],
                    "{name}: output guard"
                );
                assert!(actual.iter().all(|v| v.is_finite()));
                if name == "nemo_mlx_project_reuse" || name == "nemo_mlx_project_reuse128" {
                    assert_eq!(
                        Some(&actual),
                        outputs.last(),
                        "metadata reuse: k={k}, rows={rows}"
                    );
                }
                if (rows < 16
                    || name.starts_with("nemo_mlx_project_exact")
                    || name == "nemo_mlx_project_cooperative")
                    && !outputs.is_empty()
                {
                    assert_eq!(
                        actual, outputs[0],
                        "exact decode contraction: {name}, k={k}, rows={rows}"
                    );
                }
                if rows >= 16 {
                    // Independent F64 dot over BF16 operands. The conservative
                    // BF16 rounding envelope includes F32 midpoint/cancellation
                    // error; it does not replace the exact native-route gate.
                    for r in 0..rows {
                        for col in 0..n {
                            let want = (0..k)
                                .map(|j| {
                                    let at = col * k + j;
                                    let group = at / 64;
                                    let scale = bf((group % 7 + 1) as f32 / 4096.);
                                    let bias = bf(-((group % 7 + 1) as f32) * 127. / 4096.);
                                    f64::from(bf(bytes[at] as f32 * scale + bias))
                                        * f64::from(values[r * k + j])
                                })
                                .sum::<f64>() as f32;
                            let error = (actual[r * n + col] - bf(want)).abs();
                            assert!(
                                error <= want.abs() / 128. + 2e-5,
                                "{name} k={k} rows={rows} r={r} n={col}: {error}, want={want}"
                            );
                        }
                    }
                }
                outputs.push(actual);
            }
        }
    }
}

#[test]
#[ignore = "requires Metal GPU; router row/column reuse and guarded tails"]
fn mlx_router_reuse_preserves_f32_reduction() {
    let d = MetalDevice::new(None).unwrap();
    for k in [63usize, 64, 256, 2688] {
        let n = 33usize;
        let weights = d
            .upload(
                &((0..k * n)
                    .flat_map(|i| (((i * 17 % 509) as f32 - 254.) / 131.).to_le_bytes())
                    .collect::<Vec<_>>()),
            )
            .unwrap();
        for rows in [1usize, 2, 4, 7, 17, 33] {
            let input = d
                .upload(
                    &((0..rows * k)
                        .flat_map(|i| (((i * 29 % 997) as f32 - 498.) / 139.).to_le_bytes())
                        .collect::<Vec<_>>()),
                )
                .unwrap();
            let guard = vec![-12345f32; rows * n + 32];
            let mut outputs = Vec::new();
            for (name, columns, tile) in [("linear", 4, 1), ("nemo_mlx_router4", 8, 4)] {
                let output = d
                    .upload(
                        &guard
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    name,
                    &[&weights, &input, &output],
                    &[k as u32, n as u32, rows as u32, 0, 1f32.to_bits()],
                    [n.div_ceil(columns), rows.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                let actual = unsafe { output.read_f32(0, guard.len()) }.to_vec();
                assert_eq!(
                    &actual[rows * n..],
                    &guard[rows * n..],
                    "{name}: output guard"
                );
                assert!(actual.iter().all(|v| v.is_finite()));
                outputs.push(actual.into_iter().map(f32::to_bits).collect::<Vec<_>>());
            }
            assert_eq!(outputs[0], outputs[1], "k={k}, rows={rows}");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU; diagnostic paired router timing, not serving qualification"]
fn mlx_router_reuse_costs() {
    let d = MetalDevice::new(None).unwrap();
    let (k, n) = (2688usize, 128usize);
    let weights = d
        .upload(
            &(0..k * n)
                .flat_map(|i| (((i * 17 % 509) as f32 - 254.) / 131.).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for rows in [4usize, 32, 128, 512, 2048] {
        let input = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 29 % 997) as f32 - 498.) / 139.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = d.alloc(rows * n * 4).unwrap();
        for round in 0..5 {
            let mut modes = [("linear", 4, 1), ("nemo_mlx_router4", 8, 4)];
            if round % 2 == 1 {
                modes.reverse();
            }
            for (name, columns, tile) in modes {
                let cmd = d.begin().unwrap();
                for _ in 0..23 {
                    cmd.dispatch(
                        name,
                        &[&weights, &input, &output],
                        &[k as u32, n as u32, rows as u32, 0, 1f32.to_bits()],
                        [n.div_ceil(columns), rows.div_ceil(tile), 1],
                        128,
                    );
                }
                let seconds = cmd.finish().unwrap();
                eprintln!(
                    "NEMO_ROUTER rows={rows} round={round} kernel={name} gpu_ms={:.4}",
                    seconds * 1000.
                );
            }
        }
    }
}

#[test]
#[ignore = "requires pinned checkpoint; expert-order timing diagnostic, not serving qualification"]
fn mlx_sorted_expert_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let w = model
        .layers
        .iter()
        .find_map(|l| {
            if let Mixer::Moe(w) = &l.mixer {
                Some(w)
            } else {
                None
            }
        })
        .unwrap();
    let d = &model.device;
    for rows in [4usize, 16, 32, 64, 85, 128, 256] {
        let input = d
            .upload(
                &(0..rows * WIDTH)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 29 % 997) as f32 - 498.) / 139.)
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for experts in [6usize, 16, 128] {
            let picks = d
                .upload(
                    &(0..rows * 6)
                        .flat_map(|i| (((i * 17 + i / 6) % experts) as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let up = d.alloc(rows * 6 * FF * 4).unwrap();
            let out = d.alloc(rows * 6 * WIDTH * 4).unwrap();
            let lists = d.alloc(rows * 6 * 128 * 4).unwrap();
            let counts = d.alloc(128 * 4).unwrap();
            let order = d.alloc(rows * 6 * 4).unwrap();
            let tiles = d
                .alloc((1 + 2 * ((rows * 6).div_ceil(4) + 128)) * 4)
                .unwrap();
            let mut reference: Option<Vec<f32>> = None;
            for round in 0..5 {
                let mut modes = [0, 1, 2, 3];
                if round % 2 == 1 {
                    modes.reverse();
                }
                for mode in modes {
                    let sorted = mode == 1;
                    let cmd = d.begin().unwrap();
                    if mode != 0 {
                        cmd.dispatch(
                            "moe_align",
                            &[&picks, &lists, &counts],
                            &[(rows * 6) as u32],
                            [128, 1, 1],
                            256,
                        );
                    }
                    if sorted {
                        cmd.dispatch(
                            "nemo_mlx_route_order",
                            &[&lists, &counts, &order],
                            &[(rows * 6) as u32],
                            [1, 1, 1],
                            128,
                        );
                    }
                    let tile: usize = if mode == 3 { 8 } else { 4 };
                    if mode >= 2 {
                        cmd.dispatch(
                            "moe_tiles",
                            &[&counts, &tiles],
                            &[128, tile as u32],
                            [1, 1, 1],
                            256,
                        );
                    }
                    for (weight, x, y, k, n, plain, sorted_name) in [
                        (
                            &w.up,
                            &input,
                            &up,
                            WIDTH,
                            FF,
                            "nemo_mlx_up_decode_reuse",
                            "nemo_mlx_up_decode_sorted",
                        ),
                        (
                            &w.down,
                            &up,
                            &out,
                            FF,
                            WIDTH,
                            "nemo_mlx_down_decode_reuse",
                            "nemo_mlx_down_decode_sorted",
                        ),
                    ] {
                        if mode >= 2 {
                            cmd.dispatch(
                                if mode == 3 && plain == "nemo_mlx_up_decode_reuse" {
                                    "nemo_mlx_up_vector_group8"
                                } else if mode == 3 {
                                    "nemo_mlx_down_vector_group8"
                                } else if plain == "nemo_mlx_up_decode_reuse" {
                                    "nemo_mlx_up_vector_group4"
                                } else {
                                    "nemo_mlx_down_vector_group4"
                                },
                                &[&weight.buffer, x, &lists, &counts, &tiles, y],
                                &[k as u32, n as u32, rows as u32],
                                [n.div_ceil(8), (rows * 6).div_ceil(tile) + 128, 1],
                                128,
                            );
                            continue;
                        }
                        let mut buffers = vec![&weight.buffer, x, &picks, y];
                        if sorted {
                            buffers.push(&order);
                        }
                        cmd.dispatch(
                            if sorted { sorted_name } else { plain },
                            &buffers,
                            &[k as u32, n as u32, rows as u32],
                            [n.div_ceil(16), rows * 6, 1],
                            128,
                        );
                    }
                    let seconds = cmd.finish().unwrap();
                    let actual = unsafe { out.read_f32(0, rows * 6 * WIDTH) };
                    if let Some(want) = &reference {
                        assert!(
                            actual
                                .iter()
                                .zip(want)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "sorted expert results changed"
                        );
                    } else {
                        reference = Some(actual);
                    }
                    eprintln!(
                        "NEMO_SORT rows={rows} experts={experts} round={round} mode={mode} gpu_ms={:.4}",
                        seconds * 1000.
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned checkpoint; dense-tile timing and equality diagnostic"]
fn mlx_dense_tile_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let m = model
        .layers
        .iter()
        .find_map(|l| {
            if let Mixer::Mamba(m) = &l.mixer {
                Some(m)
            } else {
                None
            }
        })
        .unwrap();
    let e = model
        .layers
        .iter()
        .find_map(|l| {
            if let Mixer::Moe(e) = &l.mixer {
                Some(e)
            } else {
                None
            }
        })
        .unwrap();
    for (label, w) in [
        ("input", &m.input),
        ("output", &m.output),
        ("shared_up", &e.shared_up),
        ("shared_down", &e.shared_down),
    ] {
        for rows in [128usize, 129, 512, 2048] {
            let input = model
                .device
                .upload(
                    &(0..rows * w.k)
                        .flat_map(|i| {
                            half::bf16::from_f32(((i * 29 % 997) as f32 - 498.) / 139.)
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let output = model.device.alloc(rows * w.n * 4).unwrap();
            let mut reference: Option<Vec<f32>> = None;
            for round in 0..5 {
                let mut modes = [
                    ("nemo_mlx_project_reuse", 64),
                    ("nemo_mlx_project_reuse128", 128),
                ];
                if round % 2 == 1 {
                    modes.reverse();
                }
                for (name, tile) in modes {
                    let cmd = model.device.begin().unwrap();
                    cmd.dispatch(
                        name,
                        &[&w.buffer, &input, &output],
                        &[w.k as u32, w.n as u32, rows as u32, w.ty, 1],
                        [w.n.div_ceil(64), rows.div_ceil(tile), 1],
                        128,
                    );
                    let seconds = cmd.finish().unwrap();
                    let actual = unsafe { output.read_f32(0, rows * w.n) };
                    if let Some(want) = &reference {
                        assert!(
                            actual
                                .iter()
                                .zip(want)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{label} rows={rows} kernel={name}: projection results changed"
                        );
                    } else {
                        reference = Some(actual);
                    }
                    eprintln!(
                        "NEMO_DENSE label={label} rows={rows} round={round} kernel={name} tile={tile} gpu_ms={:.4}",
                        seconds * 1000.
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned checkpoint; expert decode column reuse cost/equality"]
fn mlx_decode_column_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let w = model
        .layers
        .iter()
        .find_map(|l| match &l.mixer {
            Mixer::Moe(w) => Some(w),
            _ => None,
        })
        .unwrap();
    let d = &model.device;
    for rows in [1usize, 4, 16, 32, 64] {
        let input = d
            .upload(
                &(0..rows * WIDTH)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 29 % 997) as f32 - 498.) / 139.)
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for experts in [6usize, 128] {
            let picks = d
                .upload(
                    &(0..rows * 6)
                        .flat_map(|i| (((i * 17 + i / 6) % experts) as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let up = d.alloc(rows * 6 * FF * 4).unwrap();
            let out = d.alloc(rows * 6 * WIDTH * 4).unwrap();
            let mut reference: Option<Vec<f32>> = None;
            for round in 0..7 {
                let mut columns = [4usize, 2, 8, 16];
                if round % 2 == 1 {
                    columns.reverse();
                }
                for cols in columns {
                    let cmd = d.begin().unwrap();
                    for (weight, x, y, down) in
                        [(&w.up, &input, &up, false), (&w.down, &up, &out, true)]
                    {
                        // Packed Weight::n includes all 128 experts; the
                        // gather kernel receives the per-expert output width.
                        let n = if down { WIDTH } else { FF };
                        assert_eq!(weight.n, n * 128);
                        assert_eq!(y.len(), rows * 6 * n * 4);
                        let name = match (down, cols) {
                            (false, 2) => "nemo_mlx_up_decode_cols2",
                            (true, 2) => "nemo_mlx_down_decode_cols2",
                            (false, 8) => "nemo_mlx_up_decode_cols8",
                            (true, 8) => "nemo_mlx_down_decode_cols8",
                            (false, 16) => "nemo_mlx_up_decode_cols16",
                            (true, 16) => "nemo_mlx_down_decode_cols16",
                            (false, _) => "nemo_mlx_up_decode_reuse",
                            (true, _) => "nemo_mlx_down_decode_reuse",
                        };
                        cmd.dispatch(
                            name,
                            &[&weight.buffer, x, &picks, y],
                            &[weight.k as u32, n as u32, rows as u32],
                            [n.div_ceil(cols * 4), rows * 6, 1],
                            128,
                        );
                    }
                    let seconds = cmd.finish().unwrap();
                    let actual = unsafe { out.read_f32(0, rows * 6 * WIDTH) };
                    if let Some(want) = &reference {
                        assert!(
                            actual
                                .iter()
                                .zip(want)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "column reuse changed results, rows={rows} experts={experts} cols={cols}"
                        );
                    } else {
                        reference = Some(actual);
                    }
                    eprintln!(
                        "NEMO_COLS rows={rows} experts={experts} round={round} cols={cols} gpu_ms={:.4}",
                        seconds * 1000.
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned checkpoint; grouped expert tile cost/equality diagnostic"]
fn mlx_tensor_expert_tile_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let w = model
        .layers
        .iter()
        .find_map(|l| {
            if let Mixer::Moe(w) = &l.mixer {
                Some(w)
            } else {
                None
            }
        })
        .unwrap();
    let d = &model.device;
    for rows in [512usize, 1024, 2048] {
        let input = d
            .upload(
                &(0..rows * WIDTH)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 29 % 997) as f32 - 498.) / 139.)
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for experts in [16usize, 128] {
            let picks = d
                .upload(
                    &(0..rows * 6)
                        .flat_map(|i| (((i * 17 + i / 6) % experts) as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let up = d.alloc(rows * 6 * FF * 4).unwrap();
            let out = d.alloc(rows * 6 * WIDTH * 4).unwrap();
            let packed = d.alloc(rows * 6 * WIDTH * 2).unwrap();
            let lists = d.alloc(rows * 6 * 128 * 4).unwrap();
            let counts = d.alloc(128 * 4).unwrap();
            let offsets = d.alloc(128 * 4).unwrap();
            let tiles = d.alloc((1 + 2 * (rows * 6 / 32 + 128)) * 4).unwrap();
            let mut reference: Option<Vec<f32>> = None;
            for round in 0..5 {
                let mut modes = [(32usize, 64), (64, 64), (128, 64), (64, 128)];
                if round % 2 == 1 {
                    modes.reverse();
                }
                for (tile, bk) in modes {
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        "moe_align",
                        &[&picks, &lists, &counts],
                        &[(rows * 6) as u32],
                        [128, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "moe_tiles",
                        &[&counts, &tiles],
                        &[128, tile as u32],
                        [1, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "nemo_mlx_expert_offsets",
                        &[&counts, &offsets],
                        &[],
                        [1, 1, 1],
                        128,
                    );
                    for (down, weight, x, y, k, n) in [
                        (false, &w.up, &input, &up, WIDTH, FF),
                        (true, &w.down, &up, &out, FF, WIDTH),
                    ] {
                        cmd.dispatch(
                            "nemo_mlx_expert_pack",
                            &[x, &lists, &counts, &offsets, &tiles, &packed],
                            &[k as u32, rows as u32, u32::from(down), tile as u32],
                            [
                                (tile * k / 4).div_ceil(256),
                                (rows * 6).div_ceil(tile) + 128,
                                1,
                            ],
                            256,
                        );
                        let name = match (down, tile) {
                            (false, 32) => "nemo_mlx_up_reuse",
                            (true, 32) => "nemo_mlx_down_reuse",
                            (false, 64) => "nemo_mlx_up_reuse64",
                            (true, 64) => "nemo_mlx_down_reuse64",
                            (false, 128) => "nemo_mlx_up_reuse128",
                            (true, 128) => "nemo_mlx_down_reuse128",
                            _ => unreachable!(),
                        };
                        let name = if bk == 128 {
                            if down {
                                "nemo_mlx_down_reuse64_k128"
                            } else {
                                "nemo_mlx_up_reuse64_k128"
                            }
                        } else {
                            name
                        };
                        cmd.dispatch(
                            name,
                            &[
                                &weight.buffer,
                                &packed,
                                &lists,
                                &counts,
                                &tiles,
                                &offsets,
                                y,
                            ],
                            &[k as u32, n as u32, rows as u32],
                            [n.div_ceil(64), (rows * 6).div_ceil(tile) + 128, 1],
                            128,
                        );
                    }
                    let seconds = cmd.finish().unwrap();
                    let actual = unsafe { out.read_f32(0, rows * 6 * WIDTH) };
                    if let Some(want) = &reference {
                        assert!(
                            actual
                                .iter()
                                .zip(want)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "rows={rows} experts={experts} tile={tile} bk={bk}: expert results changed"
                        );
                    } else {
                        reference = Some(actual);
                    }
                    eprintln!(
                        "NEMO_TENSOR_EXPERT rows={rows} experts={experts} round={round} tile={tile} bk={bk} gpu_ms={:.4}",
                        seconds * 1000.
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned MLX checkpoint; fixed-work timing, not serving qualification"]
fn mlx_checkpoint_fixed_work_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixtures = std::env::var("PADDOCK_NEMOTRON_MLX_REFERENCE").unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{fixtures}/cases.json")).unwrap()).unwrap();
    let seed = cases[1]["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    for batch in [1, 4] {
        let rows = (0..batch)
            .flat_map(|slot| {
                let seed = &seed;
                (0..512 / batch).map(move |i| (slot, seed[i % seed.len()], i as u32))
            })
            .collect::<Vec<_>>();
        let outputs = (1..=batch).map(|i| i * 512 / batch - 1).collect::<Vec<_>>();
        let mut reference = None;
        for repeat in 0..3 {
            model.reset();
            while model.evict_prefix() {}
            let start = std::time::Instant::now();
            let logits = model.execute(&rows, &outputs).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.;
            assert!(logits.iter().all(|v| v.is_finite()));
            if let Some(expected) = &reference {
                assert_eq!(expected, &logits);
            } else {
                reference = Some(logits);
            }
            eprintln!(
                "NEMO_FIXED prefill batch={batch} rows=512 repeat={repeat} wall_ms={ms:.3} gpu_ms={:.3}",
                model.last_gpu_seconds * 1000.
            );
            let start = std::time::Instant::now();
            for step in 0..16 {
                let decodes = (0..batch)
                    .map(|slot| (slot, 100 + step, (512 / batch) as u32 + step))
                    .collect::<Vec<_>>();
                let logits = model
                    .execute(&decodes, &(0..batch).collect::<Vec<_>>())
                    .unwrap();
                assert!(logits.iter().all(|v| v.is_finite()));
            }
            eprintln!(
                "NEMO_FIXED decode batch={batch} steps=16 repeat={repeat} step_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000. / 16.
            );
        }
    }
}

#[test]
#[ignore = "requires real MLX checkpoint; long-context decode stage diagnostic"]
fn mlx_long_decode_costs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixture = std::env::var("PADDOCK_NEMOTRON_PROMPT_FILE").unwrap();
    let tokenizer =
        paddock_tokenizer::GgufTokenizer::from_hf_dir(std::path::Path::new(&path)).unwrap();
    let tokens = tokenizer
        .encode(&std::fs::read_to_string(fixture).unwrap())
        .unwrap();
    assert!(tokens.len() >= 16384);
    let mut model = Nemotron::load(std::path::Path::new(&path), 32768, 4, None).unwrap();
    for depth in [8192, 16384] {
        for batch in [1, 4] {
            model.reset();
            while model.evict_prefix() {}
            for slot in 0..batch {
                model.prefill(slot, &tokens[..depth]).unwrap();
            }
            let start = std::time::Instant::now();
            let mut gpu = 0.;
            for step in 0..16 {
                let rows = (0..batch)
                    .map(|s| (s, 100 + (step + s * 19) as u32, (depth + step) as u32))
                    .collect::<Vec<_>>();
                let logits = model
                    .execute(&rows, &(0..batch).collect::<Vec<_>>())
                    .unwrap();
                assert!(logits.iter().all(|v| v.is_finite()));
                gpu += model.last_gpu_seconds;
            }
            eprintln!(
                "NEMO_LONG_DECODE c={batch} depth={depth} step_ms={:.3} gpu_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000. / 16.,
                gpu * 1000. / 16.
            );
        }
    }
}

#[test]
#[ignore = "requires checkpoint and captured briefing; bounded mixed-work cost rung"]
fn mlx_mixed_prefill_cost_rung() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixture = std::env::var("PADDOCK_NEMOTRON_PROMPT_FILE").unwrap();
    let tokenizer =
        paddock_tokenizer::GgufTokenizer::from_hf_dir(std::path::Path::new(&path)).unwrap();
    let tokens = tokenizer
        .encode(&std::fs::read_to_string(fixture).unwrap())
        .unwrap();
    assert!(tokens.len() >= 16384);
    let mut model = Nemotron::load(std::path::Path::new(&path), 32768, 4, None).unwrap();
    let allocated = model.device.allocated_bytes();
    for depth in [8192usize, 16384] {
        model.reset();
        while model.evict_prefix() {}
        model.prefill(0, &tokens[..depth]).unwrap();
        let mut prompt = tokens[..depth].to_vec();
        prompt.extend(tokens.iter().copied().cycle().take(513));
        let mut reference: Option<Vec<f32>> = None;
        for round in 0..5 {
            let mut sizes = [64usize, 128, 256, 512];
            if round % 2 == 1 {
                sizes.reverse();
            }
            for size in sizes {
                model.reset();
                for slot in 0..4 {
                    assert_eq!(model.prepare(slot, &prompt).unwrap(), depth);
                }
                let mut rows: Vec<_> = (0..3).map(|s| (s, 100 + s as u32, depth as u32)).collect();
                rows.extend((0..size).map(|i| (3, prompt[depth + i], (depth + i) as u32)));
                let start = std::time::Instant::now();
                let logits = model.execute(&rows, &[0, 1, 2]).unwrap();
                let wall = start.elapsed().as_secs_f64() * 1000.;
                assert!(logits.iter().all(|v| v.is_finite()));
                if let Some(want) = &reference {
                    assert!(
                        logits == *want,
                        "live decode changed with pending width {size}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(model.device.allocated_bytes(), allocated);
                eprintln!(
                    "NEMO_MIXED depth={depth} round={round} prefill={size} decodes=3 wall_ms={wall:.3} gpu_ms={:.3}",
                    model.last_gpu_seconds * 1000.
                );
            }
        }
    }
}

#[test]
#[ignore = "requires pinned Nemotron MLX checkpoint and GPU"]
fn mlx_checkpoint_conversation_rotation_retains_newest_state() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 1, None).unwrap();
    let used = model.device.allocated_bytes();
    let old = vec![17; 513];
    model.prefill(0, &old).unwrap();
    model.anchor_at(0, &old, 512);
    model.reset();
    let new = (0..1537).map(|i| 100 + (i % 41) as u32).collect::<Vec<_>>();
    let cold = model.prefill(0, &new).unwrap();
    assert_eq!(model.take_prefill_reused(0), 0);
    model.reset();
    let warm = model.prefill(0, &new).unwrap();
    assert_eq!(model.take_prefill_reused(0), 1536);
    assert_eq!(cold, warm, "rotated recurrent state and BF16 KV restore");
    model.reset();
    while model.evict_prefix() {}
    assert_eq!(
        model.pool.free_blocks(),
        model.page_stride * (1 + MLX_PREFIX_PER_SLOT)
    );
    assert_eq!(model.device.allocated_bytes(), used);
}

#[test]
#[ignore = "requires pinned Nemotron MLX checkpoint and GPU"]
fn mlx_checkpoint_cache_batch_and_lifecycle() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let path = std::path::Path::new(&path);
    for (context, batch) in [(0, 1), (32769, 1), (4096, 0), (4096, 65)] {
        assert!(Nemotron::load(path, context, batch, None).is_err());
    }
    assert!(matches!(
        Nemotron::load(path, 4096, 4, Some(1 << 20)),
        Err(MetalError::Memory(_))
    ));
    let mut model = Nemotron::load(path, 4096, 4, None).unwrap();
    assert_eq!(model.embedding.ty, mlx::AFFINE8);
    assert_eq!(model.weight_bytes, 33577599744);
    let used = model.device.allocated_bytes();
    let prompt = (0..513).map(|i| 100 + (i % 41) as u32).collect::<Vec<_>>();
    let cold = model.prefill(3, &prompt).unwrap();
    model.reset();
    let reused = model.prefill(0, &prompt).unwrap();
    assert_eq!(model.take_prefill_reused(0), 512);
    assert_eq!(cold, reused, "complete recurrent and BF16 KV restore");
    let prompts = (0..4)
        .map(|slot| {
            (0..25)
                .map(|i| 100 + (i % 41) as u32 + slot * 17)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for p in &prompts {
        model.reset();
        while model.evict_prefix() {}
        expected.push(model.prefill(0, p).unwrap());
    }
    model.reset();
    while model.evict_prefix() {}
    let rows = prompts
        .iter()
        .enumerate()
        .flat_map(|(s, p)| p.iter().enumerate().map(move |(i, &t)| (s, t, i as u32)))
        .collect::<Vec<_>>();
    let mixed = model.execute(&rows, &[24, 49, 74, 99]).unwrap();
    for (slot, logits) in mixed.chunks_exact(VOCAB).enumerate() {
        let top = |a: &[f32]| {
            a.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(top(logits), top(&expected[slot]), "packed request {slot}");
    }
    model.reset();
    while model.evict_prefix() {}
    assert_eq!(
        model.pool.free_blocks(),
        model.page_stride * 4 * (1 + MLX_PREFIX_PER_SLOT)
    );
    assert_eq!(model.device.allocated_bytes(), used);
    let one = model.forward(17).unwrap();
    model.reset();
    let again = model.forward(17).unwrap();
    assert_eq!(one, again, "reset cannot inherit recurrent state");
}

#[test]
#[ignore = "requires Metal GPU"]
fn affine8_expert_routes_preserve_packed_planes_and_mask_tails() {
    let d = MetalDevice::new(None).unwrap();
    let (k, n) = (64usize, 33usize);
    let mut bytes = (0..128)
        .flat_map(|e| vec![(e % 13 + 1) as u8; k * n])
        .collect::<Vec<_>>();
    bytes.extend((0..128 * n).flat_map(|_| half::bf16::from_f32(0.5).to_le_bytes()));
    bytes.extend((0..128 * n).flat_map(|_| half::bf16::from_f32(-1.).to_le_bytes()));
    let weights = d.upload(&bytes).unwrap();
    for rows in [1, 4, 17, 33, 512] {
        let entries = rows * 6;
        let ids = (0..entries)
            .map(|i| ((i * 17 + i / 6) % 128) as u32)
            .collect::<Vec<_>>();
        let picks = d
            .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let input = d
            .upload(
                &(0..entries * k)
                    .flat_map(|_| 1f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let lists = d.alloc(128 * entries * 4).unwrap();
        let counts = d.alloc(128 * 4).unwrap();
        let tiles = d.alloc((1 + 2 * (entries.div_ceil(4) + 128)) * 4).unwrap();
        for down in [false, true] {
            for (tile, reuse, sorted, bk128) in [
                (0usize, false, false, false),
                (0, true, false, false),
                (0, true, true, false),
                (4, false, false, false),
                (8, false, false, false),
                (16, false, false, false),
                (32, false, false, false),
                (32, true, false, false),
                (64, true, false, false),
                (64, true, false, true),
            ] {
                let grouped = tile != 0;
                let output = d.upload(&vec![0xff; (entries * n + 32) * 4]).unwrap();
                let packed = d.upload(&vec![0xff; entries * k * 2 + 64]).unwrap();
                let offsets = d.alloc(128 * 4).unwrap();
                let order = d.upload(&vec![0xff; (entries + 16) * 4]).unwrap();
                let cmd = d.begin().unwrap();
                let p = [k as u32, n as u32, rows as u32];
                if tile == 4 || tile == 8 {
                    cmd.dispatch(
                        "moe_align",
                        &[&picks, &lists, &counts],
                        &[entries as u32],
                        [128, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "moe_tiles",
                        &[&counts, &tiles],
                        &[128, tile as u32],
                        [1, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        if tile == 8 && down {
                            "nemo_mlx_down_vector_group8"
                        } else if tile == 8 {
                            "nemo_mlx_up_vector_group8"
                        } else if down {
                            "nemo_mlx_down_vector_group4"
                        } else {
                            "nemo_mlx_up_vector_group4"
                        },
                        &[&weights, &input, &lists, &counts, &tiles, &output],
                        &p,
                        [n.div_ceil(8), entries.div_ceil(tile) + 128, 1],
                        128,
                    );
                } else if sorted {
                    cmd.dispatch(
                        "moe_align",
                        &[&picks, &lists, &counts],
                        &[entries as u32],
                        [128, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "nemo_mlx_route_order",
                        &[&lists, &counts, &order],
                        &[entries as u32],
                        [1, 1, 1],
                        128,
                    );
                    cmd.dispatch(
                        if down {
                            "nemo_mlx_down_decode_sorted"
                        } else {
                            "nemo_mlx_up_decode_sorted"
                        },
                        &[&weights, &input, &picks, &output, &order],
                        &p,
                        [n.div_ceil(16), entries, 1],
                        128,
                    );
                } else if grouped {
                    cmd.dispatch(
                        "moe_align",
                        &[&picks, &lists, &counts],
                        &[entries as u32],
                        [128, 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "moe_tiles",
                        &[&counts, &tiles],
                        &[128, tile as u32],
                        [1, 1, 1],
                        256,
                    );
                    if reuse {
                        cmd.dispatch(
                            "nemo_mlx_expert_offsets",
                            &[&counts, &offsets],
                            &[],
                            [1, 1, 1],
                            128,
                        );
                        cmd.dispatch(
                            "nemo_mlx_expert_pack",
                            &[&input, &lists, &counts, &offsets, &tiles, &packed],
                            &[k as u32, rows as u32, u32::from(down), tile as u32],
                            [
                                (tile * k / 4).div_ceil(256),
                                entries.div_ceil(tile) + 128,
                                1,
                            ],
                            256,
                        );
                        cmd.dispatch(
                            if bk128 && down {
                                "nemo_mlx_down_reuse64_k128"
                            } else if bk128 {
                                "nemo_mlx_up_reuse64_k128"
                            } else if tile == 64 && down {
                                "nemo_mlx_down_reuse64"
                            } else if tile == 64 {
                                "nemo_mlx_up_reuse64"
                            } else if down {
                                "nemo_mlx_down_reuse"
                            } else {
                                "nemo_mlx_up_reuse"
                            },
                            &[
                                &weights, &packed, &lists, &counts, &tiles, &offsets, &output,
                            ],
                            &p,
                            [n.div_ceil(64), entries.div_ceil(tile) + 128, 1],
                            128,
                        );
                    } else {
                        cmd.dispatch(
                            if tile == 32 && down {
                                "nemo_mlx_down_grouped32"
                            } else if tile == 32 {
                                "nemo_mlx_up_grouped32"
                            } else if down {
                                "nemo_mlx_down_grouped"
                            } else {
                                "nemo_mlx_up_grouped"
                            },
                            &[&weights, &input, &lists, &counts, &tiles, &output],
                            &p,
                            [
                                n.div_ceil(if tile == 32 { 64 } else { 32 }),
                                entries.div_ceil(tile) + 128,
                                1,
                            ],
                            128,
                        );
                    }
                } else {
                    cmd.dispatch(
                        if reuse && down {
                            "nemo_mlx_down_decode_reuse"
                        } else if reuse {
                            "nemo_mlx_up_decode_reuse"
                        } else if down {
                            "nemo_mlx_down_decode"
                        } else {
                            "nemo_mlx_up_decode"
                        },
                        &[&weights, &input, &picks, &output],
                        &p,
                        [n.div_ceil(if reuse { 16 } else { 4 }), entries, 1],
                        128,
                    );
                }
                cmd.finish().unwrap();
                if sorted {
                    let sorted_entries = unsafe { order.read_u32(entries + 16) };
                    assert!(sorted_entries[entries..].iter().all(|&v| v == u32::MAX));
                    assert!(
                        sorted_entries[..entries]
                            .windows(2)
                            .all(|w| ids[w[0] as usize] <= ids[w[1] as usize])
                    );
                    let mut permutation = sorted_entries[..entries].to_vec();
                    permutation.sort_unstable();
                    assert_eq!(permutation, (0..entries as u32).collect::<Vec<_>>());
                }
                let packed_words = unsafe { packed.read_u32((entries * k * 2 + 64) / 4) };
                assert!(
                    packed_words[entries * k / 2..]
                        .iter()
                        .all(|&v| v == u32::MAX)
                );
                // Exact power-of-two scalar fixture, not a CPU inference oracle.
                let actual = unsafe { output.read_f32(0, entries * n + 32) };
                for (e, id) in ids.iter().enumerate() {
                    let sum = k as f32 * ((id % 13 + 1) as f32 * 0.5 - 1.);
                    let want = if down { sum } else { sum.max(0.).powi(2) };
                    assert!(
                        actual[e * n..(e + 1) * n].iter().all(|&v| v == want),
                        "rows={rows} entry={e} down={down} grouped={grouped}"
                    );
                }
                assert!(
                    actual[entries * n..]
                        .iter()
                        .all(|v| v.to_bits() == u32::MAX)
                );
            }
        }
    }
}

#[test]
#[ignore = "requires pinned Nemotron MLX checkpoint and GPU"]
fn mlx_checkpoint_reference_generations() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let reference = std::env::var("PADDOCK_NEMOTRON_MLX_REFERENCE").unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{reference}/cases.json")).unwrap()).unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let used = model.device.allocated_bytes();
    eprintln!(
        "MLX weights={} cache={} allocated={used}",
        model.weight_bytes, model.cache_bytes
    );
    let mut equal = 0;
    for (i, case) in cases.as_array().unwrap().iter().enumerate() {
        model.reset();
        while model.evict_prefix() {}
        let prompt = case["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let expected = case["generated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let mut logits = model.prefill(0, &prompt).unwrap();
        let mut actual = Vec::new();
        for step in 0..expected.len() {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            actual.push(token);
            if step + 1 < expected.len() {
                logits = model
                    .execute(&[(0, token, (prompt.len() + step) as u32)], &[0])
                    .unwrap();
            }
        }
        eprintln!("case {i} expected={expected:?} actual={actual:?}");
        equal += usize::from(actual == expected);
        assert_eq!(used, model.device.allocated_bytes());
    }
    assert_eq!(
        equal,
        cases.as_array().unwrap().len(),
        "complete-generation parity remains incomplete"
    );
}

#[test]
#[ignore = "diagnostic BF16 graph; full generation must match the independent MLX reference"]
fn mlx_bf16_checkpoint_reference_generations() {
    crate::projection::with_affine_mlx(mlx_checkpoint_reference_generations);
}

#[test]
#[ignore = "diagnostic tensor graph; fixed-work timing, not serving qualification"]
fn mlx_bf16_checkpoint_fixed_work_costs() {
    crate::projection::with_affine_mlx(mlx_checkpoint_fixed_work_costs);
}

#[test]
#[ignore = "tensor graph: unrelated prompt arrivals must not change live decode"]
fn mlx_bf16_mixed_arrival_contract() {
    crate::projection::with_affine_mlx(mlx_live_decode_is_independent_of_prompt_admission);
}

#[test]
#[ignore = "independent layer-input replay diagnostic; not full-generation qualification"]
fn mlx_independent_layer_inputs() {
    let path = std::env::var("PADDOCK_NEMOTRON_MLX_MODEL").unwrap();
    let fixtures = std::env::var("PADDOCK_NEMOTRON_MLX_REFERENCE").unwrap();
    let trace = std::env::var("PADDOCK_NEMOTRON_MLX_TRACE").unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{fixtures}/cases.json")).unwrap()).unwrap();
    let mut model = Nemotron::load(std::path::Path::new(&path), 4096, 1, None).unwrap();
    let f = paddock_models::safetensors::SafetensorsFile::open(
        &std::path::Path::new(&fixtures).join("case-1.safetensors"),
    )
    .unwrap();
    for i in 0..model.layers.len() {
        let name = if i == 0 {
            "embed".to_owned()
        } else {
            format!("layer-{}", i - 1)
        };
        model
            .reference_inputs
            .push(model.device.upload(f.bytes(&name).unwrap().1).unwrap());
    }
    let prompt = cases[1]["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    model.prefill(0, &prompt).unwrap();
    for i in 0..model.layers.len() {
        let bytes = std::fs::read(format!("{trace}/layer-{i}-{}.f32", prompt.len())).unwrap();
        let actual = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        let expected = f
            .bytes(&format!("layer-{i}"))
            .unwrap()
            .1
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        let mut error = 0f64;
        let mut norm = 0f64;
        let mut different = 0;
        for (a, b) in actual.zip(expected) {
            assert!(a.is_finite());
            error += f64::from(a - b).powi(2);
            norm += f64::from(b).powi(2);
            different += usize::from(a.to_bits() != b.to_bits());
        }
        eprintln!(
            "NEMO_LAYER layer={i} different={different} relative={:.8}",
            (error / norm).sqrt()
        );
    }
}
