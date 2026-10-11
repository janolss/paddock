use super::*;
use paddock_engine::generator::Generator;

fn upload(d: &MetalDevice, x: &[f32]) -> Buffer {
    d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn ids(d: &MetalDevice, x: &[u32]) -> Buffer {
    d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn fixture(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + i / 113) % 251) as f32 / 125.0 * scale - scale)
        .collect()
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let mut error = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite(), "{x} {y}");
        error = error.max((x - y).abs());
    }
    eprintln!("maximum difference {error}, tolerance {tol}");
    assert!(error <= tol, "{error} > {tol}");
}
fn top(a: &[f32]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[test]
fn native_scratch_bound_matches_published_ceiling() {
    assert_eq!(Scratch::sizes(4, 256).iter().sum::<usize>(), 377959812);
    assert_eq!(Scratch::sizes(64, 2048).iter().sum::<usize>(), 677817604);
}

#[test]
#[ignore = "requires M5 GPU"]
fn pipelines_fit() {
    let d = MetalDevice::new(None).unwrap();
    eprintln!("Nemotron pipelines compiled, grant {}", d.budget_bytes());
}

#[test]
#[ignore = "requires M5 GPU"]
fn softplus_retains_small_positive_steps() {
    let d = MetalDevice::new(None).unwrap();
    let x = upload(&d, &[-80., -25., -15., -9., 0., 25., 80.]);
    let y = d.alloc(7 * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch("nemo_dt_check", &[&x, &y], &[7], [1, 1, 1], 32);
    cmd.finish().unwrap();
    // Analytic scalar constants, not a CPU inference/reference graph.
    let expected = [
        1.8048513e-35f32,
        1.3887944e-11,
        3.0590226e-7,
        0.00012340219,
        std::f32::consts::LN_2,
        25.,
        80.,
    ];
    // SAFETY: the scalar GPU fixture completed.
    for (actual, expected) in unsafe { y.read_f32(0, 7) }.into_iter().zip(expected) {
        assert!(actual > 0.);
        assert!(
            (actual - expected).abs() <= expected * 0.00001,
            "{actual} {expected}"
        );
    }
}

#[test]
#[ignore = "requires M5 GPU"]
fn router_bias_changes_selection_not_probabilities() {
    let d = MetalDevice::new(None).unwrap();
    let logits = upload(&d, &vec![0.; 128]);
    let output = d.alloc(6 * 4).unwrap();
    let weights = d.alloc(6 * 4).unwrap();
    for reverse in [false, true] {
        let bias = upload(
            &d,
            &(0..128)
                .map(|i| if reverse { i as f32 * 0.01 } else { 0. })
                .collect::<Vec<_>>(),
        );
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "nemo_route",
            &[&logits, &bias, &output, &weights],
            &[],
            [1, 1, 1],
            32,
        );
        cmd.finish().unwrap();
        // SAFETY: command completed; expected constants describe a tie and
        // six equal sigmoid probabilities, not a host-side model oracle.
        unsafe {
            assert_eq!(
                output.read_u32(6),
                (0..6)
                    .map(|i| if reverse { 127 - i } else { i })
                    .collect::<Vec<_>>()
            );
            close(&weights.read_f32(0, 6), &[2.5 / 6.; 6], 0.000001);
        }
    }
}

#[test]
#[ignore = "requires M5 GPU"]
fn ssd_matches_gpu_scan_ragged_nonzero_state() {
    let d = MetalDevice::new(None).unwrap();
    for counts in [[1, 15, 33, 129], [17, 31, 32, 255], [1, 1, 1, 509]] {
        let m: usize = counts.iter().sum();
        let mut seq = Vec::new();
        let mut tiles = Vec::new();
        let mut first = 0;
        for (i, (&slot, &count)) in [3, 0, 2, 1].iter().zip(&counts).enumerate() {
            seq.extend([slot, first as u32, count as u32, (tiles.len() / 4) as u32]);
            if count >= 16 {
                for offset in (0..count).step_by(32) {
                    tiles.extend([
                        slot,
                        (first + offset) as u32,
                        (count - offset).min(32) as u32,
                        i as u32,
                    ]);
                }
            }
            first += count;
        }
        let nt = tiles.len() / 4;
        let conv = upload(&d, &fixture(m * CONV, 0.3));
        let mut projection = fixture(m * PROJECTED, 0.4);
        for row in 0..m {
            for h in 0..64 {
                projection[row * PROJECTED + 10240 + h] = match h % 8 {
                    0 => -25.,
                    1 => 30.,
                    _ => -3. + (row % 7) as f32 * 0.1,
                };
            }
        }
        let proj = upload(&d, &projection);
        let a = upload(
            &d,
            &(0..64).map(|i| -0.1 - i as f32 * 0.02).collect::<Vec<_>>(),
        );
        let dw = upload(&d, &fixture(64, 0.7));
        let bias = upload(&d, &fixture(64, 0.2));
        let initial = fixture(STATE * 5, 0.05);
        let state = upload(&d, &initial);
        let reference = upload(&d, &initial);
        let sq = ids(&d, &seq);
        let ts = ids(&d, &tiles);
        let y = upload(&d, &vec![f32::NAN; m * INNER + 64]);
        let yr = upload(&d, &vec![f32::NAN; m * INNER + 64]);
        let decay = d.alloc(nt * 64 * 32 * 4).unwrap();
        let dt = d.alloc(m * 64 * 4).unwrap();
        let mat = d.alloc(nt * 64 * 1024 * 4).unwrap();
        let delta = d.alloc(nt * STATE * 4).unwrap();
        let incoming = d.alloc(nt * STATE * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "nemo_scan",
            &[&conv, &proj, &a, &dw, &bias, &reference, &sq, &yr],
            &[1],
            [16, 64, 4],
            128,
        );
        cmd.dispatch(
            "nemo_scan",
            &[&conv, &proj, &a, &dw, &bias, &state, &sq, &y],
            &[0],
            [16, 64, 4],
            128,
        );
        cmd.dispatch(
            "nemo_ssd_prepare",
            &[&proj, &a, &bias, &ts, &decay, &dt],
            &[],
            [64, nt, 1],
            32,
        );
        cmd.dispatch(
            "nemo_ssd_matrix",
            &[&conv, &ts, &decay, &dt, &mat],
            &[],
            [64, nt, 1],
            128,
        );
        cmd.dispatch(
            "nemo_ssd_delta",
            &[&conv, &ts, &decay, &dt, &delta],
            &[],
            [16, 64, nt],
            128,
        );
        cmd.dispatch(
            "nemo_ssd_states",
            &[&state, &delta, &decay, &sq, &ts, &incoming],
            &[],
            [STATE.div_ceil(256), 4, 1],
            256,
        );
        cmd.dispatch(
            "nemo_ssd_output",
            &[&conv, &ts, &decay, &mat, &incoming, &dw, &y],
            &[],
            [4, 64, nt],
            128,
        );
        cmd.finish().unwrap();
        // SAFETY: both independent GPU contractions completed before comparison.
        unsafe {
            close(&y.read_f32(0, m * INNER), &yr.read_f32(0, m * INNER), 0.001);
            close(
                &state.read_f32(0, STATE * 4),
                &reference.read_f32(0, STATE * 4),
                0.0001,
            );
            assert!(y.read_f32(m * INNER, 64).iter().all(|v| v.is_nan()));
            close(&state.read_f32(STATE * 4, STATE), &initial[STATE * 4..], 0.);
        }
    }
}

#[test]
#[ignore = "requires M5 GPU"]
fn convolution_chunk_boundaries_and_slot_isolation() {
    let d = MetalDevice::new(None).unwrap();
    let m = 37;
    let proj = upload(&d, &fixture(m * PROJECTED, 0.5));
    let w = upload(&d, &fixture(CONV * 4, 0.3));
    let b = upload(&d, &fixture(CONV, 0.2));
    let initial = fixture(WINDOW * 5, 0.1);
    let win = upload(&d, &initial);
    let ref_win = upload(&d, &initial);
    let y = d.alloc(m * CONV * 4).unwrap();
    let yr = d.alloc(m * CONV * 4).unwrap();
    let full = ids(&d, &[3, 0, m as u32, 0]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "nemo_conv",
        &[&proj, &ref_win, &w, &b, &full, &yr],
        &[],
        [(m * CONV).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "nemo_conv_commit",
        &[&proj, &ref_win, &full],
        &[],
        [CONV.div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    let mut first = 0;
    for count in [1, 2, 17, 1, 16] {
        let sq = ids(&d, &[3, first, count, 0]);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "nemo_conv",
            &[&proj, &win, &w, &b, &sq, &y],
            &[],
            [(count as usize * CONV).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "nemo_conv_commit",
            &[&proj, &win, &sq],
            &[],
            [CONV.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        first += count;
    }
    // SAFETY: all convolutions and commits completed.
    unsafe {
        close(&y.read_f32(0, m * CONV), &yr.read_f32(0, m * CONV), 0.);
        close(
            &win.read_f32(0, WINDOW * 5),
            &ref_win.read_f32(0, WINDOW * 5),
            0.,
        );
    }
}

#[test]
#[ignore = "requires M5 GPU"]
fn routed_relu2_grouped_matches_gpu_decode() {
    let d = MetalDevice::new(None).unwrap();
    let rows = 33;
    let quant = |k, n| {
        let mut bytes = vec![0u8; k * n * 128 / 32 * 34];
        for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(
                &half::f16::from_f32(0.002 + (i % 3) as f32 * 0.001).to_le_bytes(),
            );
            for (j, v) in block[2..].iter_mut().enumerate() {
                *v = (((i * 13 + j * 7) % 33) as i8 - 16) as u8;
            }
        }
        d.upload(&bytes).unwrap()
    };
    let up = quant(WIDTH, FF);
    let down = quant(FF, WIDTH);
    for hot in [false, true] {
        let x = upload(&d, &fixture(rows * WIDTH, 0.2));
        let picks = ids(
            &d,
            &(0..rows * 6)
                .map(|i| {
                    if hot {
                        (i % 6) as u32
                    } else {
                        ((i * 17 + i / 6) % 128) as u32
                    }
                })
                .collect::<Vec<_>>(),
        );
        let lists = d.alloc(128 * rows * 6 * 4).unwrap();
        let counts = d.alloc(128 * 4).unwrap();
        let tiles = d
            .alloc((1 + 2 * ((rows * 6).div_ceil(16) + 128)) * 4)
            .unwrap();
        let u = d.alloc(rows * 6 * FF * 4).unwrap();
        let ur = d.alloc(rows * 6 * FF * 4).unwrap();
        let y = d.alloc(rows * 6 * WIDTH * 4).unwrap();
        let yr = d.alloc(rows * 6 * WIDTH * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&picks, &lists, &counts],
            &[(rows * 6) as u32],
            [128, 1, 1],
            256,
        );
        cmd.dispatch("moe_tiles", &[&counts, &tiles], &[128, 16], [1, 1, 1], 256);
        for (w, input, out, reference, k, n, name) in [
            (&up, &x, &u, &ur, WIDTH, FF, "nemo_up"),
            (&down, &u, &y, &yr, FF, WIDTH, "nemo_down"),
        ] {
            let p = [k as u32, n as u32, rows as u32];
            cmd.dispatch(
                if name == "nemo_up" {
                    "nemo_up_grouped"
                } else {
                    "nemo_down_grouped"
                },
                &[w, input, &lists, &counts, &tiles, out],
                &p,
                [n.div_ceil(32), (rows * 6).div_ceil(16) + 128, 1],
                128,
            );
            cmd.dispatch(
                if name == "nemo_up" {
                    "nemo_up_decode"
                } else {
                    "nemo_down_decode"
                },
                &[w, input, &picks, reference],
                &p,
                [n.div_ceil(4), rows * 6, 1],
                128,
            );
        }
        cmd.finish().unwrap();
        // SAFETY: both GPU projection routes completed. No CPU tensor oracle.
        unsafe {
            close(
                &u.read_f32(0, rows * 6 * FF),
                &ur.read_f32(0, rows * 6 * FF),
                0.001,
            );
            close(
                &y.read_f32(0, rows * 6 * WIDTH),
                &yr.read_f32(0, rows * 6 * WIDTH),
                0.001,
            );
        }
    }
}

#[test]
#[ignore = "requires elected Nemotron Q8 in PADDOCK_NEMOTRON_TEST_MODEL and M5"]
fn elected_checkpoint_lifecycle() {
    let path = std::env::var("PADDOCK_NEMOTRON_TEST_MODEL").unwrap();
    let path = std::path::Path::new(&path);
    for (ctx, batch) in [(0, 1), (32769, 1), (4096, 0), (4096, 65)] {
        assert!(Nemotron::load(path, ctx, batch, None).is_err());
    }
    assert!(matches!(
        Nemotron::load(path, 4096, 4, Some(1 << 20)),
        Err(MetalError::Memory(_))
    ));
    let mut m = Nemotron::load(path, 4096, 4, None).unwrap();
    let used = m.device.allocated_bytes();
    assert_eq!(m.weight_bytes, 33577599744);
    assert_eq!(m.layers.len(), 52);
    eprintln!(
        "weights={} cache={} scratch={} total={used}",
        m.weight_bytes,
        m.cache_bytes,
        used - m.weight_bytes - m.cache_bytes
    );
    let prompt = (0..513).map(|i| 100 + (i % 41) as u32).collect::<Vec<_>>();
    let cold = m.prefill(3, &prompt).unwrap();
    eprintln!(
        "top {} range {}..{}",
        top(&cold),
        cold.iter().copied().fold(f32::INFINITY, f32::min),
        cold.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    m.reset();
    let resumed = m.prefill(0, &prompt).unwrap();
    assert_eq!(m.take_prefill_reused(0), 512);
    close(&cold, &resumed, 0.00001);
    // Independent serial GPU state recurrence versus matrix SSD, same dense
    // and expert projections. Neither graph is a CPU reference model.
    while m.evict_prefix() {}
    m.reset();
    m.scan_only = true;
    let serial = m.prefill(0, &prompt[..145]).unwrap();
    m.reset();
    m.scan_only = false;
    let ssd = m.prefill(0, &prompt[..145]).unwrap();
    close(&serial, &ssd, 0.1);
    let prompts = (0..4)
        .map(|i| {
            (0..(145 + i * 7))
                .map(|j| 200 + i as u32 + (j % 31) as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for p in &prompts {
        m.reset();
        expected.push(m.prefill(0, p).unwrap());
    }
    m.reset();
    for (slot, p) in prompts.iter().enumerate() {
        m.prefill_begin(slot, p.clone()).unwrap();
    }
    let mut completed = 0;
    while !m.pending.is_empty() {
        for (slot, logits, n) in m.forward_mixed(&[], 127).unwrap().1 {
            assert_eq!(n, prompts[slot].len());
            close(&expected[slot], &logits, 0.1);
            completed += 1;
        }
    }
    assert_eq!(completed, 4);
    m.reset();
    m.prefill(3, &prompts[0]).unwrap();
    let next = m
        .execute(&[(3, 77, prompts[0].len() as u32)], &[0])
        .unwrap();
    m.reset();
    m.prefill(3, &prompts[0]).unwrap();
    m.prefill_begin(0, prompts[1].clone()).unwrap();
    let mixed = m
        .forward_mixed(&[(3, 77, prompts[0].len() as u32)], 127)
        .unwrap()
        .0;
    close(&next, &mixed, 0.1);
    assert!(m.prefill_abort(0));
    assert!(m.pending.is_empty());
    assert!(m.slots[0].history.is_empty());
    assert!(m.execute(&[(3, 77, 0)], &[0]).is_err());
    assert!(m.prefill(4, &[1]).is_err());
    assert!(m.prefill(0, &[VOCAB as u32]).is_err());
    m.release_inactive_slots(&[]);
    while m.evict_prefix() {}
    assert_eq!(
        m.pool.free_blocks(),
        m.page_stride * (4 + PREFIX_PER_SLOT * 4)
    );
    assert_eq!(m.device.allocated_bytes(), used);
    m.reset();
    let first = m.forward(17).unwrap();
    m.reset();
    let again = m.forward(17).unwrap();
    close(&first, &again, 0.);
}
