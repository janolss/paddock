//! Replay independent MLX recurrent-operation inputs without altering serving.
use super::*;

#[test]
#[ignore = "requires metal-lighton-language.py recurrent capture"]
fn lighton_recurrent_operation_contracts() {
    let root = std::env::var_os("PADDOCK_METAL_LANGUAGE_SEAMS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let force_precise = std::env::var_os("PADDOCK_METAL_PRECISE_RECURRENT").is_some();
    let baseline = std::env::var_os("PADDOCK_METAL_BASELINE_LIGHTON").is_some();
    struct Reset(bool, bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::PRECISE_RECURRENT_FOR_TEST.set(self.0);
            forward::BASELINE_LIGHTON_FOR_TEST.set(self.1);
        }
    }
    let _reset = Reset(
        forward::PRECISE_RECURRENT_FOR_TEST.replace(force_precise),
        forward::BASELINE_LIGHTON_FOR_TEST.replace(baseline),
    );
    let model = Qwen35::load(
        Path::new(manifest["model"].as_str().unwrap()),
        4096,
        1,
        None,
    )
    .unwrap();
    let precise = model.precise_recurrent_arithmetic();
    let d = &model.device;
    let Mixer::Linear(w) = &model.layers[0].mixer else {
        panic!("linear layer")
    };
    let g = model.geometry;
    let read = |name: &str| {
        let bytes = std::fs::read(root.join(format!("layer-0-{name}.f32"))).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let upload = |v: &[f32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let words = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let compare = |name: &str, a: &[f32], b: &[f32]| {
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        assert!(a.iter().all(|v| v.is_finite()));
        let diff = a
            .iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "RECURRENT_CONTRACT {name} different={diff}/{} max_abs={max}",
            a.len()
        );
        diff
    };
    let query = read("recurrent-query");
    let key = read("recurrent-key");
    let value = read("recurrent-value");
    let alpha = read("recurrent-alpha");
    let beta = read("recurrent-beta-input");
    let rows = alpha.len() / g.value_heads;
    assert!((1..=512).contains(&rows));
    assert_eq!(query.len(), rows * KEY_HEADS * 128);
    assert_eq!(key.len(), query.len());
    assert_eq!(value.len(), rows * g.value_heads * 128);
    assert_eq!(beta.len(), alpha.len());
    let qkv = (0..rows)
        .flat_map(|r| {
            query[r * KEY_HEADS * 128..][..KEY_HEADS * 128]
                .iter()
                .chain(&key[r * KEY_HEADS * 128..][..KEY_HEADS * 128])
                .chain(&value[r * g.value_heads * 128..][..g.value_heads * 128])
                .copied()
        })
        .collect::<Vec<_>>();
    let qkv_ref = upload(&qkv);
    let raw = read("linear_attn.in_proj_qkv-output");
    assert_eq!(raw.len(), rows * g.conv());
    let raw = upload(&raw);
    let aa = upload(&alpha);
    let bb = upload(&beta);
    let reference_a = read("recurrent-negative-a");
    assert_eq!(reference_a.len(), g.value_heads);
    let a_differences = compare(
        "negative-a",
        &unsafe { w.a.buffer.read_f32(0, g.value_heads) },
        &reference_a,
    );
    let meta = words(&(0..rows as u32).flat_map(|r| [0, r]).collect::<Vec<_>>());
    let bounds = words(&(0..rows).flat_map(|_| [0, rows as u32]).collect::<Vec<_>>());
    let spans = words(&[0, rows as u32, 0, 0]);
    let checkpoints = words(&vec![0; rows]);
    let history = d.alloc(g.conv() * 3 * 4).unwrap();
    let conv = d.upload(&vec![0xff; (rows * g.conv() + 32) * 4]).unwrap();
    let gates = d.alloc(alpha.len() * 2 * 4).unwrap();
    let state = d.alloc(g.state() * 4).unwrap();
    let out = d.upload(&vec![0xff; (value.len() + 32) * 4]).unwrap();
    let p = [
        KEY_HEADS as u32,
        g.value_heads as u32,
        g.conv() as u32,
        rows as u32,
        1,
        0,
        0,
    ];
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        if precise {
            "mlx_dn_conv_precise"
        } else {
            "mlx_dn_conv"
        },
        &[&raw, &w.conv.buffer, &history, &meta, &bounds, &conv],
        &p,
        [(rows * g.conv()).div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch("mlx_dn_qk_norm", &[&conv], &p, [KEY_HEADS * 2, rows, 1], 32);
    cmd.dispatch(
        if precise {
            "mlx_dn_gates_precise"
        } else {
            "mlx_dn_gates"
        },
        &[&aa, &bb, &w.a.buffer, &w.dt.buffer, &gates],
        &p,
        [alpha.len().div_ceil(256), 1, 1],
        256,
    );
    cmd.dispatch(
        "mlx_dn_recurrent_quad",
        &[&qkv_ref, &gates, &state, &spans, &meta, &out, &checkpoints],
        &p,
        [8, g.value_heads, 1],
        128,
    );
    cmd.finish().unwrap();
    let mut diff = a_differences
        + compare(
            "conv+normalization",
            &unsafe { conv.read_f32(0, qkv.len()) },
            &qkv,
        );
    let actual_gates = unsafe { gates.read_f32(0, alpha.len() * 2) };
    let actual_beta = actual_gates
        .chunks_exact(2)
        .map(|v| v[1])
        .collect::<Vec<_>>();
    diff += compare("beta", &actual_beta, &read("recurrent-beta"));
    diff += compare(
        "recurrent+gates",
        &unsafe { out.read_f32(0, value.len()) },
        &read("recurrent-output"),
    );
    for (buffer, count) in [(&conv, qkv.len()), (&out, value.len())] {
        assert!(
            unsafe { buffer.read_u32(count + 32) }[count..]
                .iter()
                .all(|&v| v == u32::MAX)
        );
    }
    // Reuse the existing GPU exponential to observe the actual decay used by
    // recurrence. Its minus sign is removed only for the diagnostic comparison.
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "bonsai_a",
        &[&gates],
        &[(alpha.len() * 2) as u32],
        [(alpha.len() * 2).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    let decay = unsafe { gates.read_f32(0, alpha.len() * 2) }
        .chunks_exact(2)
        .map(|v| -v[0])
        .collect::<Vec<_>>();
    diff += compare("decay", &decay, &read("recurrent-decay"));

    let norm_input = read("linear_attn.norm-input");
    let norm_gate = read("linear_attn.norm-gate");
    let norm_expected = read("linear_attn.norm-output");
    let norm_rows = norm_input.len() / (g.value_heads * 128);
    assert!(norm_rows > 0);
    assert_eq!(norm_input.len(), norm_rows * g.value_heads * 128);
    assert_eq!(norm_input.len(), norm_gate.len());
    assert_eq!(norm_input.len(), norm_expected.len());
    let norm = upload(&norm_input);
    let z = upload(&norm_gate);
    let mut p = p;
    p[3] = norm_rows as u32;
    p[6] = model.eps.to_bits();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "mlx_dn_gated_norm",
        &[&norm, &z, &w.norm.buffer],
        &p,
        [g.value_heads, norm_rows, 1],
        32,
    );
    cmd.finish().unwrap();
    diff += compare(
        "gated_norm",
        &unsafe { norm.read_f32(0, norm_input.len()) },
        &norm_expected,
    );
    for layer in [0, 3] {
        let plane =
            |name: &str| std::fs::read(root.join(format!("layer-{layer}-mlp.{name}.f32"))).unwrap();
        let gate = d.upload(&plane("gate_proj-output")).unwrap();
        let up = d.upload(&plane("up_proj-output")).unwrap();
        let expected = plane("down_proj-input");
        let count = expected.len() / 4;
        assert_eq!(gate.len(), expected.len());
        assert_eq!(up.len(), expected.len());
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            if precise {
                "mlx_swiglu_precise"
            } else {
                "mlx_swiglu"
            },
            &[&gate, &up],
            &[count as u32],
            [count.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let expected = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        diff += compare(
            &format!("layer-{layer}-swiglu"),
            &unsafe { gate.read_f32(0, count) },
            &expected,
        );
    }
    if std::env::var_os("PADDOCK_METAL_REQUIRE_PARITY").is_some() {
        assert_eq!(diff, 0, "recurrent operation contracts differ from MLX");
    }
}
