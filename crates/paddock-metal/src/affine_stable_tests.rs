//! Independent scalar oracle for the explicit affine contraction, including
//! ragged fused planes, distinct/reordered inputs and partially filled tiles.
use crate::device::MetalDevice;
use crate::{affine, weights::Weight};

fn bf(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

#[test]
fn llama_affine_cohorts_preserve_isolated_contracts_and_scratch_grants() {
    let device = MetalDevice::new(None).unwrap();
    let lengths = [1usize, 1, 4, 29, 17, 33, 34];
    let rows: usize = lengths.iter().sum();
    for (k, n) in [(2048, 256), (2048, 2048), (2048, 6144), (6144, 2048)] {
        let mut bytes: Vec<_> = (0..k * n / 8)
            .flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes())
            .collect();
        for bias in [false, true] {
            bytes.extend((0..k * n / 64).flat_map(|i| {
                let s = (i % 251 + 1) as f32 / 9973.;
                half::bf16::from_f32(if bias { -7.5 * s } else { s })
                    .to_bits()
                    .to_le_bytes()
            }));
        }
        let weight = Weight {
            buffer: device.upload(&bytes).unwrap(),
            ty: affine::AFFINE4,
            k,
            n,
        };
        let values: Vec<_> = (0..rows * k)
            .map(|i| bf(((i * 37) % 1999) as f32 / 113. - 8.))
            .collect();
        let input = device
            .upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.alloc((rows * n + 32) * 4).unwrap();
        let mut spans = Vec::new();
        let mut first = 0;
        for &count in &lengths {
            spans.push((first, count, count));
            first += count;
        }
        for capacity in [34, 512] {
            // The smaller grant forces compatible cohorts to split before
            // their partial-result tail would overrun existing workspace.
            let bytes = affine::workspace_bytes(k, n, capacity);
            let workspace = device.alloc(bytes).unwrap();
            let mut expected = Vec::new();
            for &(first, count, _) in &spans {
                let x = device
                    .upload(
                        &values[first * k..(first + count) * k]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let cmd = device.begin().unwrap();
                affine::project_llama(&cmd, &[(&weight, &output)], &x, count, &workspace);
                cmd.finish().unwrap();
                expected.extend(unsafe { output.read_f32(0, count * n) });
            }
            unsafe {
                output.write_u32(&vec![u32::MAX; rows * n + 32]);
            }
            let cmd = device.begin().unwrap().with_projection_rows(&spans);
            affine::project_llama(&cmd, &[(&weight, &output)], &input, rows, &workspace);
            cmd.finish().unwrap();
            let actual = unsafe { output.read_f32(0, rows * n) };
            assert!(actual.iter().all(|v| v.is_finite()));
            assert_eq!(actual, expected, "k={k} n={n} capacity={capacity}");
            assert!(
                unsafe { output.read_u32(rows * n + 32) }[rows * n..]
                    .iter()
                    .all(|v| *v == u32::MAX)
            );
        }
    }
}

#[test]
fn stable_affine_mixed_phases_preserve_isolated_rows_and_guards() {
    stable_affine_mixed_phases(false);
}

#[test]
fn lighton_bf16_decode_preserves_mixed_phases_and_guards() {
    stable_affine_mixed_phases(true);
}

fn stable_affine_mixed_phases(bf16_bias_decode: bool) {
    let device = MetalDevice::new(None).expect("Metal mixed-phase device");
    let widths = if bf16_bias_decode {
        [2560usize, 4096, 9216]
    } else {
        [5120, 6144, 17408]
    };
    for k in widths {
        let ns = [48usize, 65, 129];
        let weights: Vec<_> = ns
            .iter()
            .enumerate()
            .map(|(plane, &n)| {
                let mut bytes: Vec<_> = (0..k * n / 8)
                    .flat_map(|i| {
                        ((i + plane * 127) as u32)
                            .wrapping_mul(2654435761)
                            .to_le_bytes()
                    })
                    .collect();
                for bias in [false, true] {
                    bytes.extend((0..k * n / 64).flat_map(|i| {
                        let s = (i % 251 + 1) as f32 / 9973.;
                        half::bf16::from_f32(if bias { -7.5 * s } else { s })
                            .to_bits()
                            .to_le_bytes()
                    }));
                }
                Weight {
                    buffer: device.upload(&bytes).expect("affine test weights"),
                    ty: affine::AFFINE4,
                    k,
                    n,
                }
            })
            .collect();
        let x = device
            .upload(
                &(0..512 * k)
                    .flat_map(|i| bf(((i * 37) % 1999) as f32 / 113. - 8.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("affine test input");
        let outputs: Vec<_> = ns
            .iter()
            .map(|n| device.alloc((512 * n + 32) * 4).expect("guarded output"))
            .collect();
        let workspace_bytes = ns
            .iter()
            .map(|n| affine::workspace_bytes(k, *n, 512))
            .max()
            .expect("nonempty projection group");
        let workspace = device
            .alloc(workspace_bytes + 128)
            .expect("guarded workspace");
        let planes: Vec<_> = weights.iter().zip(&outputs).collect();
        let cmd = device
            .begin()
            .expect("prefill command")
            .with_affine_prefill_rows(if bf16_bias_decode { 2048 } else { 512 });
        affine::project(&cmd, &planes, &x, 512, &workspace);
        cmd.finish().expect("prefill reference execution");
        let prefill: Vec<_> = planes
            .iter()
            .map(|(w, out)| unsafe { out.read_f32(0, 31 * w.n) })
            .collect();
        let cmd = device.begin().expect("decode reference command");
        affine::project_stable(
            &cmd,
            &planes,
            &x,
            31,
            &workspace,
            &[(0, 31, 1)],
            bf16_bias_decode,
        );
        cmd.finish().expect("decode reference execution");
        let decode: Vec<_> = planes
            .iter()
            .map(|(w, out)| unsafe { out.read_f32(0, 31 * w.n) })
            .collect();
        for spans in [
            vec![(0, 1, 512)],
            vec![(0, 2, 512)],
            vec![(0, 12, 512)],
            vec![(0, 13, 512)],
            vec![(0, 1, 1), (1, 30, 512)],
            vec![(0, 12, 512), (12, 4, 1), (16, 15, 512)],
            vec![(0, 1, 512), (1, 1, 1), (2, 1, 512), (3, 1, 1)],
        ] {
            for out in &outputs {
                unsafe {
                    out.write_u32(&vec![u32::MAX; out.len() / 4]);
                }
            }
            unsafe {
                workspace.write_u32(&vec![u32::MAX; workspace.len() / 4]);
            }
            let rows = spans.iter().map(|s| s.1).sum();
            let cmd = device.begin().expect("mixed-phase command");
            affine::project_stable(
                &cmd,
                &planes,
                &x,
                rows,
                &workspace,
                &spans,
                bf16_bias_decode,
            );
            cmd.finish().expect("mixed-phase execution");
            for (plane, (w, out)) in planes.iter().enumerate() {
                let got = unsafe { out.read_f32(0, 512 * w.n + 32) };
                for &(first, count, phase) in &spans {
                    let expected = if phase == 1 {
                        &decode[plane]
                    } else {
                        &prefill[plane]
                    };
                    for i in first * w.n..(first + count) * w.n {
                        assert_eq!(
                            got[i].to_bits(),
                            expected[i].to_bits(),
                            "k={k} n={} spans={spans:?} i={i}",
                            w.n
                        );
                    }
                }
                assert!(got[rows * w.n..].iter().all(|v| v.to_bits() == u32::MAX));
            }
            assert!(
                unsafe { workspace.read_f32(workspace_bytes / 4, 32) }
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
        }
    }
}

#[test]
fn stable_affine_matches_explicit_scalar_contract() {
    let device = MetalDevice::new(None).unwrap();
    for k in [64usize, 192, 512, 5120, 6144, 17408] {
        let ns = [17usize, 9, 1];
        let max_rows = 17;
        let inputs: Vec<f32> = (0..max_rows * k)
            .map(|i| {
                let permuted_row = (i / k * 7 + 3) % max_rows;
                let code = ((permuted_row * k + i % k) as u32).wrapping_mul(2654435761);
                let signed = (code >> 8) as i32 - (1 << 23);
                bf(signed as f32 * 2f32.powi((code % 17) as i32 - 31))
            })
            .collect();
        let packed: Vec<u8> = inputs
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let input = device
            .upload(&[packed.as_slice(), &[0xa5; 128]].concat())
            .unwrap();
        let mut weights = Vec::new();
        let mut expected = Vec::new();
        let mut outputs = Vec::new();
        for (plane, &n) in ns.iter().enumerate() {
            let codes: Vec<u32> = (0..k * n / 8)
                .map(|i| (i as u32 + plane as u32 * 131).wrapping_mul(2654435761))
                .collect();
            let scales: Vec<f32> = (0..k * n / 64)
                .map(|i| bf((i % 127) as f32 * 2f32.powi((i % 13) as i32 - 20)))
                .collect();
            let biases: Vec<f32> = scales
                .iter()
                .enumerate()
                .map(|(i, &s)| bf(if i % 7 == 0 { s * 3.25 } else { -s * 7.5 }))
                .collect();
            let mut bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
            bytes.extend(
                scales
                    .iter()
                    .chain(&biases)
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes()),
            );
            weights.push(device.upload(&bytes).unwrap());
            outputs.push(device.alloc((max_rows * n + 32) * 4).unwrap());
            let mut values = Vec::new();
            for row in 0..max_rows {
                for col in 0..n {
                    let mut lanes = [0f32; 8];
                    for (lane, sum) in lanes.iter_mut().enumerate() {
                        for base in (lane * 64..k).step_by(512) {
                            let group = col * (k / 64) + base / 64;
                            for word in 0..8 {
                                let bits = codes[col * (k / 8) + base / 8 + word];
                                let mut v = 0f32;
                                for j in 0..8 {
                                    let code = ((bits >> (j * 4)) & 15) as f32;
                                    let weight = code.mul_add(scales[group], biases[group]);
                                    let at = row * k + base + word * 8 + j;
                                    v = if j == 0 {
                                        inputs[at] * weight
                                    } else {
                                        inputs[at].mul_add(weight, v)
                                    };
                                }
                                *sum += v;
                            }
                        }
                    }
                    for shift in [4, 2, 1] {
                        for lane in 0..shift {
                            lanes[lane] += lanes[lane + shift];
                        }
                    }
                    values.push(bf(lanes[0]).to_bits());
                }
            }
            expected.push(values);
        }
        for m in [1usize, 2, 3, 4, 5, 7, 12, 13, 17] {
            let r = m.div_ceil(m.div_ceil(5));
            for planes in 1..=3 {
                for (output, &n) in outputs.iter().zip(&ns) {
                    unsafe {
                        output.write_u32(&vec![u32::MAX; max_rows * n + 32]);
                    }
                }
                let second = usize::from(planes >= 2);
                let third = if planes == 3 { 2 } else { second };
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    &format!("mlx_affine_stable{r}"),
                    &[
                        &weights[0],
                        &weights[second],
                        &weights[third],
                        &input,
                        &outputs[0],
                        &outputs[second],
                        &outputs[third],
                    ],
                    &[
                        k as u32,
                        ns[0] as u32,
                        if planes >= 2 { ns[1] as u32 } else { 0 },
                        if planes == 3 { ns[2] as u32 } else { 0 },
                        m as u32,
                    ],
                    [
                        ns[..planes].iter().map(|n| n.div_ceil(8)).sum(),
                        m.div_ceil(r),
                        1,
                    ],
                    64,
                );
                cmd.finish().unwrap();
                for (plane, (&n, output)) in ns.iter().zip(&outputs).enumerate() {
                    let actual = unsafe { output.read_f32(0, max_rows * n + 32) };
                    let valid = if plane < planes { m * n } else { 0 };
                    for (i, value) in actual[..valid].iter().enumerate() {
                        assert_eq!(
                            value.to_bits(),
                            expected[plane][i],
                            "K={k} M={m} planes={planes} plane={plane} i={i}"
                        );
                    }
                    assert!(actual[valid..].iter().all(|v| v.to_bits() == u32::MAX));
                }
            }
        }
        let actual = unsafe { input.read_f32(packed.len() / 4, 32) };
        assert!(actual.iter().all(|v| v.to_bits() == 0xa5a5a5a5));
    }
}
