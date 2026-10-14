//! Head normalization against the separately dispatched GPU RMS primitive.
//! The independent nonzero-position MLX capture lives in the language gate.
use super::*;

#[test]
fn contiguous_head_norm_preserves_gates_pages_and_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload = |v: &[f32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let words = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let weight: Vec<_> = (0..256)
        .map(|i| half::bf16::from_f32(0.5 + (i % 97) as f32 / 64.).to_f32())
        .collect();
    let w = upload(&weight);
    for heads in [8usize, 16] {
        for rows in [1usize, 17, 33] {
            let count = rows * heads * 256;
            let q: Vec<_> = (0..count)
                .map(|i| half::bf16::from_f32(((i * 43 % 257) as f32 - 128.) / 13.).to_f32())
                .collect();
            // NaN gates detect accidental normalization of the gated half.
            let qraw: Vec<_> = q
                .chunks_exact(256)
                .flat_map(|h| h.iter().copied().chain(std::iter::repeat_n(f32::NAN, 256)))
                .collect();
            let raw = upload(&qraw);
            let compact = upload(&q);
            let reference = d.alloc(count * 4).unwrap();
            let output = d.upload(&vec![0xff; (count + 32) * 4]).unwrap();
            let key_count = rows * 2 * 256;
            let k = upload(&q[..key_count]);
            let key_reference = d.alloc(key_count * 4).unwrap();
            let blocks = rows.div_ceil(16);
            let kv_count = blocks * 16 * 2 * 256;
            let keys = d.upload(&vec![0xff; (kv_count + 32) * 2]).unwrap();
            let values = d.upload(&vec![0xff; (kv_count + 32) * 2]).unwrap();
            let meta = words(&(0..rows as u32).flat_map(|r| [1, r]).collect::<Vec<_>>());
            let pages = words(
                &std::iter::repeat_n(u32::MAX, blocks)
                    .chain((0..blocks as u32).rev())
                    .collect::<Vec<_>>(),
            );
            let positions = words(&vec![0; rows * 4]);
            let p = [
                heads as u32,
                2,
                blocks as u32,
                10_000_000f32.to_bits(),
                1e-6f32.to_bits(),
                64,
            ];
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "mlx_rms",
                &[&compact, &w, &reference],
                &[256, 0, 1e-6f32.to_bits()],
                [rows * heads, 1, 1],
                64,
            );
            cmd.dispatch(
                "mlx_rms",
                &[&k, &w, &key_reference],
                &[256, 0, 1e-6f32.to_bits()],
                [rows * 2, 1, 1],
                64,
            );
            cmd.dispatch(
                "mlx_qnorm_rope_contiguous",
                &[&raw, &w, &positions, &output],
                &p,
                [heads, rows, 1],
                64,
            );
            cmd.dispatch(
                "mlx_knorm_store_contiguous",
                &[&k, &k, &w, &meta, &pages, &keys, &values, &positions],
                &p,
                [2, rows, 1],
                64,
            );
            cmd.finish().unwrap();
            assert_eq!(unsafe { output.read_u32(count) }, unsafe {
                reference.read_u32(count)
            });
            assert!(
                unsafe { output.read_f32(count, 32) }
                    .iter()
                    .all(|v| v.is_nan())
            );
            let expected = unsafe { key_reference.read_f32(0, key_count) };
            let unpack = |b: &Buffer| {
                unsafe { b.read_u32((kv_count + 32) / 2) }
                    .into_iter()
                    .flat_map(|w| [w as u16, (w >> 16) as u16])
                    .collect::<Vec<_>>()
            };
            let key_bits = unpack(&keys);
            let value_bits = unpack(&values);
            let mut written = vec![false; kv_count + 32];
            for row in 0..rows {
                let physical = (blocks - 1 - row / 16) * 16 + row % 16;
                for col in 0..512 {
                    let src = row * 512 + col;
                    let dst = physical * 512 + col;
                    written[dst] = true;
                    assert_eq!(key_bits[dst], half::bf16::from_f32(expected[src]).to_bits());
                    assert_eq!(value_bits[dst], half::bf16::from_f32(q[src]).to_bits());
                }
            }
            for (i, used) in written.into_iter().enumerate() {
                if !used {
                    assert_eq!(key_bits[i], u16::MAX);
                    assert_eq!(value_bits[i], u16::MAX);
                }
            }
        }
    }
}
