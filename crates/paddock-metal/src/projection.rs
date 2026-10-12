//! Sequence-local affine arithmetic contracts. Nemotron MLX tensor prefill
//! uses its BF16 boundaries; narrow live rows retain their SIMD reduction
//! when unrelated prompts join. Historical F32 tiles remain GPU witnesses.
use crate::device::{Buffer, Commands};
use crate::weights::Weight;

#[cfg(test)]
thread_local! {
    static AFFINE_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static AFFINE_MLX: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static AFFINE_PACKED_EXPERTS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
pub(crate) fn affine_packed_experts() -> bool {
    #[cfg(test)]
    {
        AFFINE_PACKED_EXPERTS.get()
    }
    #[cfg(not(test))]
    {
        true
    }
}
#[cfg(test)]
pub(crate) fn with_unpacked_experts<T>(run: impl FnOnce() -> T) -> T {
    // Also retain the pre-reuse dense/vector staging as a whole-graph witness.
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            AFFINE_PACKED_EXPERTS.set(self.0);
        }
    }
    let _reset = Reset(AFFINE_PACKED_EXPERTS.replace(false));
    run()
}

#[cfg(test)]
pub(crate) fn affine_reference() -> bool {
    AFFINE_REFERENCE.get()
}
pub(crate) fn affine_mlx() -> bool {
    #[cfg(test)]
    {
        AFFINE_MLX.get() && !affine_reference()
    }
    #[cfg(not(test))]
    {
        true
    }
}
#[cfg(test)]
pub(crate) fn with_affine_legacy<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            AFFINE_MLX.set(self.0);
        }
    }
    let _reset = Reset(AFFINE_MLX.replace(false));
    run()
}
#[cfg(test)]
pub(crate) fn with_affine_mlx<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            AFFINE_MLX.set(self.0);
        }
    }
    let _reset = Reset(AFFINE_MLX.replace(true));
    run()
}

#[cfg(test)]
pub(crate) fn with_affine_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            AFFINE_REFERENCE.set(self.0);
        }
    }
    let _reset = Reset(AFFINE_REFERENCE.replace(true));
    run()
}

pub(crate) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
) {
    for &(w, out) in planes {
        if w.ty == 0x108 {
            #[cfg(test)]
            if affine_reference() {
                cmd.dispatch(
                    if rows < 16 {
                        "dg_affine_vector"
                    } else {
                        "dg_project"
                    },
                    &[&w.buffer, input, out],
                    &[w.k as u32, w.n as u32, rows as u32, w.ty, 1],
                    [
                        w.n.div_ceil(if rows < 16 { 4 } else { 32 }),
                        rows.div_ceil(if rows < 16 { 1 } else { 16 }),
                        1,
                    ],
                    128,
                );
                continue;
            }
            let (kernel, columns, tile) = if rows == 1 {
                if affine_packed_experts() {
                    ("nemo_mlx_vector1_reuse", 8, 1)
                } else {
                    ("nemo_mlx_vector1", 4, 1)
                }
            } else if rows < 16 {
                if affine_packed_experts() {
                    ("nemo_mlx_vector4_reuse", 8, 4)
                } else {
                    ("nemo_mlx_vector4", 4, 4)
                }
            } else if rows < 32 {
                ("nemo_mlx_project_exact", 32, 16)
            } else {
                ("nemo_mlx_project_exact_wide", 64, 32)
            };
            cmd.dispatch(
                kernel,
                &[&w.buffer, input, out],
                &[w.k as u32, w.n as u32, rows as u32, w.ty, 1],
                [w.n.div_ceil(columns), rows.div_ceil(tile), 1],
                128,
            );
            continue;
        }
        let (kernel, columns, tile) = match w.ty {
            12..=14 => match rows {
                1 => ("linear_kquant1", 4, 1),
                2 => ("linear_kquant2", 16, 2),
                3 => ("linear_kquant3", 16, 3),
                4 => ("linear_kquant4", 16, 4),
                5 => ("gemma_full5", 16, 5),
                6 => ("gemma_full6", 16, 6),
                7 => ("gemma_full7", 16, 7),
                8 => ("gemma_full8", 16, 8),
                _ => ("laguna_dense_f32", 16, 32),
            },
            8 => match rows {
                1 => ("linear_q8_r1", 4, 1),
                2..=4 => ("linear_q8_r4", 4, 4),
                5..=8 => ("linear_q8_r8", 4, 8),
                9..=15 => ("linear_q8_r16", 4, 16),
                _ => ("laguna_dense_f32", 16, 32),
            },
            _ => unreachable!("validated precise projection type"),
        };
        cmd.dispatch(
            kernel,
            &[&w.buffer, input, out],
            &[w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(columns), rows.div_ceil(tile), 1],
            128,
        );
    }
}
