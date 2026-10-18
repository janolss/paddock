//! Exact memory model of the native Gemma 4 Metal loader.
//!
//! The loader (`paddock-metal`, `gemma4/load.rs` and `gemma4/mlx.rs`) refuses a
//! grant below `weights + KV + scratch`, so the manager has to price those
//! terms with the same arithmetic. The generic estimator cannot express this
//! layout: global layers keep `2 * batch + 1` sequences of pages, while every
//! sliding layer keeps two ring banks of `window + CHUNK` tokens per slot no
//! matter how long the context is. Keep every constant in step with the loader.

use paddock_estimator::ModelShape;

const BLOCK_TOKENS: u64 = 16;
const CHUNK: u64 = 512;
const HEADS: u64 = 32;
const SPLITS: u64 = 32;
const WINDOW: u64 = 1024;
const VOCAB: u64 = 262_144;
/// F16 KV, the only cache width the Metal loader has.
const KV_ELEM_BYTES: u64 = 2;

/// The two graphs the loader's allowlist accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gemma4Metal {
    layers: u64,
    sliding_dim: u64,
    global_dim: u64,
    width: u64,
    ff: u64,
    moe: bool,
}

impl Gemma4Metal {
    const DENSE_31B: Self = Self {
        layers: 60,
        sliding_dim: 4096,
        global_dim: 2048,
        width: 5376,
        ff: 21504,
        moe: false,
    };
    const MOE_26B_A4B: Self = Self {
        layers: 30,
        sliding_dim: 2048,
        global_dim: 1024,
        width: 2816,
        ff: 2112,
        moe: true,
    };

    /// Recognise the loader's exact geometry from the published layer list:
    /// five sliding layers (1024-token window) then one global layer, repeated.
    pub(crate) fn detect(shape: &ModelShape) -> Option<Self> {
        let candidate = [Self::DENSE_31B, Self::MOE_26B_A4B]
            .into_iter()
            .find(|g| g.layers == shape.kv_layers.len() as u64)?;
        let matches = shape.vocab == VOCAB
            && shape.kv_layers.iter().enumerate().all(|(i, l)| {
                let (dim, window) = if i % 6 != 5 {
                    (candidate.sliding_dim, Some(WINDOW))
                } else {
                    (candidate.global_dim, None)
                };
                l.k_dim == dim && l.v_dim == dim && l.window == window
            });
        matches.then_some(candidate)
    }

    /// Bytes of K and V for `ctx` tokens and `batch` slots, as the loader
    /// allocates them up front.
    pub(crate) fn kv_bytes(&self, ctx: u64, batch: u64) -> u64 {
        let pages = ctx.div_ceil(BLOCK_TOKENS);
        let ring = ctx.min(WINDOW + CHUNK).next_multiple_of(BLOCK_TOKENS);
        let global_layers = self.layers / 6;
        let sliding_layers = global_layers * 5;
        let global = pages
            .saturating_mul(2 * batch + 1)
            .saturating_mul(BLOCK_TOKENS * self.global_dim * KV_ELEM_BYTES);
        let sliding = ring * batch * 2 * self.sliding_dim * KV_ELEM_BYTES;
        global
            .saturating_mul(global_layers)
            .saturating_add(sliding * sliding_layers)
            * 2
    }

    /// The per-slot page table, the only scratch term that grows with context.
    pub(crate) fn page_table_bytes(ctx: u64, batch: u64) -> u64 {
        batch * ctx.div_ceil(BLOCK_TOKENS) * 4
    }

    /// Scratch the loader pins regardless of context: activation planes, the
    /// split-attention partials, the logits rows, the GEMM workspace and, for
    /// the MoE graph, the expert staging planes.
    pub(crate) fn scratch_bytes(&self, batch: u64) -> u64 {
        let ff = if self.moe { self.ff.max(8192) } else { self.ff };
        let planes = CHUNK * (self.width * 5 + HEADS * 512 * 2 + 4096 * 2 + ff * 2 + 9) * 4;
        let per_slot = batch * (VOCAB + HEADS * SPLITS * 514) * 4;
        let gemm = (CHUNK * ff).max((CHUNK + 32) * HEADS * 512) * 2;
        planes + per_slot + gemm + if self.moe { moe_workspace_bytes() } else { 0 }
    }
}

/// `moe::Workspace::bytes(CHUNK)`: eleven f32 planes sized by the staged rows.
fn moe_workspace_bytes() -> u64 {
    const WIDTH: u64 = 2816;
    const FF: u64 = 704;
    const EXPERTS: u64 = 128;
    const ACTIVE: u64 = 8;
    let rows = CHUNK;
    let elems: u64 = [
        rows * WIDTH,
        rows * WIDTH,
        rows * EXPERTS,
        rows * ACTIVE,
        rows * ACTIVE,
        EXPERTS * rows * ACTIVE,
        EXPERTS,
        1 + 2 * ((rows * ACTIVE).div_ceil(16) + EXPERTS),
        rows * ACTIVE * FF * 2,
        rows * ACTIVE * WIDTH,
        rows * WIDTH,
    ]
    .into_iter()
    .sum();
    elems * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values were computed independently from the loader's source
    // expressions (gemma4/load.rs kv/scratch sizing, gemma4/moe.rs sizes).
    #[test]
    fn dense_31b_matches_the_loader_arithmetic() {
        let g = Gemma4Metal::DENSE_31B;
        assert_eq!(g.kv_bytes(8192, 1), 4_529_848_320);
        assert_eq!(g.kv_bytes(8192, 4), 16_106_127_360);
        assert_eq!(g.kv_bytes(32768, 1), 10_569_646_080);
        assert_eq!(g.kv_bytes(32768, 4), 34_225_520_640);
        assert_eq!(g.scratch_bytes(1), 252_209_152);
        assert_eq!(g.scratch_bytes(4), 261_670_912);
        assert_eq!(Gemma4Metal::page_table_bytes(8192, 1), 2048);
    }

    #[test]
    fn moe_26b_matches_the_loader_arithmetic() {
        let g = Gemma4Metal::MOE_26B_A4B;
        assert_eq!(moe_workspace_bytes(), 88_903_172);
        assert_eq!(g.kv_bytes(8192, 1), 1_132_462_080);
        assert_eq!(g.kv_bytes(8192, 4), 4_026_531_840);
        assert_eq!(g.kv_bytes(32768, 1), 2_642_411_520);
        assert_eq!(g.kv_bytes(32768, 4), 8_556_380_160);
        assert_eq!(g.scratch_bytes(1), 256_177_668);
        assert_eq!(g.scratch_bytes(4), 265_639_428);
    }

    #[test]
    fn sliding_ring_stops_growing_past_window_plus_chunk() {
        let g = Gemma4Metal::MOE_26B_A4B;
        // 1536 = WINDOW + CHUNK: beyond it only the 5 global layers grow.
        let grown = g.kv_bytes(2048, 1) - g.kv_bytes(1536, 1);
        let pages = (2048 - 1536) / BLOCK_TOKENS;
        assert_eq!(
            grown,
            pages * 3 * BLOCK_TOKENS * g.global_dim * KV_ELEM_BYTES * 5 * 2
        );
    }
}
