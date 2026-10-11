//! Native Laguna XS/S: original quantized experts, partial YaRN and gated
//! full/sliding attention. The scheduler owns slot lifetimes; GPU routing
//! never reads back expert IDs or expands the checkpoint into dense weights.
use crate::device::{Buffer, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use std::collections::VecDeque;
mod forward;
#[cfg(test)]
mod kernel_tests;
mod load;
use crate::projection;
mod serving;
#[cfg(test)]
mod tests;

const CHUNK: usize = 512;
const SPLITS: usize = 16;
const KVWIDTH: usize = 1024;
const VOCAB: usize = 100352;
const EXPERTS: usize = 256;

// Top-k is an architecture property. Separate compiled reductions avoid
// introducing runtime division and changed unrolling into the XS graph.
fn expert_kernel(active: usize, name: &'static str) -> &'static str {
    match (active, name) {
        (8, _) | (10, "laguna_down_decode") => name,
        (10, "laguna_route") => "laguna_route_top10",
        (10, "laguna_gu_decode") => "laguna_gu_decode_top10",
        (10, "laguna_gu_grouped16") => "laguna_gu_grouped16_top10",
        (10, "laguna_gu_grouped32") => "laguna_gu_grouped32_top10",
        (10, "laguna_down_grouped16") => "laguna_down_grouped16_top10",
        (10, "laguna_down_grouped32") => "laguna_down_grouped32_top10",
        (10, "laguna_fold") => "laguna_fold_top10",
        _ => unreachable!("validated Laguna expert geometry/kernel"),
    }
}

/// Two checkpoint geometries, not an arbitrary family-wide admission. Keep
/// allocation strides and kernel dispatch derived from the same election.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    width: usize,
    layers: usize,
    ff: usize,
    active: usize,
    sliding_heads: usize,
    dense_ff: usize,
}
impl Geometry {
    fn for_width(width: usize) -> Option<Self> {
        let (layers, ff, active, sliding_heads, dense_ff) = match width {
            2048 => (40, 512, 8, 64, 8192),
            3072 => (48, 1024, 10, 72, 12288),
            _ => return None,
        };
        Some(Self {
            width,
            layers,
            ff,
            active,
            sliding_heads,
            dense_ff,
        })
    }
    fn heads(self, layer: usize) -> usize {
        if layer.is_multiple_of(4) {
            48
        } else {
            self.sliding_heads
        }
    }
    fn beta_fast(self) -> f32 {
        if self.width == 2048 { 64. } else { 32. }
    }
}

struct Experts {
    router: Weight,
    bias: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}
struct Layer {
    heads: usize,
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    qnorm: Weight,
    knorm: Weight,
    gate: Weight,
    o: Weight,
    post: Weight,
    fg: Weight,
    fu: Weight,
    fd: Weight,
    experts: Option<Experts>,
    keys: Buffer,
    values: Buffer,
}
struct Scratch {
    ids: Buffer,
    meta: Buffer,
    pages: Buffer,
    output_rows: Buffer,
    decode_rows: Buffer,
    attention_tiles: Buffer,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    gate: Buffer,
    attn: Buffer,
    parts: Buffer,
    delta: Buffer,
    fg: Buffer,
    fu: Buffer,
    router: Buffer,
    picks: Buffer,
    probabilities: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    gu: Buffer,
    expert_out: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
pub struct Laguna {
    cold: Option<crate::paged_offload::PagedTier>,
    source_versions: Vec<crate::offload::FileVersion>,
    geometry: Geometry,
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    pool: KvPool,
    radix: PagedRadix,
    context: usize,
    page_stride: usize,
    rope: [u32; 5],
    eps: f32,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}

impl Laguna {
    /// Enable once after all companions attach, before admitting requests.
    pub fn enable_kv_offload(
        &mut self,
        config: crate::KvOffloadConfig,
        paths: &[&std::path::Path],
    ) -> Result<()> {
        if self.cold.is_some() || self.slots.iter().any(|s| !s.history.is_empty()) {
            return Err(MetalError::Model(
                "enable KV offload once before inference".into(),
            ));
        }
        let layout = format!(
            "laguna-v1:{}:{:?}:{:?}:{}:{}",
            self.context,
            self.geometry,
            self.rope,
            self.eps.to_bits(),
            self.device.checkpoint_platform()
        );
        let planes = self
            .layers
            .iter()
            .flat_map(|l| {
                [
                    (&l.keys, BLOCK_TOKENS * (KVWIDTH) * 2),
                    (&l.values, BLOCK_TOKENS * (KVWIDTH) * 2),
                ]
            })
            .collect::<Vec<_>>();
        self.cold = Some(crate::paged_offload::PagedTier::open(
            &self.device,
            config,
            paths,
            layout.as_bytes(),
            &self.source_versions,
            &planes,
            self.context,
        )?);
        Ok(())
    }
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
        {
            return Err(MetalError::Model(
                "invalid Laguna prefill slot/tokens/context".into(),
            ));
        }
        let s = &mut self.slots[slot];
        s.table.clear(&mut self.pool);
        s.history.clear();
        let blocks = self.radix.match_prefix(tokens);
        let reused = blocks.len() * BLOCK_TOKENS;
        s.table.share_prefix(&blocks, &mut self.pool);
        s.history.extend_from_slice(&tokens[..reused]);
        s.reused = reused;
        Ok(reused)
    }
    fn publish(&mut self, slot: usize) {
        let s = &self.slots[slot];
        if let Some(tier) = &mut self.cold {
            let planes = self
                .layers
                .iter()
                .flat_map(|l| {
                    [
                        (&l.keys, BLOCK_TOKENS * (KVWIDTH) * 2),
                        (&l.values, BLOCK_TOKENS * (KVWIDTH) * 2),
                    ]
                })
                .collect::<Vec<_>>();
            tier.capture(&self.device, &planes, &s.history, s.table.blocks());
        }
        self.radix
            .insert(&s.history, s.table.blocks(), &mut self.pool);
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        for chunk in tokens[reused..].chunks(CHUNK) {
            let pos = self.slots[slot].history.len();
            let rows = chunk
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect::<Vec<_>>();
            let output = if pos + chunk.len() == tokens.len() {
                vec![chunk.len() - 1]
            } else {
                Vec::new()
            };
            last = self.execute(&rows, &output)?;
        }
        self.publish(slot);
        Ok(last)
    }
}
