//! Native GPT-OSS: paged F16 attention and GPU-routed MXFP4 experts.
//! Slot ownership, prefix retention and chunked admission use the same engine
//! contracts as the dense Metal graphs. No external executor or host experts.
use crate::device::{Buffer, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use std::collections::VecDeque;
mod forward;
mod load;
mod serving;
#[cfg(test)]
mod tests;

const CHUNK: usize = 512;
const SPLITS: usize = 16;
const WIDTH: usize = 2880;
const QWIDTH: usize = 4096;
const KVWIDTH: usize = 512;
const VOCAB: usize = 201088;
// Wider output tiles reuse gathered activations and halve the grid width.
// BN128 exceeds the compiled M5 threadgroup-memory limit; BN64 is guarded
// by device::tests::gpt_oss_pipelines_fit_threadgroup_memory.
const MOE_NTILE: usize = 64;

struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qb: Weight,
    kb: Weight,
    vb: Weight,
    ob: Weight,
    sinks: Weight,
    post: Weight,
    router: Weight,
    router_bias: Weight,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    gate_bias: Weight,
    up_bias: Weight,
    down_bias: Weight,
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
    qhalf: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    parts: Buffer,
    delta: Buffer,
    gemm: Buffer,
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

pub struct GptOss {
    cold: Option<crate::paged_offload::PagedTier>,
    source_versions: Vec<crate::offload::FileVersion>,
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
    experts: usize,
    context: usize,
    page_stride: usize,
    rope: [u32; 5],
    eps: f32,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}

impl GptOss {
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
            "gpt-oss-v1:{}:{}:{:?}:{}:{}",
            self.context,
            self.layers.len(),
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
                "invalid GPT-OSS prefill slot/tokens/context".into(),
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
