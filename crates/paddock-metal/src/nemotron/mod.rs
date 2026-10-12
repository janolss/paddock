//! Native elected Nemotron Lightning Q8: packed mixed requests, F32 SSD
//! state and grouped GPU-only ReLU² experts. Speculative heads/NVFP4 are
//! deliberately separate graphs, not inferred from the family name.
use crate::device::{Buffer, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_models::nemotron::NemotronBlock;
use std::collections::VecDeque;
mod forward;
mod load;
mod mlx;
#[cfg(test)]
mod mlx_tests;
mod prefix;
#[cfg(test)]
mod prefix_tests;
mod serving;
#[cfg(test)]
mod tests;

const WIDTH: usize = 2688;
const VOCAB: usize = 131072;
const INNER: usize = 4096;
const CONV: usize = 6144;
const PROJECTED: usize = 10304;
const STATE: usize = 524288;
const WINDOW: usize = CONV * 3;
const FF: usize = 1856;
const SHARED: usize = 3712;
const CHUNK: usize = 512;
const MLX_CHUNK: usize = 2048;
const SPLITS: usize = 16;
const PREFIX_PER_SLOT: usize = 2;
const MLX_PREFIX_PER_SLOT: usize = 3;

struct Mamba {
    input: Weight,
    output: Weight,
    conv_w: Weight,
    conv_b: Weight,
    a: Weight,
    d: Weight,
    dt: Weight,
    norm: Weight,
    // Live slots + bounded complete prefix snapshots, always F32.
    state: Buffer,
    window: Buffer,
}
struct Attention {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    keys: Buffer,
    values: Buffer,
}
struct Experts {
    router: Weight,
    bias: Weight,
    up: Weight,
    down: Weight,
    shared_up: Weight,
    shared_down: Weight,
}
enum Mixer {
    Mamba(Mamba),
    Attention(Attention),
    Moe(Experts),
}
struct Layer {
    norm: Weight,
    mixer: Mixer,
}
struct Scratch {
    ids: Buffer,
    meta: Buffer,
    pages: Buffer,
    output_rows: Buffer,
    decode_rows: Buffer,
    attention_tiles: Buffer,
    sequences: Buffer,
    ssd_tiles: Buffer,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    proj: Buffer,
    conv: Buffer,
    y: Buffer,
    yn: Buffer,
    dt: Buffer,
    decay: Buffer,
    matrix: Buffer,
    state_delta: Buffer,
    state_in: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    parts: Buffer,
    router: Buffer,
    picks: Buffer,
    probabilities: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    up: Buffer,
    expert_out: Buffer,
    shared: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
    prompt_end: usize,
    tool_started: bool,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
pub struct Nemotron {
    device: MetalDevice,
    embedding: Weight,
    head: Weight,
    output_norm: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    // KV refs and recurrent snapshots are published atomically after GPU
    // completion. Attention-only radix reuse is invalid for a Mamba model.
    prefixes: Vec<prefix::Checkpoint>,
    prefix_clock: u64,
    // Tokenizer-derived trailing-whitespace tokens are unsafe reply cuts:
    // template trimming can merge their final newline on a later user turn.
    reply_trim_tokens: Vec<bool>,
    pool: KvPool,
    context: usize,
    page_stride: usize,
    weight_bytes: u64,
    cache_bytes: u64,
    pub last_gpu_seconds: f64,
    #[cfg(test)]
    scan_only: bool,
    #[cfg(test)]
    single_snapshot: bool,
    #[cfg(test)]
    reference_inputs: Vec<Buffer>,
}
impl Nemotron {
    fn mixed_row_cap(&self, decodes: usize) -> usize {
        mixed_row_cap(
            self.embedding.ty == mlx::AFFINE8 && self.device.tensor_accelerated(),
            decodes,
            self.scratch.ids.len() / 4,
        )
    }
    fn pending_span(&self, offset: usize, length: usize) -> usize {
        if self.embedding.ty != mlx::AFFINE8 {
            return length - offset;
        }
        // Large cold work must not erase the first user anchor or the last
        // resumable cut before a changed/truncated final token. Mixed work
        // retains the shared 128-row latency cap independently of this arena.
        let cut = if offset < CHUNK {
            CHUNK
        } else {
            length.saturating_sub(1) / CHUNK * CHUNK
        };
        if offset < cut && cut < length {
            cut - offset
        } else {
            length - offset
        }
    }
    fn copy_state(&self, dst: usize, src: Option<usize>) -> Result<()> {
        if let Some(src) = src {
            // Copy bytes, not arithmetic: one fenced blit replaces 46 compute
            // dispatches for the complete Mamba state and convolution windows.
            let copies = self
                .layers
                .iter()
                .filter_map(|l| match &l.mixer {
                    Mixer::Mamba(m) => Some(m),
                    _ => None,
                })
                .flat_map(|m| {
                    [(&m.state, STATE * 4), (&m.window, WINDOW * 4)]
                        .map(|(b, n)| (b, src * n, b, dst * n, n))
                })
                .collect::<Vec<_>>();
            self.device.copy_regions(&copies)
        } else {
            self.copy_state_kernel(dst, None)
        }
    }
    fn copy_state_kernel(&self, dst: usize, src: Option<usize>) -> Result<()> {
        let cmd = self.device.begin()?;
        for l in &self.layers {
            if let Mixer::Mamba(m) = &l.mixer {
                for (b, n) in [(&m.state, STATE), (&m.window, WINDOW)] {
                    cmd.dispatch(
                        "nemo_state_copy",
                        &[b],
                        &[n as u32, dst as u32, src.map_or(u32::MAX, |n| n as u32)],
                        [n.div_ceil(256), 1, 1],
                        256,
                    );
                }
            }
        }
        cmd.finish()?;
        Ok(())
    }
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
        {
            return Err(MetalError::Model(
                "invalid Nemotron prefill slot/tokens/context".into(),
            ));
        }
        // A warm HTTP tool turn is not a new conversation. Peers may be
        // awaiting tool results with no occupied slot; keep their anchors.
        // Only an admission outside the held conversations retires stale pins.
        if !self
            .prefixes
            .iter()
            .any(|c| c.anchor && tokens.starts_with(&c.history))
        {
            let active = self
                .slots
                .iter()
                .enumerate()
                .filter(|(i, s)| *i != slot && !s.history.is_empty())
                .map(|(_, s)| s.history.as_slice())
                .chain(self.pending.iter().map(|p| p.tokens.as_slice()))
                .chain(std::iter::once(tokens))
                .collect::<Vec<_>>();
            prefix::retain_active_anchors(&mut self.prefixes, &active);
        }
        let matched = self
            .prefixes
            .iter()
            .take(self.prefix_limit())
            .enumerate()
            .filter(|(_, c)| {
                !c.history.is_empty()
                    && c.history.len() < tokens.len()
                    && tokens.starts_with(&c.history)
            })
            .max_by_key(|(_, c)| c.history.len())
            .map(|(i, c)| (i, c.history.len()));
        let reused = matched.map_or(0, |(_, n)| n);
        self.copy_state(slot, matched.map(|(i, _)| self.slots.len() + i))?;
        let s = &mut self.slots[slot];
        s.table.clear(&mut self.pool);
        s.history.clear();
        if let Some((index, _)) = matched {
            s.table
                .share_prefix(self.prefixes[index].table.blocks(), &mut self.pool);
            self.prefix_clock += 1;
            self.prefixes[index].touched = self.prefix_clock;
        }
        s.history.extend_from_slice(&tokens[..reused]);
        s.reused = reused;
        s.prompt_end = tokens.len();
        s.tool_started = false;
        Ok(reused)
    }
    fn publish(&mut self, slot: usize) -> Result<()> {
        let n = self.slots[slot].history.len();
        // MLX fair-share grants need not end on a page boundary. Retain the
        // complete state at their actual end; execute copy-on-writes the last
        // attention page before either the live slot or a restore appends.
        if n == 0 || (self.embedding.ty != mlx::AFFINE8 && !n.is_multiple_of(BLOCK_TOKENS)) {
            return Ok(());
        }
        let history = &self.slots[slot].history;
        let index = if self.prefix_limit() == 1 {
            Some(0)
        } else {
            prefix::replacement(&self.prefixes, history)
        };
        let Some(index) = index else { return Ok(()) };
        self.file_checkpoint(slot, index, false)
    }
    fn file_checkpoint(&mut self, slot: usize, index: usize, reply_candidate: bool) -> Result<()> {
        // Never leave an old key pointing at a partially overwritten state if
        // the copy fails. Only the successful fence publishes a new pairing.
        self.prefixes[index].clear(&mut self.pool);
        self.copy_state(self.slots.len() + index, Some(slot))?;
        self.prefixes[index]
            .table
            .share_prefix(self.slots[slot].table.blocks(), &mut self.pool);
        self.prefixes[index]
            .history
            .clone_from(&self.slots[slot].history);
        self.prefix_clock += 1;
        self.prefixes[index].touched = self.prefix_clock;
        self.prefixes[index].reply_candidate = reply_candidate;
        Ok(())
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.pending.iter().any(|p| p.slot == slot) {
            return Err(MetalError::Model("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        let mut offset = reused;
        while offset < tokens.len() {
            let count = self
                .pending_span(offset, tokens.len())
                .min(self.scratch.ids.len() / 4);
            let chunk = &tokens[offset..offset + count];
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
            self.publish(slot)?;
            offset += count;
        }
        Ok(last)
    }
}

fn mixed_row_cap(mlx_tensor: bool, decodes: usize, capacity: usize) -> usize {
    if mlx_tensor && decodes > 0 {
        // Qualify the allowance in prompt rows, separately from live decodes.
        // At 8K/16K depth the 128-row pass is dominated by fixed expert work;
        // a 512-row ceiling lets the shared measured-cost pacer find the knee.
        // It remains free to shrink the grant, never exceed this fixed arena.
        decodes.saturating_add(512).min(capacity)
    } else {
        crate::schedule::row_cap(decodes, capacity)
    }
}
