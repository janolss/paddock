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
const SPLITS: usize = 16;
const PREFIX_PER_SLOT: usize = 2;

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
}
impl Nemotron {
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
        Ok(reused)
    }
    fn publish(&mut self, slot: usize) -> Result<()> {
        let n = self.slots[slot].history.len();
        if n == 0 || !n.is_multiple_of(BLOCK_TOKENS) {
            return Ok(());
        }
        let history = &self.slots[slot].history;
        let index = if self.prefix_limit() == 1 {
            Some(0)
        } else {
            prefix::replacement(&self.prefixes, history)
        };
        let Some(index) = index else { return Ok(()) };
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
        Ok(())
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.pending.iter().any(|p| p.slot == slot) {
            return Err(MetalError::Model("slot already prefilling".into()));
        }
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
            self.publish(slot)?;
        }
        Ok(last)
    }
}
