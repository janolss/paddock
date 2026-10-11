//! Dense Granite/Llama Metal graph, including the MiniCPM5 MLX checkpoint.
//! Metadata controls scales and rotary conventions; physical KV pages are
//! owned here, logical sharing by KvPool.
use crate::device::Commands;
use crate::device::{Buffer, MetalDevice, MetalError, Result};
mod llama;
#[cfg(test)]
mod minicpm_tests;
mod mlx;
mod multimodal;
mod projection;
mod source;
mod speech;
#[cfg(test)]
mod tests;
mod vision;
use paddock_engine::generator::{GenError, Generator};
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;
use std::collections::VecDeque;
use std::path::Path;

const CHUNK: usize = 512;
const MAX_SPLITS: usize = 32;

use crate::weights::{Weight, projections};

struct Layer {
    norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    ffn_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    keys: Buffer,
    values: Buffer,
}

struct Scratch {
    gemm_input: Buffer,
    ids: Buffer,
    output_rows: Buffer,
    attention_rows: Buffer,
    attention_tiles: Buffer,
    meta: Buffer,
    pages: Buffer,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    attn_parts: Buffer,
    delta: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
    mm: Option<multimodal::Layout>,
    audio: Vec<speech::admission::Span>,
    radix_tokens: Vec<u32>,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}

pub struct Granite {
    mlx: bool,
    cold: Option<crate::paged_offload::PagedTier>,
    source_versions: Vec<crate::offload::FileVersion>,
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Option<Weight>,
    layers: Vec<Layer>,
    scratch: Scratch,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    pool: KvPool,
    radix: PagedRadix,
    vision: Option<vision::Vision>,
    audio: Option<speech::Tower>,
    audio_id: Option<u32>,
    speech_plus: bool,
    audio_encoding: VecDeque<speech::admission::Encoding>,
    deepstack: Vec<Option<usize>>,
    image_id: Option<u32>,
    encoding: VecDeque<multimodal::Encoding>,
    image_cache: VecDeque<multimodal::Cached>,
    next_image_symbol: u32,
    image_cache_reused: u64,
    width: usize,
    kv_width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ff: usize,
    vocab: usize,
    context: usize,
    page_stride: usize,
    eps: f32,
    rope: f32,
    embedding_scale: f32,
    residual_scale: f32,
    logit_scale: f32,
    attention_scale: f32,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}

impl Granite {
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
        // MLX's sequence-owned projection tree and probability contraction
        // must not reuse KV persisted by the earlier physical-batch graph.
        let namespace = if self.mlx {
            "llama-mlx-v2"
        } else if self.direct_prefill() {
            "llama-gguf-paged-v2"
        } else {
            "granite-v1"
        };
        let layout = format!(
            "{namespace}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            self.context,
            self.layers.len(),
            self.width,
            self.kv_width,
            self.eps.to_bits(),
            self.rope.to_bits(),
            self.embedding_scale.to_bits(),
            self.residual_scale.to_bits(),
            self.attention_scale.to_bits(),
            self.device.checkpoint_platform(),
            self.mlx
        );
        let planes = self
            .layers
            .iter()
            .flat_map(|l| {
                [
                    (&l.keys, BLOCK_TOKENS * (self.kv_width) * 2),
                    (&l.values, BLOCK_TOKENS * (self.kv_width) * 2),
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
    /// Load GGUF or the elected MLX checkpoint without requantizing weights.
    /// The context and concurrency grant must fit before allocating physical KV
    /// or running work.
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let source_versions = crate::offload::versions(path)?;
        let map = source::Source::open(path)?;
        let mlx = map.is_mlx();
        let arch = map.gguf().architecture().unwrap_or("");
        if !matches!(arch, "granite" | "llama") {
            return Err(MetalError::Model(
                "Metal dense decoder requires general.architecture=granite or llama".into(),
            ));
        }
        if arch == "llama" {
            llama::validate(map.gguf())?;
        }
        let u = |key: &str| -> Result<usize> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0 && n <= u32::MAX as usize)
                .ok_or_else(|| MetalError::Model(format!("invalid/missing {arch}.{key}")))
        };
        let f = |key: &str| -> Result<f32> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|n| n.is_finite())
                .ok_or_else(|| MetalError::Model(format!("invalid/missing {arch}.{key}")))
        };
        let width = u("embedding_length")?;
        let heads = u("attention.head_count")?;
        let kv_heads = u("attention.head_count_kv")?;
        let count = u("block_count")?;
        let ff = u("feed_forward_length")?;
        let trained = u("context_length")?;
        let head_dim = width / heads;
        if width % heads != 0
            || !matches!(head_dim, 64 | 128)
            || heads % kv_heads != 0
            || u("rope.dimension_count")? != head_dim
            || (head_dim == 64 && heads != kv_heads * 5)
        {
            return Err(MetalError::Model(
                "Metal Granite requires full rotary and hd128 integral GQA or hd64 GQA5".into(),
            ));
        }
        if context == 0 || context > trained || max_batch == 0 || max_batch > CHUNK {
            return Err(MetalError::Model(format!(
                "context must be 1..={trained}, batch 1..={CHUNK}"
            )));
        }
        if map.gguf().arch_field("rope.scaling.type").is_some() {
            return Err(MetalError::Model(
                "rotary scaling is not implemented for Metal Granite".into(),
            ));
        }
        let deepstack = match map.gguf().arch_field("deepstack_mapping") {
            None => Vec::new(),
            Some(Value::Array(a)) if a.len() == count => {
                let mut expected = vec![-1i64; count];
                if width != 2560 || count != 40 {
                    return Err(MetalError::Model(
                        "unqualified Granite DeepStack geometry".into(),
                    ));
                }
                for stream in 1..8 {
                    expected[stream * 3] = stream as i64;
                }
                if a.iter().map(Value::as_i64).collect::<Option<Vec<_>>>() != Some(expected.clone())
                {
                    return Err(MetalError::Model(
                        "invalid Granite DeepStack mapping".into(),
                    ));
                }
                expected
                    .into_iter()
                    .map(|s| usize::try_from(s).ok())
                    .collect()
            }
            _ => {
                return Err(MetalError::Model(
                    "invalid Granite DeepStack metadata".into(),
                ));
            }
        };
        let image_id = map
            .gguf()
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                Value::Array(a) => a
                    .iter()
                    .position(|t| t.as_str() == Some("<image>"))
                    .map(|i| i as u32),
                _ => None,
            });
        if !deepstack.is_empty() && image_id.is_none() {
            return Err(MetalError::Model(
                "Granite vision image marker missing".into(),
            ));
        }
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let audio_id = match map.gguf().metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(a)) => a
                .iter()
                .position(|t| t.as_str() == Some("<|audio|>"))
                .map(|i| i as u32),
            _ => None,
        };
        let speech_plus = ["general.finetune", "general.name"].iter().any(|k| {
            map.gguf()
                .metadata
                .get(*k)
                .and_then(Value::as_str)
                .is_some_and(|s| s.to_lowercase().ends_with("plus"))
        });
        if audio_id.is_some()
            && (width != 2048
                || count != 40
                || heads != 16
                || kv_heads != 4
                || ff != 4096
                || context > 4096
                || max_batch > 16
                || !deepstack.is_empty())
        {
            return Err(MetalError::Model(
                "Granite Speech requires the elected 2B graph, context <=4096 and batch <=16"
                    .into(),
            ));
        }
        if audio_id.is_some() {
            let expected = if speech_plus { 362 } else { 363 };
            if map.gguf().tensors.len() != expected
                || map.gguf().tensors.iter().any(|t| {
                    t.raw_type
                        != if t.name.ends_with("_norm.weight") {
                            0
                        } else {
                            8
                        }
                })
            {
                return Err(MetalError::Model(
                    "Granite Speech Metal requires the elected Q8/F32 decoder inventory".into(),
                ));
            }
        }
        let rope = f("rope.freq_base")?;
        let (embedding_scale, residual_scale, logit_scale, attention_scale) = if arch == "llama" {
            (1.0, 1.0, 1.0, 1.0 / (head_dim as f32).sqrt())
        } else {
            (
                f("embedding_scale")?,
                f("residual_scale")?,
                f("logit_scale")?,
                f("attention.scale")?,
            )
        };
        if eps <= 0.0 || rope <= 0.0 || logit_scale == 0.0 {
            return Err(MetalError::Model(
                "invalid normalization or rotary scale".into(),
            ));
        }
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| t.dims.get(1))
            .copied()
            .ok_or_else(|| MetalError::Model("embedding table missing".into()))?
            as usize;
        // Kernels use 32-bit dispatch/row indices and 64-bit weight offsets.
        // Prove the former cannot wrap before any Metal buffer is allocated.
        if [width, ff, vocab]
            .iter()
            .any(|&n| n == 0 || n > u32::MAX as usize / CHUNK)
        {
            return Err(MetalError::Model(
                "model dimensions exceed Metal dispatch limits".into(),
            ));
        }
        let max_matrix = width * width.max(ff).max(vocab);
        if max_matrix > u32::MAX as usize {
            return Err(MetalError::Model(
                "matrix exceeds Metal preparation grid limits".into(),
            ));
        }
        let kv_width = kv_heads * head_dim;
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        // One full context per admitted slot plus bounded shared-prefix retention.
        // Use checked arithmetic for externally supplied context/concurrency.
        let blocks = page_stride
            .checked_mul(max_batch + 1)
            .filter(|&n| n <= u32::MAX as usize / BLOCK_TOKENS)
            .ok_or_else(|| MetalError::Memory("KV page count overflow".into()))?;
        let kv_layer = blocks
            .checked_mul(BLOCK_TOKENS)
            .and_then(|n| n.checked_mul(kv_width * 2))
            .ok_or_else(|| MetalError::Memory("KV byte count overflow".into()))?;
        let device = MetalDevice::new(budget)?;
        let scratch_bytes = CHUNK as u64
            * ((width * 5 + kv_width * 2 + ff * 2 + heads * MAX_SPLITS * (head_dim + 2)) * 4 + 28)
                as u64
            + (max_batch * (vocab + page_stride) * 4) as u64;
        let kv_bytes = (kv_layer as u64)
            .checked_mul(2 * count as u64)
            .ok_or_else(|| MetalError::Memory("KV total overflow".into()))?;
        let gemm_elements = (CHUNK * width.max(ff).div_ceil(128) * 128).max((CHUNK + 32) * width);
        if gemm_elements > u32::MAX as usize {
            return Err(MetalError::Memory(
                "activation workspace exceeds Metal index range".into(),
            ));
        }
        let gemm_bytes = if mlx {
            [
                (width, width),
                (width, kv_width),
                (width, ff),
                (ff, width),
                (width, vocab),
            ]
            .into_iter()
            .map(|(k, n)| crate::affine::workspace_bytes(k, n, CHUNK))
            .max()
            .unwrap_or(0)
            .max(gemm_elements * 2)
        } else {
            gemm_elements * 2
        };
        let required = map
            .weight_budget_bytes()
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes)
            .saturating_add(gemm_bytes as u64);
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "weights + KV + scratch need {:.2} GiB; grant {:.2} GiB",
                required as f64 / (1u64 << 30) as f64,
                device.budget_bytes() as f64 / (1u64 << 30) as f64
            )));
        }
        let embedding = map.load(&device, "token_embd.weight", &[width, vocab])?;
        let output_norm = map.load(&device, "output_norm.weight", &[width])?;
        let head = if map.tensor_info("output.weight").is_some() {
            Some(map.load(&device, "output.weight", &[width, vocab])?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let w = |name: &str, dims: &[usize]| {
                map.load(&device, &format!("blk.{i}.{name}.weight"), dims)
            };
            layers.push(Layer {
                norm: w("attn_norm", &[width])?,
                q: w("attn_q", &[width, width])?,
                k: w("attn_k", &[width, kv_width])?,
                v: w("attn_v", &[width, kv_width])?,
                o: w("attn_output", &[width, width])?,
                ffn_norm: w("ffn_norm", &[width])?,
                gate: w("ffn_gate", &[width, ff])?,
                up: w("ffn_up", &[width, ff])?,
                down: w("ffn_down", &[ff, width])?,
                keys: device.alloc(kv_layer)?,
                values: device.alloc(kv_layer)?,
            });
        }
        let weight_bytes = device.allocated_bytes() - kv_bytes;
        let a = |n: usize| device.alloc(CHUNK * n * 4);
        let scratch = Scratch {
            gemm_input: device.alloc(gemm_bytes)?,
            ids: a(1)?,
            output_rows: a(1)?,
            attention_rows: a(1)?,
            attention_tiles: a(2)?,
            meta: a(2)?,
            pages: device.alloc(max_batch * page_stride * 4)?,
            x: a(width)?,
            norm: a(width)?,
            q: a(width)?,
            k: a(kv_width)?,
            v: a(kv_width)?,
            attn: a(width)?,
            delta: a(width)?,
            gate: a(ff)?,
            up: a(ff)?,
            logits: device.alloc(max_batch * vocab * 4)?,
            attn_parts: a(heads * MAX_SPLITS * (head_dim + 2))?,
        };
        tracing::info!(
            arch,
            layers = count,
            width,
            vocab,
            weight_bytes,
            kv_bytes,
            "Dense decoder weights loaded on Metal"
        );
        Ok(Self {
            mlx,
            cold: None,
            source_versions,
            device,
            embedding,
            output_norm,
            head,
            layers,
            scratch,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            pool: KvPool::with_blocks(blocks as u32),
            radix: PagedRadix::new(),
            vision: None,
            audio: None,
            audio_id,
            speech_plus,
            audio_encoding: VecDeque::new(),
            deepstack,
            image_id,
            encoding: VecDeque::new(),
            image_cache: VecDeque::new(),
            next_image_symbol: vocab as u32,
            image_cache_reused: 0,
            width,
            kv_width,
            heads,
            kv_heads,
            head_dim,
            ff,
            vocab,
            context,
            page_stride,
            eps,
            rope,
            embedding_scale,
            residual_scale,
            logit_scale,
            attention_scale,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.0,
        })
    }

    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        self.prepare_layout(slot, tokens, None)
    }

    fn prepare_layout(
        &mut self,
        slot: usize,
        tokens: &[u32],
        layout: Option<multimodal::Layout>,
    ) -> Result<usize> {
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
            || tokens.iter().any(|t| Some(*t) == self.audio_id)
        {
            return Err(MetalError::Model(
                "invalid prefill slot, tokens or context".into(),
            ));
        }
        self.slots[slot].table.clear(&mut self.pool);
        self.slots[slot].history.clear();
        self.slots[slot].audio.clear();
        let radix_tokens = layout
            .as_ref()
            .map_or_else(|| tokens.to_vec(), |l| l.radix_tokens());
        let blocks = self.radix.match_prefix(&radix_tokens);
        self.slots[slot].radix_tokens = radix_tokens;
        self.slots[slot].mm = layout;
        let reused = blocks.len() * BLOCK_TOKENS;
        self.slots[slot].table.share_prefix(&blocks, &mut self.pool);
        self.slots[slot]
            .history
            .extend_from_slice(&tokens[..reused]);
        self.slots[slot].reused = reused;
        Ok(reused)
    }

    fn publish(&mut self, slot: usize) {
        let s = &self.slots[slot];
        // Placeholder IDs are not audio identities. Until content-keyed audio
        // cache publication is implemented, no audio KV may enter text radix.
        if !s.audio.is_empty() {
            return;
        }
        // Resident image symbols are process-local; never persist those under
        // a text digest. The text tier stays safe on vision/speech endpoints.
        if s.mm.is_none()
            && let Some(tier) = &mut self.cold
        {
            let planes = self
                .layers
                .iter()
                .flat_map(|l| {
                    [
                        (&l.keys, BLOCK_TOKENS * (self.kv_width) * 2),
                        (&l.values, BLOCK_TOKENS * (self.kv_width) * 2),
                    ]
                })
                .collect::<Vec<_>>();
            tier.capture(&self.device, &planes, &s.history, s.table.blocks());
        }
        let keys = s
            .history
            .iter()
            .enumerate()
            .map(|(i, &t)| s.radix_tokens.get(i).copied().unwrap_or(t))
            .collect::<Vec<_>>();
        self.radix.insert(&keys, s.table.blocks(), &mut self.pool);
    }

    /// A single weight-amortized pass across arbitrary (slot, token, position)
    /// rows. Metadata is shared by every layer, including the paged KV writer.
    fn execute(&mut self, rows: &[(usize, u32, u32)], output_rows: &[usize]) -> Result<Vec<f32>> {
        // The engine thread is Rust-owned, with no Cocoa event-loop pool.
        // Drain transient Objective-C objects after every synchronous pass.
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, output_rows))
    }

    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        if rows.is_empty() || rows.len() > CHUNK {
            return Err(MetalError::Model("invalid execution row count".into()));
        }
        if output_rows.len() > self.slots.len() || output_rows.iter().any(|&r| r >= rows.len()) {
            return Err(MetalError::Model("invalid output row selection".into()));
        }
        let mut lengths: Vec<usize> = self.slots.iter().map(|s| s.history.len()).collect();
        for &(slot, tok, pos) in rows {
            if slot >= self.slots.len()
                || tok as usize >= self.vocab
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid row slot={slot}, token={tok}, position={pos}"
                )));
            }
            lengths[slot] += 1;
        }
        for &(slot, _, pos) in rows {
            while self.slots[slot]
                .table
                .ensure(pos as usize, &mut self.pool)
                .is_err()
            {
                if self.radix.evict_lru(&mut self.pool).is_none() {
                    return Err(MetalError::Memory("KV pool exhausted".into()));
                }
            }
        }
        let ids: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let meta: Vec<u32> = rows.iter().flat_map(|r| [r.0 as u32, r.2]).collect();
        let mut pages = vec![0u32; self.slots.len() * self.page_stride];
        for (i, s) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + s.table.blocks().len()]
                .copy_from_slice(s.table.blocks());
        }
        let s = &self.scratch;
        let m = rows.len();
        // Plan once for all layers. A query tile must not cross a sequence
        // boundary; short decode segments instead expose KV-split parallelism.
        let mut attention_plan = Vec::new();
        let mut first = 0;
        while first < m {
            let mut end = first + 1;
            while end < m && rows[end].0 == rows[first].0 {
                end += 1;
            }
            let count = end - first;
            let splits = (rows[end - 1].2 as usize + 1)
                .div_ceil(256)
                .min(512usize.div_ceil(count * self.heads))
                .clamp(1, MAX_SPLITS);
            attention_plan.push((first, count, splits));
            first = end;
        }
        let attention_tiles: Vec<u32> = attention_plan
            .iter()
            .filter(|&&(_, count, _)| count >= 16)
            .flat_map(|&(first, count, _)| {
                (0..count).step_by(32).flat_map(move |offset| {
                    [(first + offset) as u32, (count - offset).min(32) as u32]
                })
            })
            .collect();
        let gqa8 = self.head_dim == 128
            && self.heads == self.kv_heads * 8
            && self.attention_scale.to_bits() == (1.0 / 128f32.sqrt()).to_bits();
        #[cfg(test)]
        let gqa8 = gqa8 && !minicpm_tests::BASELINE_ATTENTION.get();
        let grouped_decode =
            self.mlx || self.head_dim == 64 || self.heads == self.kv_heads * 4 || gqa8;
        let attention_rows: Vec<u32> = if grouped_decode {
            attention_plan
                .iter()
                .filter(|&&(_, count, _)| count < 16)
                .flat_map(|&(first, count, _)| (first..first + count).map(|r| r as u32))
                .collect()
        } else {
            Vec::new()
        };
        let attention_length = attention_rows
            .iter()
            .map(|&r| rows[r as usize].2 as usize + 1)
            .max()
            .unwrap_or(1);
        let occupancy_splits = if attention_length > 256 {
            attention_length
                .div_ceil(32)
                .min(32_usize.div_ceil(attention_rows.len().max(1)))
        } else {
            1
        };
        let attention_splits = attention_length
            .div_ceil(128)
            .max(8_usize.div_ceil(attention_rows.len().max(1)))
            .max(occupancy_splits)
            .clamp(1, MAX_SPLITS);
        // SAFETY: execute is synchronous and the preceding submission completed.
        unsafe {
            s.ids.write_u32(&ids);
            s.meta.write_u32(&meta);
            s.pages.write_u32(&pages);
            s.output_rows
                .write_u32(&output_rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.attention_rows.write_u32(&attention_rows);
            s.attention_tiles.write_u32(&attention_tiles);
        }
        let cmd = self.device.begin()?;
        let projection_spans: Vec<_> = attention_plan
            .iter()
            .filter(|_| self.mlx)
            .map(|&(first, count, _)| (first, count, count))
            .collect();
        let cmd = if self.mlx {
            cmd.with_projection_rows(&projection_spans)
        } else {
            cmd
        };
        let norm = |input: &Buffer, w: &Weight, out: &Buffer| {
            cmd.dispatch(
                if self.mlx { "mlx_rms" } else { "rms" },
                &[input, &w.buffer, out],
                &[self.width as u32, w.ty, self.eps.to_bits()],
                [m, 1, 1],
                if self.mlx {
                    (self.width.div_ceil(128) * 32).min(1024)
                } else {
                    256
                },
            )
        };
        let residual_norm = |w: &Weight| {
            cmd.dispatch(
                if self.mlx {
                    "mlx_residual_rms"
                } else {
                    "residual_rms"
                },
                &[&s.x, &s.delta, &w.buffer, &s.norm],
                &[
                    self.width as u32,
                    w.ty,
                    self.eps.to_bits(),
                    self.residual_scale.to_bits(),
                ],
                [m, 1, 1],
                if self.mlx {
                    (self.width.div_ceil(128) * 32).min(1024)
                } else {
                    256
                },
            );
        };
        cmd.dispatch(
            if self.mlx { "mlx_embed" } else { "embed" },
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[
                self.width as u32,
                m as u32,
                if self.mlx {
                    self.vocab as u32
                } else {
                    self.embedding.ty
                },
                self.embedding_scale.to_bits(),
            ],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
        let image_rows = self.has_image_rows(rows);
        if image_rows {
            self.inject_images(&cmd, rows, 0);
        }
        self.inject_audio(&cmd, rows);
        norm(&s.x, &self.layers[0].norm, &s.norm);
        for (index, layer) in self.layers.iter().enumerate() {
            self.project(
                &cmd,
                &[(&layer.q, &s.q), (&layer.k, &s.k), (&layer.v, &s.v)],
                &s.norm,
                m,
                1.0,
            );
            if self.mlx {
                self.mlx_attention(
                    &cmd,
                    layer,
                    m,
                    attention_tiles.len() / 2,
                    attention_rows.len(),
                    attention_splits,
                );
            } else {
                cmd.dispatch(
                    "rope_store",
                    &[
                        &s.q,
                        &s.k,
                        &s.v,
                        &layer.keys,
                        &layer.values,
                        &s.meta,
                        &s.pages,
                    ],
                    &[
                        self.width as u32,
                        self.kv_width as u32,
                        self.head_dim as u32,
                        m as u32,
                        self.page_stride as u32,
                        self.rope.to_bits(),
                    ],
                    [(m * (self.width + self.kv_width) / 2).div_ceil(256), 1, 1],
                    256,
                );
                if !attention_tiles.is_empty() {
                    cmd.dispatch(
                        "attention_query",
                        &[&s.q, &s.gemm_input],
                        &[self.width as u32, 0, m as u32],
                        [((m + 32) * self.width).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        if self.direct_prefill() {
                            "llama_prefill_direct"
                        } else if self.head_dim == 64 {
                            "attention_prefill_batched64"
                        } else {
                            "granite_attention_prefill128"
                        },
                        &[
                            &s.gemm_input,
                            &layer.keys,
                            &layer.values,
                            &s.meta,
                            &s.pages,
                            &s.attn,
                            &s.attention_tiles,
                        ],
                        &[
                            self.heads as u32,
                            self.kv_heads as u32,
                            self.page_stride as u32,
                            self.attention_scale.to_bits(),
                        ],
                        [self.heads, attention_tiles.len() / 2, 1],
                        128,
                    );
                }
                for &(first, count, splits) in &attention_plan {
                    if count >= 16 {
                        continue;
                    }
                    if grouped_decode {
                        continue;
                    }
                    cmd.dispatch(
                        "attention",
                        &[
                            &s.q,
                            &layer.keys,
                            &layer.values,
                            &s.meta,
                            &s.pages,
                            if splits == 1 { &s.attn } else { &s.attn_parts },
                        ],
                        &[
                            self.heads as u32,
                            self.kv_heads as u32,
                            128,
                            first as u32,
                            self.page_stride as u32,
                            self.attention_scale.to_bits(),
                            splits as u32,
                        ],
                        [self.heads, count, splits],
                        32,
                    );
                    if splits > 1 {
                        cmd.dispatch(
                            "attention_merge",
                            &[&s.attn_parts, &s.attn],
                            &[splits as u32, (first * self.heads) as u32],
                            [count * self.heads, 1, 1],
                            32,
                        );
                    }
                }
                if !attention_rows.is_empty() && gqa8 {
                    cmd.dispatch(
                        "llama_decode",
                        &[
                            &s.q,
                            &layer.keys,
                            &layer.values,
                            &s.meta,
                            &s.pages,
                            &s.attention_rows,
                            &s.attn_parts,
                        ],
                        &[
                            self.heads as u32,
                            self.kv_heads as u32,
                            self.page_stride as u32,
                            0,
                            0,
                            attention_splits as u32,
                        ],
                        [self.kv_heads, attention_rows.len(), attention_splits],
                        128,
                    );
                    cmd.dispatch(
                        "muse_merge",
                        &[&s.attn_parts, &s.attn, &s.attention_rows],
                        &[self.heads as u32, attention_splits as u32, 128],
                        [self.heads * attention_rows.len(), 1, 1],
                        32,
                    );
                } else if !attention_rows.is_empty() {
                    cmd.dispatch(
                        if self.head_dim == 64 {
                            "attention_gqa5_64"
                        } else {
                            "attention_gqa4"
                        },
                        &[
                            &s.q,
                            &layer.keys,
                            &layer.values,
                            &s.meta,
                            &s.pages,
                            &s.attention_rows,
                            if attention_splits == 1 {
                                &s.attn
                            } else {
                                &s.attn_parts
                            },
                        ],
                        &[
                            self.heads as u32,
                            self.kv_heads as u32,
                            self.page_stride as u32,
                            self.attention_scale.to_bits(),
                            attention_rows.len() as u32,
                            attention_splits as u32,
                        ],
                        [self.kv_heads, attention_rows.len(), attention_splits],
                        128,
                    );
                    if attention_splits > 1 {
                        cmd.dispatch(
                            if self.head_dim == 64 {
                                "attention_gqa_merge64"
                            } else {
                                "attention_gqa_merge"
                            },
                            &[&s.attn_parts, &s.attn, &s.attention_rows],
                            &[attention_splits as u32, self.heads as u32],
                            [attention_rows.len() * self.heads, 1, 1],
                            32,
                        );
                    }
                }
            }
            self.project(&cmd, &[(&layer.o, &s.delta)], &s.attn, m, 1.0);
            residual_norm(&layer.ffn_norm);
            self.project(
                &cmd,
                &[(&layer.gate, &s.gate), (&layer.up, &s.up)],
                &s.norm,
                m,
                1.0,
            );
            cmd.dispatch(
                if self.mlx { "mlx_swiglu" } else { "swiglu" },
                &[&s.gate, &s.up],
                &[(m * self.ff) as u32],
                [(m * self.ff).div_ceil(256), 1, 1],
                256,
            );
            self.project(&cmd, &[(&layer.down, &s.delta)], &s.gate, m, 1.0);
            if let Some(next) = self.layers.get(index + 1) {
                if image_rows && let Some(stream) = self.deepstack.get(index + 1).copied().flatten()
                {
                    // The previous layer's residual precedes the unscaled,
                    // additive tap; normalization must see both contributions.
                    cmd.dispatch(
                        "residual",
                        &[&s.x, &s.delta],
                        &[(m * self.width) as u32, self.residual_scale.to_bits()],
                        [(m * self.width).div_ceil(256), 1, 1],
                        256,
                    );
                    self.inject_images(&cmd, rows, stream);
                    norm(&s.x, &next.norm, &s.norm);
                } else {
                    residual_norm(&next.norm);
                }
            } else {
                cmd.dispatch(
                    if self.mlx { "mlx_residual" } else { "residual" },
                    &[&s.x, &s.delta],
                    &[(m * self.width) as u32, self.residual_scale.to_bits()],
                    [(m * self.width).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                if self.mlx {
                    "mlx_rms_selected"
                } else {
                    "rms_selected"
                },
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[self.width as u32, self.output_norm.ty, self.eps.to_bits()],
                [output_rows.len(), 1, 1],
                if self.mlx {
                    (self.width.div_ceil(128) * 32).min(1024)
                } else {
                    256
                },
            );
            let head = self.head.as_ref().unwrap_or(&self.embedding);
            if self.mlx {
                crate::affine::project_verify(
                    &cmd,
                    &[(head, &s.logits)],
                    &s.norm,
                    output_rows.len(),
                    &s.gemm_input,
                    true,
                );
            } else {
                self.project(
                    &cmd,
                    &[(head, &s.logits)],
                    &s.norm,
                    output_rows.len(),
                    1.0 / self.logit_scale,
                );
            }
        }
        self.last_gpu_seconds = cmd.finish()?;
        for &(slot, tok, _) in rows {
            self.slots[slot].history.push(tok);
        }
        // SAFETY: command completion made all logits visible to the host.
        Ok(unsafe { s.logits.read_f32(0, output_rows.len() * self.vocab) })
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.pending.iter().any(|p| p.slot == slot)
            || self.image_slot_pending(slot)
            || self.audio_slot_pending(slot)
        {
            return Err(MetalError::Model("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        for chunk in tokens[reused..].chunks(CHUNK) {
            let pos = self.slots[slot].history.len();
            let rows: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect();
            // Intermediate chunks only update hidden state and KV. The head
            // runs once, for the one row the scheduler will actually sample.
            let output_rows = if pos + chunk.len() == tokens.len() {
                vec![chunk.len() - 1]
            } else {
                Vec::new()
            };
            last = self.execute(&rows, &output_rows)?;
        }
        self.publish(slot);
        Ok(last)
    }
}

impl From<MetalError> for GenError {
    fn from(e: MetalError) -> Self {
        match e {
            MetalError::Memory(_) => GenError::OutOfMemory,
            e => GenError::Backend(e.to_string()),
        }
    }
}

impl Generator for Granite {
    fn tier_pump(&mut self) {
        if let Some(tier) = &mut self.cold {
            tier.pump(&mut self.pool, &mut self.radix);
        }
    }
    fn tier_prefix_loading(&mut self, _slot: usize, tokens: &[u32]) -> bool {
        let Some(tier) = &mut self.cold else {
            return false;
        };
        let planes = self
            .layers
            .iter()
            .flat_map(|l| {
                [
                    (&l.keys, BLOCK_TOKENS * (self.kv_width) * 2),
                    (&l.values, BLOCK_TOKENS * (self.kv_width) * 2),
                ]
            })
            .collect::<Vec<_>>();
        tier.loading(tokens, &planes, &mut self.pool, &mut self.radix)
    }
    fn tier_stats(&self) -> Option<paddock_engine::kv_tier::TierStats> {
        self.cold.as_ref().map(|t| t.stats())
    }
    fn tier_report(&self) -> Option<paddock_engine::kv_tier::TierReport> {
        self.cold.as_ref().map(|t| t.report())
    }
    fn tier_observe_prefill(&mut self, n: u32, us: f64) {
        if let Some(t) = &mut self.cold {
            t.observe(n, us);
        }
    }
    fn reset(&mut self) {
        self.pending.clear();
        self.encoding.clear();
        self.audio_encoding.clear();
        for slot in &mut self.slots {
            slot.table.clear(&mut self.pool);
            slot.history.clear();
            slot.reused = 0;
            slot.mm = None;
            slot.audio.clear();
            slot.radix_tokens.clear();
        }
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn vision_budget(&self) -> Option<paddock_engine::generator::VisionBudget> {
        self.vision.as_ref().map(vision::Vision::budget)
    }
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some() || self.audio.is_some()
    }
    fn supports_chunked_multimodal(&self) -> bool {
        self.vision.is_some() || self.audio.is_some()
    }
    fn forward_multimodal(
        &mut self,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<Option<(Vec<f32>, usize)>, GenError> {
        if self.audio.is_some() {
            Ok(Some(self.prefill_audio(0, chunks.to_vec())?))
        } else {
            Ok(Some(self.prefill_images(0, chunks)?))
        }
    }
    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        if self.audio.is_some() {
            self.prefill_audio(slot, chunks.to_vec())
        } else {
            Ok(self.prefill_images(slot, chunks)?)
        }
    }
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)>,
    ) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        if self.audio.is_some() {
            self.admit_audio(items)
        } else {
            self.admit_images(items)
        }
    }
    fn encode_step(&mut self) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        if self.audio.is_some() {
            self.encode_audio()
        } else {
            self.step_images()
        }
    }
    fn encoding_pending(&self) -> bool {
        !self.encoding.is_empty() || !self.audio_encoding.is_empty()
    }
    fn max_context(&self) -> usize {
        self.context
    }
    fn enable_batch(&mut self, max: usize) -> std::result::Result<usize, GenError> {
        Ok(max.min(self.slots.len()))
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.kv_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn forward(&mut self, token: u32) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.execute(&[(0, token, self.slots[0].history.len() as u32)], &[0])?)
    }
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(0, tokens)?)
    }
    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(slot, tokens)?)
    }
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        if tokens.len() != positions.len() {
            return Err(GenError::Backend("batch shape mismatch".into()));
        }
        // The scheduler uses position zero for holes; real decode rows follow
        // a nonempty prefill. Preserve its row indices without writing hole KV.
        let rows: Vec<_> = tokens
            .iter()
            .zip(positions)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(i, (&t, &p))| (i, t, p))
            .collect();
        let mut out = vec![0.0; tokens.len() * self.vocab];
        if !rows.is_empty() {
            let logits = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())?;
            for (i, row) in rows.iter().enumerate() {
                out[row.0 * self.vocab..(row.0 + 1) * self.vocab]
                    .copy_from_slice(&logits[i * self.vocab..(i + 1) * self.vocab]);
            }
        }
        Ok(out)
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        std::mem::take(&mut self.slots[slot].reused)
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks() + self.radix.evictable_blocks(&self.pool))
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
                && !self.image_slot_pending(i)
                && !self.audio_slot_pending(i)
            {
                if !self.slots[i].history.is_empty() {
                    self.publish(i);
                }
                self.slots[i].table.clear(&mut self.pool);
                self.slots[i].history.clear();
                self.slots[i].mm = None;
                self.slots[i].audio.clear();
                self.slots[i].radix_tokens.clear();
            }
        }
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    // the scheduler's tick pacer reads the FIFO queue from each offset
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.pending
            .iter()
            .map(|p| (p.slot, p.offset, p.tokens.len() - p.offset))
            .collect()
    }
    // the mixed grant: row_cap less the decode rows sharing it
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        crate::schedule::row_cap(decode_rows, CHUNK).saturating_sub(decode_rows)
    }
    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> std::result::Result<(), GenError> {
        if self.pending.iter().any(|p| p.slot == slot)
            || self.image_slot_pending(slot)
            || self.audio_slot_pending(slot)
        {
            return Err(GenError::Backend("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            work: tokens.len() - reused,
            tokens,
            offset: reused,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.abort_images(slot);
        self.abort_audio(slot);
        self.pending.retain(|p| p.slot != slot);
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.mm = None;
            s.audio.clear();
            s.radix_tokens.clear();
        }
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        // The measured M5 shape-cost curve elects 128 rows while a stream is
        // active (about 71 ms vs 235 ms at 512 in the cold-admission probe).
        // Cold cohorts retain the throughput-efficient 512-row grant. This is
        // not a universal latency bound: long KV and system load add cost.
        let cap = crate::schedule::row_cap(decodes.len(), CHUNK);
        let advances = crate::schedule::grants(
            &self
                .pending
                .iter()
                .map(|p| (p.tokens.len() - p.offset, p.work))
                .collect::<Vec<_>>(),
            budget.min(cap.saturating_sub(rows.len())),
            decodes.is_empty(),
        );
        for (pending, &n) in self.pending.iter().zip(&advances) {
            for i in pending.offset..pending.offset + n {
                rows.push((pending.slot, pending.tokens[i], i as u32));
            }
            if n > 0 && pending.offset + n == pending.tokens.len() {
                complete.push((pending.slot, rows.len() - 1, pending.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let output_rows: Vec<_> = (0..decodes.len())
            .chain(complete.iter().map(|&(_, row, _)| row))
            .collect();
        let logits = self.execute(&rows, &output_rows)?;
        for (pending, n) in self.pending.iter_mut().zip(advances) {
            pending.offset += n;
        }
        let done: Vec<_> = complete
            .iter()
            .enumerate()
            .map(|(i, &(slot, _, n))| {
                let row = decodes.len() + i;
                (
                    slot,
                    logits[row * self.vocab..(row + 1) * self.vocab].to_vec(),
                    n,
                )
            })
            .collect();
        for &(slot, _, _) in &complete {
            self.publish(slot);
        }
        self.pending.retain(|p| p.offset < p.tokens.len());
        Ok((logits[..decodes.len() * self.vocab].to_vec(), done))
    }
}
