use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};

impl Gemma4 {
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        if path.is_dir() {
            let config = std::fs::read(path.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
            if config
                .as_ref()
                .is_some_and(|c| c["model_type"] == "diffusion_gemma")
            {
                return Self::load_diffusion_mlx(path, context, max_batch, budget);
            }
            return Self::load_mlx(path, context, max_batch, budget);
        }
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        if map.gguf().architecture() == Some("muse-glimmer") {
            return Self::load_muse(&map, context, max_batch, budget);
        }
        let bad = |s: &str| MetalError::Model(format!("Gemma 4 Metal: {s}"));
        let diffusion = map.gguf().architecture() == Some("diffusion-gemma");
        if map.gguf().architecture() != Some("gemma4") && !diffusion {
            return Err(bad("expected gemma4 GGUF"));
        }
        let u = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad(key))
        };
        let f = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|n| n.is_finite() && *n > 0.)
                .ok_or_else(|| bad(key))
        };
        let width = u("embedding_length")?;
        let ff = u("feed_forward_length")?;
        let count = u("block_count")?;
        let trained = u("context_length")?;
        let window = u("attention.sliding_window")?;
        let heads = u("attention.head_count")?;
        let experts = map
            .gguf()
            .arch_field("expert_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let moe = (width, ff, count, heads, experts) == (2816, 2112, 30, 16, 128);
        if diffusion && !moe {
            return Err(bad("DiffusionGemma requires the 26B-A4B graph"));
        }
        if diffusion && !(1..=8).contains(&max_batch) {
            return Err(bad("DiffusionGemma requires 1..8 concurrent slots"));
        }
        // Exact graph allowlist, not a family alias. Per-layer embeddings and
        // shared target KV remain unsupported on both elected geometries.
        if (!moe && (width, ff, count, heads, experts) != (5376, 21504, 60, HEADS, 0))
            || window != 1024
            || u("attention.key_length")? != 512
            || u("attention.value_length")? != 512
            || u("attention.key_length_swa")? != 256
            || u("attention.value_length_swa")? != 256
            || u("rope.dimension_count")? != 512
            || u("rope.dimension_count_swa")? != 256
            || u("attention.shared_kv_layers")? != 0
            || u("embedding_length_per_layer_input")? != 0
        {
            return Err(bad(
                "requires dense 31B or Q8 26B-A4B geometry; small/shared-KV variants are not implemented",
            ));
        }
        let pattern = match map.gguf().arch_field("attention.sliding_window_pattern") {
            Some(Value::Array(v)) if v.len() == count => v
                .iter()
                .enumerate()
                .all(|(i, v)| matches!(v, Value::Bool(b) if *b == (i % 6 != 5))),
            _ => false,
        };
        let kv_heads = match map.gguf().arch_field("attention.head_count_kv") {
            Some(Value::Array(v)) if v.len() == count => v.iter().enumerate().all(|(i, v)| {
                v.as_u64()
                    == Some(if i % 6 != 5 {
                        heads as u64 / 2
                    } else {
                        heads as u64 / 8
                    })
            }),
            _ => false,
        };
        if !pattern || !kv_heads {
            return Err(bad("invalid per-layer attention geometry"));
        }
        if context == 0 || context > trained || max_batch == 0 || max_batch >= CHUNK {
            return Err(bad(
                "context exceeds training limit, or batch outside 1..512",
            ));
        }
        if moe {
            if context > 32768 || max_batch > 64 {
                return Err(bad(
                    "26B-A4B native ceilings are context 32768 and batch 64; larger capacities are not implemented",
                ));
            }
            moe::validate(&map, count)?;
        }
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| t.dims.get(1))
            .copied()
            .ok_or_else(|| bad("missing embedding table"))? as usize;
        if vocab != 262144 || map.tensor_info("output.weight").is_some() {
            return Err(bad("expected tied 262144-token head"));
        }
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let image_markers = match map.gguf().metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(v)) => v
                .iter()
                .position(|v| v.as_str() == Some("<|image>"))
                .zip(v.iter().position(|v| v.as_str() == Some("<image|>")))
                .map(|(a, b)| (a as u32, b as u32)),
            _ => None,
        };
        let rope = [f("rope.freq_base_swa")?, f("rope.freq_base")?];
        let softcap = f("final_logit_softcapping")?;
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let ring = context.min(window + CHUNK).next_multiple_of(BLOCK_TOKENS);
        let blocks = page_stride
            .checked_mul(max_batch * 2 + 1)
            .filter(|&n| n <= u32::MAX as usize / BLOCK_TOKENS)
            .ok_or_else(|| bad("KV size overflow"))?;
        let global_bytes = blocks * BLOCK_TOKENS * (heads / 8 * 512) * 2;
        // The second ring bank holds bounded prefix snapshots. Global pages
        // are retained by refcount, never cloned per prefix hit.
        let sliding_bytes = ring * max_batch * 2 * (heads / 2 * 256) * 2;
        let kv_bytes =
            (global_bytes * (count / 6) * 2 + sliding_bytes * (count / 6 * 5) * 2) as u64;
        // The shared-KV assistant has a wider FFN than the MoE target. Reserve
        // that bounded scratch up front, even when speculation is off.
        let scratch_ff = if moe { ff.max(8192) } else { ff };
        let gemm_bytes = (CHUNK * scratch_ff).max((CHUNK + 32) * HEADS * 512) * 2;
        // Two additional bounded residual/normalized planes can suspend an
        // image prefill between layers while decodes reuse the main scratch.
        let scratch_bytes =
            (CHUNK * (width * 5 + HEADS * 512 * 2 + 4096 * 2 + scratch_ff * 2 + 9) * 4
                + max_batch * (page_stride + vocab + HEADS * SPLITS * 514) * 4
                + gemm_bytes) as u64
                + if moe { moe::Workspace::bytes(CHUNK) } else { 0 }
                + if diffusion {
                    diffusion::Lane::workspace_bytes(max_batch)
                } else {
                    0
                };
        let device = if diffusion {
            let required = map
                .total_len()
                .saturating_add(kv_bytes)
                .saturating_add(scratch_bytes)
                .saturating_add(64 << 20);
            diffusion_residency::admit(required, budget)?;
            MetalDevice::new_planned(budget, required)?
        } else {
            MetalDevice::new(budget)?
        };
        if map
            .total_len()
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes)
            > device.budget_bytes()
        {
            return Err(MetalError::Memory(format!(
                "Gemma weights + bounded KV + scratch need {:.2} GiB; grant {:.2} GiB",
                (map.total_len() + kv_bytes + scratch_bytes) as f64 / (1u64 << 30) as f64,
                device.budget_bytes() as f64 / (1u64 << 30) as f64
            )));
        }
        let embedding = Weight::load(&device, &map, "token_embd.weight", &[width, vocab])?;
        let output_norm = Weight::load(&device, &map, "output_norm.weight", &[width])?;
        let factors = Weight::load(&device, &map, "rope_freqs.weight", &[256])?;
        if factors.ty != 0 {
            return Err(bad("rotary factors must be F32"));
        }
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let sliding = i % 6 != 5;
            let (hd, kh) = if sliding {
                (256, heads / 2)
            } else {
                (512, heads / 8)
            };
            let w = |name: &str, dims: &[usize]| {
                Weight::load(&device, &map, &format!("blk.{i}.{name}.weight"), dims)
            };
            let scale = w("layer_output_scale", &[1])?;
            if diffusion {
                let enc = w("enc_layer_output_scale", &[1])?;
                if enc.ty != 0
                    || unsafe { enc.buffer.read_f32(0, 1) != scale.buffer.read_f32(0, 1) }
                {
                    return Err(bad("encoder and decoder layer scales differ"));
                }
            }
            if scale.ty != 0 {
                return Err(bad("layer scale must be F32"));
            }
            let scale = unsafe { scale.buffer.read_f32(0, 1)[0] };
            if !scale.is_finite() {
                return Err(bad("non-finite layer scale"));
            }
            let vname = format!("blk.{i}.attn_v.weight");
            if sliding != map.tensor_info(&vname).is_some() {
                return Err(bad("unexpected K/V projection sharing"));
            }
            let bytes = if sliding { sliding_bytes } else { global_bytes };
            layers.push(Layer {
                heads,
                moe: if moe {
                    Some(moe::Experts::load(&device, &map, i)?)
                } else {
                    None
                },
                norm: w("attn_norm", &[width])?,
                post_attn: w("post_attention_norm", &[width])?,
                ffn_norm: w("ffn_norm", &[width])?,
                post_ffn: w("post_ffw_norm", &[width])?,
                q: w("attn_q", &[width, heads * hd])?,
                k: w("attn_k", &[width, kh * hd])?,
                v: if sliding {
                    Some(w("attn_v", &[width, kh * hd])?)
                } else {
                    None
                },
                o: w("attn_output", &[heads * hd, width])?,
                attn_gate: None,
                q_norm: w("attn_q_norm", &[hd])?,
                k_norm: w("attn_k_norm", &[hd])?,
                gate: w("ffn_gate", &[width, ff])?,
                up: w("ffn_up", &[width, ff])?,
                down: w("ffn_down", &[ff, width])?,
                scale,
                sliding,
                keys: device.alloc(bytes)?,
                values: device.alloc(bytes)?,
            });
        }
        let mut weight_bytes = device.allocated_bytes() - kv_bytes;
        let a = |n: usize| device.alloc(CHUNK * n * 4);
        let scratch = Scratch {
            rows: CHUNK,
            ids: a(1)?,
            meta: a(2)?,
            limits: a(1)?,
            pages: device.alloc(max_batch * page_stride * 4)?,
            outputs: a(1)?,
            tiles: a(2)?,
            decode_rows: a(1)?,
            x: a(width)?,
            norm: a(width)?,
            delta: a(width)?,
            q: a(HEADS * 512)?,
            k: a(4096)?,
            v: a(4096)?,
            attn: a(HEADS * 512)?,
            attn_gate: device.alloc(4)?,
            parts: device.alloc(max_batch * HEADS * SPLITS * 514 * 4)?,
            gate: a(scratch_ff)?,
            up: a(scratch_ff)?,
            gemm: device.alloc(gemm_bytes)?,
            logits: device.alloc(max_batch * vocab * 4)?,
        };
        let moe_scratch = if moe {
            Some(moe::Workspace::new(&device, CHUNK)?)
        } else {
            None
        };
        let diffusion = if diffusion {
            let lane = diffusion::Lane::load_gguf(&device, &map, max_batch)?;
            weight_bytes += lane.weight_bytes();
            Some(lane)
        } else {
            None
        };
        Ok(Self {
            diffusion,
            muse: false,
            mlx: false,
            device,
            embedding,
            output: None,
            output_norm,
            factors,
            layers,
            scratch,
            moe_scratch,
            mtp: None,
            dflash: None,
            vision: None,
            image_markers,
            image_cache: Vec::new(),
            image_cache_reused: 0,
            encoding: VecDeque::new(),
            spec: None,
            verifying: false,
            greedy_verify: false,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            cache: (0..max_batch).map(|_| Checkpoint::default()).collect(),
            clock: 0,
            pending: VecDeque::new(),
            prefill_phase: None,
            pool: KvPool::with_blocks(blocks as u32),
            width,
            ff,
            vocab,
            context,
            window,
            ring,
            page_stride,
            eps,
            rope,
            softcap,
            logit_scale: 1.,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
