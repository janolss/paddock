//! Native mixed affine4 experts / affine8 attention and shared MLP. The text
//! graph shares a single decoder body with its causal encoder, as upstream.
use super::*;
use paddock_models::mlx::DiffusionConfig;

impl Gemma4 {
    pub(super) fn load_diffusion_mlx(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let cfg = DiffusionConfig::read(path).map_err(MetalError::Model)?;
        cfg.validate_vision().map_err(MetalError::Model)?;
        if context == 0 || context > cfg.context.min(32768) || !(1..=8).contains(&max_batch) {
            return Err(MetalError::Model(
                "DiffusionGemma Metal requires context 1..32768 and 1..8 slots".into(),
            ));
        }
        let source = mlx::Source::open(path)?;
        let (width, ff, vocab, window, heads) = (2816, 2112, 262144, 1024, 16);
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let ring = context.min(window + CHUNK).next_multiple_of(BLOCK_TOKENS);
        let blocks = page_stride * (max_batch * 2 + 1);
        let global = blocks * BLOCK_TOKENS * 1024 * 2;
        let sliding = ring * max_batch * 2 * 2048 * 2;
        let kv_bytes = (global * 5 * 2 + sliding * 25 * 2) as u64;
        let scratch_bytes = (CHUNK * (width * 3 + 8192 * 2 + 2048 * 2 + ff * 2 + 9) * 4
            + max_batch * (page_stride + vocab + HEADS * SPLITS * 514) * 4)
            as u64
            + moe::Workspace::bytes(CHUNK)
            + diffusion::Lane::workspace_bytes(max_batch);
        let required = source
            .bytes()
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes)
            // Bounded four-image encoder wave, casts and cached features.
            .saturating_add(1 << 30)
            .saturating_add(64 << 20);
        diffusion_residency::admit(required, budget)?;
        let device = MetalDevice::new_planned(budget, required)?;
        let packed = |name: &str, k, n| {
            source.packed(
                &device,
                name,
                k,
                &[n],
                cfg.bits(name).map_err(MetalError::Model)?,
            )
        };
        let raw = |name: &str, n| source.raw(&device, name, &[n], true);
        let embedding = packed("model.decoder.embed_tokens", width, vocab)?;
        let output_norm = raw("model.decoder.norm.weight", width)?;
        let mut layers = Vec::with_capacity(30);
        for i in 0..30 {
            let slide = i % 6 != 5;
            let (hd, kh) = if slide { (256, 8) } else { (512, 2) };
            let p = format!("model.decoder.layers.{i}");
            let w = |name: &str, k, n| packed(&format!("{p}.{name}"), k, n);
            let norm = |name: &str, n| raw(&format!("{p}.{name}.weight"), n);
            let scale = raw(&format!("{p}.layer_scalar"), 1)?;
            let enc = raw(
                &format!("model.encoder.language_model.layers.{i}.layer_scalar"),
                1,
            )?;
            let scale = unsafe {
                let a = scale.buffer.read_f32(0, 1)[0];
                if enc.buffer.read_f32(0, 1)[0] != a {
                    return Err(MetalError::Model("encoder/decoder scales differ".into()));
                }
                a
            };
            let bytes = if slide { sliding } else { global };
            layers.push(Layer {
                heads,
                moe: Some(moe::Experts::load_mlx(&device, &source, &cfg, &p)?),
                norm: norm("input_layernorm", width)?,
                post_attn: norm("post_attention_layernorm", width)?,
                ffn_norm: norm("pre_feedforward_layernorm", width)?,
                post_ffn: norm("post_feedforward_layernorm", width)?,
                q: w("self_attn.q_proj", width, heads * hd)?,
                k: w("self_attn.k_proj", width, kh * hd)?,
                v: if slide {
                    Some(w("self_attn.v_proj", width, kh * hd)?)
                } else {
                    None
                },
                o: w("self_attn.o_proj", heads * hd, width)?,
                attn_gate: None,
                q_norm: norm("self_attn.q_norm", hd)?,
                k_norm: norm("self_attn.k_norm", hd)?,
                gate: w("mlp.gate_proj", width, ff)?,
                up: w("mlp.up_proj", width, ff)?,
                down: w("mlp.down_proj", ff, width)?,
                scale,
                sliding: slide,
                keys: device.alloc(bytes)?,
                values: device.alloc(bytes)?,
            });
        }
        let pre = raw("model.decoder.self_conditioning.pre_norm.weight", width)?;
        let gate = packed("model.decoder.self_conditioning.gate_proj", width, ff)?;
        let up = packed("model.decoder.self_conditioning.up_proj", width, ff)?;
        let down = packed("model.decoder.self_conditioning.down_proj", ff, width)?;
        let vision = vision::Vision::load_mlx_at(&device, &source, "model.encoder.", width)?;
        source.finish()?;
        let tokenizer: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("tokenizer.json"))
                .map_err(|e| MetalError::Model(e.to_string()))?,
        )
        .map_err(|e| MetalError::Model(e.to_string()))?;
        let marker = |text: &str| {
            tokenizer["added_tokens"]
                .as_array()
                .and_then(|a| a.iter().find(|v| v["content"] == text))
                .and_then(|v| v["id"].as_u64())
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| (n as usize) < vocab)
        };
        let image_markers = Some(marker("<|image>").zip(marker("<image|>")).ok_or_else(|| {
            MetalError::Model("DiffusionGemma tokenizer lacks image markers".into())
        })?);
        let weight_bytes = device.allocated_bytes() - kv_bytes;
        let lane = diffusion::Lane::new(&device, max_batch, pre, gate, up, down)?;
        let a = |n| device.alloc(CHUNK * n * 4);
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
            q: a(8192)?,
            k: a(2048)?,
            v: a(2048)?,
            attn: a(8192)?,
            attn_gate: device.alloc(4)?,
            parts: device.alloc(max_batch * HEADS * SPLITS * 514 * 4)?,
            gate: a(ff)?,
            up: a(ff)?,
            gemm: device.alloc(4)?,
            logits: device.alloc(max_batch * vocab * 4)?,
        };
        let factors = Weight {
            buffer: device.alloc(4)?,
            ty: 0,
            k: 1,
            n: 1,
        };
        let moe_scratch = Some(moe::Workspace::new(&device, CHUNK)?);
        Ok(Self {
            diffusion: Some(lane),
            muse: false,
            mlx: true,
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
            vision: Some(tower::Tower::Gemma(Box::new(vision))),
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
            eps: 1e-6,
            rope: [10000., 1000000.],
            softcap: 30.,
            logit_scale: 1.,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
