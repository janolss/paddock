use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf, nemotron::NemotronConfig};
use std::{collections::HashSet, path::Path};

impl Nemotron {
    /// Exact Q8 artifact only, with all tensor shapes/bytes checked before
    /// uploading any weights. The in-file MTP planes are validated but stay
    /// unloaded; they are not a claim of speculative execution support.
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let hp =
            NemotronConfig::from_gguf(map.gguf()).map_err(|e| MetalError::Model(e.to_string()))?;
        if hp.hidden != WIDTH
            || hp.vocab != VOCAB
            || hp.n_layer != 52
            || hp.n_heads != 32
            || hp.n_kv_heads != 2
            || hp.head_dim != 128
            || hp.mamba_heads != 64
            || hp.mamba_head_dim != 64
            || hp.d_state != 128
            || hp.d_conv != 4
            || hp.n_groups != 8
            || hp.n_expert != 128
            || hp.n_active != 6
            || hp.moe_ff != FF
            || hp.shared_ff != SHARED
            || hp.eps != 1e-5
            || hp.routed_scale != 2.5
            || hp.blocks.len() != 52
            || hp
                .blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| **b == NemotronBlock::Attention)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
                != [5, 12, 19, 26, 33, 42]
            || hp
                .blocks
                .iter()
                .filter(|&&b| b == NemotronBlock::Mamba)
                .count()
                != 23
            || hp
                .blocks
                .iter()
                .filter(|&&b| b == NemotronBlock::Moe)
                .count()
                != 23
        {
            return Err(MetalError::Model(
                "Metal Nemotron requires elected Lightning 30B-A3B geometry".into(),
            ));
        }
        if context == 0 || context > hp.max_pos.min(32768) || max_batch == 0 || max_batch > 64 {
            return Err(MetalError::Model("Metal Nemotron context must be 1..=32768, batch 1..=64 (implementation ceilings, not qualification)".into()));
        }
        // Sigmoid is defined by nemotron_h_moe; the elected GGUF does not
        // stamp expert_gating_func. Grouped routing is a different graph.
        for (key, expected) in [
            ("expert_group_count", 1),
            ("expert_group_used_count", 1),
            ("expert_shared_count", 1),
            ("nextn_predict_layers", 1),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(expected) {
                return Err(MetalError::Model(format!("unsupported Nemotron {key}")));
            }
        }
        let mut expected = HashSet::new();
        let mut payload = 0u64;
        let mut check = |name: String, dims: &[usize], ty: u32, resident: bool| -> Result<()> {
            let (t, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if t.raw_type != ty
                || t.dims.iter().map(|&n| n as usize).collect::<Vec<_>>() != dims
                || t.ggml_type.byte_size(dims.iter().product::<usize>() as u64)
                    != Some(bytes.len() as u64)
            {
                return Err(MetalError::Model(format!(
                    "Nemotron {name}: unsupported shape/type/bytes {:?} {:?}",
                    t.dims, t.ggml_type
                )));
            }
            if resident {
                payload += bytes.len() as u64;
            }
            expected.insert(name);
            Ok(())
        };
        for (name, dims, ty) in [
            ("token_embd.weight", vec![WIDTH, VOCAB], 8),
            ("output.weight", vec![WIDTH, VOCAB], 8),
            ("output_norm.weight", vec![WIDTH], 0),
        ] {
            check(name.into(), &dims, ty, true)?;
        }
        for i in 0..53 {
            let resident = i < 52;
            let mut c = |name: &str, dims: &[usize], ty| {
                check(format!("blk.{i}.{name}"), dims, ty, resident)
            };
            c("attn_norm.weight", &[WIDTH], 0)?;
            if i == 52 || hp.blocks[i] == NemotronBlock::Attention {
                for (name, dims) in [
                    ("attn_q.weight", [WIDTH, 4096]),
                    ("attn_k.weight", [WIDTH, 256]),
                    ("attn_v.weight", [WIDTH, 256]),
                    ("attn_output.weight", [4096, WIDTH]),
                ] {
                    c(name, &dims, 8)?;
                }
            }
            if i == 52 || hp.blocks[i] == NemotronBlock::Moe {
                c(
                    "ffn_gate_inp.weight",
                    &[WIDTH, 128],
                    if i == 52 { 30 } else { 0 },
                )?;
                c("exp_probs_b.bias", &[128], 0)?;
                c("ffn_up_exps.weight", &[WIDTH, FF, 128], 8)?;
                c("ffn_down_exps.weight", &[FF, WIDTH, 128], 8)?;
                c("ffn_up_shexp.weight", &[WIDTH, SHARED], 8)?;
                c("ffn_down_shexp.weight", &[SHARED, WIDTH], 8)?;
            }
            if i < 52 && hp.blocks[i] == NemotronBlock::Mamba {
                c("ssm_in.weight", &[WIDTH, PROJECTED], 8)?;
                c("ssm_out.weight", &[INNER, WIDTH], 8)?;
                c("ssm_conv1d.weight", &[4, CONV], 0)?;
                c("ssm_conv1d.bias", &[CONV], 0)?;
                for name in ["ssm_a", "ssm_d"] {
                    c(name, &[1, 64], 0)?;
                }
                c("ssm_dt.bias", &[64], 0)?;
                c("ssm_norm.weight", &[512, 8], 0)?;
            }
            if i == 52 {
                c("nextn.eh_proj.weight", &[WIDTH * 2, WIDTH], 8)?;
                for name in [
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "nextn.shared_head_norm.weight",
                    "post_attention_norm.weight",
                ] {
                    c(name, &[WIDTH], 0)?;
                }
            }
        }
        if let Some(t) = map.tensor_infos().find(|t| !expected.contains(&t.name)) {
            return Err(MetalError::Model(format!(
                "unexpected Nemotron tensor {}",
                t.name
            )));
        }
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let prefix_count = PREFIX_PER_SLOT * max_batch;
        let state_slots = max_batch + prefix_count;
        let blocks = page_stride * state_slots;
        let kv_layer = blocks * BLOCK_TOKENS * 256 * 2;
        let cache_bytes = (12 * kv_layer + 23 * state_slots * (STATE + WINDOW) * 4) as u64;
        let sizes = Scratch::sizes(max_batch, page_stride);
        let device = MetalDevice::new(budget)?;
        let required = payload + cache_bytes + sizes.iter().sum::<usize>() as u64;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Nemotron weights/KV/recurrent/scratch need {required} bytes, grant {}",
                device.budget_bytes()
            )));
        }
        let w = |name: &str, dims: &[usize]| Weight::load(&device, &map, name, dims);
        let embedding = w("token_embd.weight", &[WIDTH, VOCAB])?;
        let head = w("output.weight", &[WIDTH, VOCAB])?;
        let output_norm = w("output_norm.weight", &[WIDTH])?;
        let mut layers = Vec::new();
        for (i, kind) in hp.blocks.iter().enumerate() {
            let w = |name: &str, dims: &[usize]| {
                Weight::load(&device, &map, &format!("blk.{i}.{name}"), dims)
            };
            let norm = w("attn_norm.weight", &[WIDTH])?;
            let mixer = match kind {
                NemotronBlock::Mamba => Mixer::Mamba(Mamba {
                    input: w("ssm_in.weight", &[WIDTH, PROJECTED])?,
                    output: w("ssm_out.weight", &[INNER, WIDTH])?,
                    conv_w: w("ssm_conv1d.weight", &[4, CONV])?,
                    conv_b: w("ssm_conv1d.bias", &[CONV])?,
                    a: w("ssm_a", &[1, 64])?,
                    d: w("ssm_d", &[1, 64])?,
                    dt: w("ssm_dt.bias", &[64])?,
                    norm: w("ssm_norm.weight", &[512, 8])?,
                    state: device.alloc(state_slots * STATE * 4)?,
                    window: device.alloc(state_slots * WINDOW * 4)?,
                }),
                NemotronBlock::Attention => Mixer::Attention(Attention {
                    q: w("attn_q.weight", &[WIDTH, 4096])?,
                    k: w("attn_k.weight", &[WIDTH, 256])?,
                    v: w("attn_v.weight", &[WIDTH, 256])?,
                    out: w("attn_output.weight", &[4096, WIDTH])?,
                    keys: device.alloc(kv_layer)?,
                    values: device.alloc(kv_layer)?,
                }),
                NemotronBlock::Moe => Mixer::Moe(Experts {
                    router: w("ffn_gate_inp.weight", &[WIDTH, 128])?,
                    bias: w("exp_probs_b.bias", &[128])?,
                    up: w("ffn_up_exps.weight", &[WIDTH, FF, 128])?,
                    down: w("ffn_down_exps.weight", &[FF, WIDTH, 128])?,
                    shared_up: w("ffn_up_shexp.weight", &[WIDTH, SHARED])?,
                    shared_down: w("ffn_down_shexp.weight", &[SHARED, WIDTH])?,
                }),
            };
            layers.push(Layer { norm, mixer });
        }
        assert_eq!(device.allocated_bytes(), payload + cache_bytes);
        let scratch = Scratch::new(&device, max_batch, page_stride)?;
        assert_eq!(device.allocated_bytes(), required);
        tracing::info!(
            weight_bytes = payload,
            cache_bytes,
            allocated = required,
            context,
            max_batch,
            "native Metal Nemotron Q8, F32 SSD state/F16 paged KV; no speculative heads"
        );
        Ok(Self {
            device,
            embedding,
            head,
            output_norm,
            layers,
            scratch,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            prefixes: (0..prefix_count)
                .map(|_| prefix::Checkpoint::default())
                .collect(),
            prefix_clock: 0,
            pool: KvPool::with_blocks(blocks as u32),
            context,
            page_stride,
            weight_bytes: payload,
            cache_bytes,
            last_gpu_seconds: 0.,
            #[cfg(test)]
            scan_only: false,
            #[cfg(test)]
            single_snapshot: false,
        })
    }
}
impl Scratch {
    pub(super) fn sizes(batch: usize, pages: usize) -> [usize; 35] {
        let tiles = CHUNK.div_ceil(32) + batch;
        [
            CHUNK,
            CHUNK * 2,
            batch * pages,
            CHUNK,
            CHUNK,
            CHUNK * 2,
            batch * 4,
            tiles * 4,
            CHUNK * WIDTH,
            CHUNK * WIDTH,
            CHUNK * WIDTH,
            CHUNK * PROJECTED,
            CHUNK * CONV,
            CHUNK * INNER,
            CHUNK * INNER,
            CHUNK * 64,
            tiles * 64 * 32,
            tiles * 64 * 32 * 32,
            tiles * STATE,
            tiles * STATE,
            CHUNK * 4096,
            CHUNK * 256,
            CHUNK * 256,
            CHUNK * 4096,
            CHUNK * 32 * SPLITS * 130,
            CHUNK * 128,
            CHUNK * 6,
            CHUNK * 6,
            128 * CHUNK * 6,
            128,
            1 + 2 * ((CHUNK * 6).div_ceil(16) + 128),
            CHUNK * 6 * FF,
            CHUNK * 6 * WIDTH,
            CHUNK * SHARED,
            batch * VOCAB,
        ]
        .map(|n| n * 4)
    }
    fn new(d: &MetalDevice, batch: usize, pages: usize) -> Result<Self> {
        let mut sizes = Self::sizes(batch, pages).into_iter();
        let mut a = || d.alloc(sizes.next().expect("scratch layout"));
        let result = Self {
            ids: a()?,
            meta: a()?,
            pages: a()?,
            output_rows: a()?,
            decode_rows: a()?,
            attention_tiles: a()?,
            sequences: a()?,
            ssd_tiles: a()?,
            x: a()?,
            norm: a()?,
            delta: a()?,
            proj: a()?,
            conv: a()?,
            y: a()?,
            yn: a()?,
            dt: a()?,
            decay: a()?,
            matrix: a()?,
            state_delta: a()?,
            state_in: a()?,
            q: a()?,
            k: a()?,
            v: a()?,
            attention: a()?,
            parts: a()?,
            router: a()?,
            picks: a()?,
            probabilities: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            up: a()?,
            expert_out: a()?,
            shared: a()?,
            logits: a()?,
        };
        assert!(sizes.next().is_none());
        Ok(result)
    }
}
