//! Direct affine-8/group-64 ingestion. Packed checkpoint bytes stay packed;
//! only the small normalization/recurrent constants are cast to F32 on GPU.
use super::*;
use paddock_models::{
    nemotron::NemotronConfig,
    safetensors::{ShardedSafetensors, StDtype},
};
use std::{collections::HashSet, path::Path};

pub(super) const AFFINE8: u32 = 0x108;

struct Plane {
    name: String,
    shape: Vec<usize>,
    dtype: StDtype,
}
struct Plan {
    source: ShardedSafetensors,
    config: NemotronConfig,
    bytes: u64,
}
fn error(s: impl ToString) -> MetalError {
    MetalError::Model(s.to_string())
}

impl Plan {
    fn open(path: &Path) -> Result<Self> {
        let config = NemotronConfig::read(path).map_err(error)?;
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).map_err(error)?)
                .map_err(error)?;
        for (key, expected) in [
            ("hidden_size", WIDTH),
            ("vocab_size", VOCAB),
            ("num_hidden_layers", 52),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 2),
            ("head_dim", 128),
            ("mamba_num_heads", 64),
            ("mamba_head_dim", 64),
            ("ssm_state_size", 128),
            ("conv_kernel", 4),
            ("n_groups", 8),
            ("n_routed_experts", 128),
            ("num_experts_per_tok", 6),
            ("moe_intermediate_size", FF),
            ("moe_shared_expert_intermediate_size", SHARED),
            ("n_group", 1),
            ("topk_group", 1),
            ("n_shared_experts", 1),
        ] {
            if v[key].as_u64() != Some(expected as u64) {
                return Err(error(format!("Nemotron MLX: unsupported {key}")));
            }
        }
        if config.eps != 1e-5
            || config.routed_scale != 2.5
            || v["quantization"] != serde_json::json!({"bits":8,"group_size":64,"mode":"affine"})
            || v["quantization_config"] != v["quantization"]
            || v["mlp_hidden_act"] != "relu2"
            || v["mamba_hidden_act"] != "silu"
            || !v["moe_latent_size"].is_null()
            || v["norm_topk_prob"] != true
            || v["use_conv_bias"] != true
            || v["tie_word_embeddings"] != false
            || ["attention_bias", "mlp_bias", "mamba_proj_bias"]
                .iter()
                .any(|k| v[k] != false)
            || v.get("time_step_limit").is_some_and(|t| !t.is_null())
        {
            return Err(error(
                "Nemotron MLX requires the affine-8/group-64 Lightning graph",
            ));
        }
        let source = ShardedSafetensors::open_dir(path).map_err(error)?;
        let mut planes = Vec::new();
        let raw = |p: &mut Vec<Plane>, name: String, shape: Vec<usize>, dtype| {
            p.push(Plane { name, shape, dtype })
        };
        let packed = |p: &mut Vec<Plane>, name: &str, k: usize, dims: &[usize]| {
            for (suffix, dtype, last) in [
                ("weight", StDtype::U32, k / 4),
                ("scales", StDtype::Bf16, k / 64),
                ("biases", StDtype::Bf16, k / 64),
            ] {
                let mut shape = dims.to_vec();
                shape.push(last);
                p.push(Plane {
                    name: format!("{name}.{suffix}"),
                    shape,
                    dtype,
                });
            }
        };
        packed(&mut planes, "backbone.embeddings", WIDTH, &[VOCAB]);
        packed(&mut planes, "lm_head", WIDTH, &[VOCAB]);
        raw(
            &mut planes,
            "backbone.norm_f.weight".into(),
            vec![WIDTH],
            StDtype::Bf16,
        );
        for (i, kind) in config.blocks.iter().enumerate() {
            let b = format!("backbone.layers.{i}");
            raw(
                &mut planes,
                format!("{b}.norm.weight"),
                vec![WIDTH],
                StDtype::Bf16,
            );
            let b = format!("{b}.mixer");
            match kind {
                NemotronBlock::Attention => {
                    for (n, k, o) in [
                        ("q_proj", WIDTH, 4096),
                        ("k_proj", WIDTH, 256),
                        ("v_proj", WIDTH, 256),
                        ("o_proj", 4096, WIDTH),
                    ] {
                        packed(&mut planes, &format!("{b}.{n}"), k, &[o]);
                    }
                }
                NemotronBlock::Mamba => {
                    packed(&mut planes, &format!("{b}.in_proj"), WIDTH, &[PROJECTED]);
                    packed(&mut planes, &format!("{b}.out_proj"), INNER, &[WIDTH]);
                    for (name, shape) in [
                        ("A_log", vec![64]),
                        ("D", vec![64]),
                        ("dt_bias", vec![64]),
                        ("conv1d.weight", vec![CONV, 4, 1]),
                        ("conv1d.bias", vec![CONV]),
                        ("norm.weight", vec![INNER]),
                    ] {
                        raw(&mut planes, format!("{b}.{name}"), shape, StDtype::Bf16);
                    }
                }
                NemotronBlock::Moe => {
                    raw(
                        &mut planes,
                        format!("{b}.gate.weight"),
                        vec![128, WIDTH],
                        StDtype::Bf16,
                    );
                    raw(
                        &mut planes,
                        format!("{b}.gate.e_score_correction_bias"),
                        vec![128],
                        StDtype::F32,
                    );
                    packed(
                        &mut planes,
                        &format!("{b}.switch_mlp.fc1"),
                        WIDTH,
                        &[128, FF],
                    );
                    packed(
                        &mut planes,
                        &format!("{b}.switch_mlp.fc2"),
                        FF,
                        &[128, WIDTH],
                    );
                    packed(
                        &mut planes,
                        &format!("{b}.shared_experts.up_proj"),
                        WIDTH,
                        &[SHARED],
                    );
                    packed(
                        &mut planes,
                        &format!("{b}.shared_experts.down_proj"),
                        SHARED,
                        &[WIDTH],
                    );
                }
            }
        }
        let mut expected = HashSet::new();
        let mut bytes = 0;
        // Complete validation before device creation/allocation. No missing or
        // extra tensors, silent bias drops, latent adapters or hidden drafter.
        for plane in planes {
            let (info, data) = source
                .bytes(&plane.name)
                .ok_or_else(|| error(format!("missing {}", plane.name)))?;
            if info.shape != plane.shape || info.dtype != plane.dtype {
                return Err(error(format!("{}: invalid shape/dtype", plane.name)));
            }
            if info.dtype != StDtype::U32 {
                let finite = if info.dtype == StDtype::Bf16 {
                    data.chunks_exact(2)
                        .all(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7f80) != 0x7f80)
                } else {
                    data.chunks_exact(4).all(|b| {
                        f32::from_le_bytes(b.try_into().expect("four-byte chunk")).is_finite()
                    })
                };
                if !finite {
                    return Err(error(format!("{}: nonfinite parameters", plane.name)));
                }
            }
            let packed_side = plane.name.ends_with(".scales") || plane.name.ends_with(".biases");
            bytes += if info.dtype == StDtype::Bf16 && !packed_side {
                data.len() as u64 * 2
            } else {
                data.len() as u64
            };
            expected.insert(plane.name);
        }
        if let Some(name) = source.names().find(|n| !expected.contains(*n)) {
            return Err(error(format!("unexpected Nemotron MLX tensor {name}")));
        }
        Ok(Self {
            source,
            config,
            bytes,
        })
    }
    fn packed(&self, d: &MetalDevice, name: &str, k: usize, n: usize) -> Result<Weight> {
        let parts = ["weight", "scales", "biases"].map(|s| {
            self.source
                .bytes(&format!("{name}.{s}"))
                .expect("validated plane")
                .1
        });
        Ok(Weight {
            buffer: d.upload_parts(&parts)?,
            ty: AFFINE8,
            k,
            n,
        })
    }
    fn raw(&self, d: &MetalDevice, name: &str) -> Result<Weight> {
        let (info, data) = self.source.bytes(name).expect("validated plane");
        let n = info.shape.iter().product::<usize>();
        let buffer = if info.dtype == StDtype::F32 {
            d.upload(data)?
        } else {
            let input = d.upload(data)?;
            let out = d.alloc(n * 4)?;
            let bad = d.upload(&0u32.to_le_bytes())?;
            let cmd = d.begin()?;
            cmd.dispatch(
                "vis_cast",
                &[&input, &out, &bad],
                &[n as u32, 30, 0],
                [n.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            out
        };
        if name.ends_with(".A_log") {
            let cmd = d.begin()?;
            cmd.dispatch(
                "nemo_mlx_a",
                &[&buffer],
                &[n as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
        }
        Ok(Weight {
            buffer,
            ty: 0,
            k: *info.shape.last().expect("validated nonempty shape"),
            n,
        })
    }
}

impl Nemotron {
    pub(super) fn load_mlx(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        if context == 0 || context > 32768 || !(1..=64).contains(&max_batch) {
            return Err(error("Nemotron MLX context must be 1..32768, batch 1..64"));
        }
        let plan = Plan::open(path)?;
        // Inspect the real tokenizer, not hard-coded newline IDs. A token may
        // combine punctuation and whitespace (e.g. ":\n"), so checking only
        // whitespace-only tokens would miss the observed template rewind.
        let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(path).map_err(error)?;
        let reply_trim_tokens = (0..tokenizer.vocab_size())
            .map(|id| {
                tokenizer
                    .decode(&[id as u32], false)
                    .map(|text| text.chars().last().is_some_and(char::is_whitespace))
                    .map_err(error)
            })
            .collect::<Result<Vec<_>>>()?;
        drop(tokenizer);
        if context == 0
            || context > plan.config.max_pos.min(32768)
            || !(1..=64).contains(&max_batch)
        {
            return Err(error("Nemotron MLX context must be 1..32768, batch 1..64"));
        }
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let prefix_count = MLX_PREFIX_PER_SLOT * max_batch;
        let state_slots = max_batch + prefix_count;
        let blocks = page_stride * state_slots;
        let kv_layer = blocks * BLOCK_TOKENS * 256 * 2;
        let mambas = plan
            .config
            .blocks
            .iter()
            .filter(|&&b| b == NemotronBlock::Mamba)
            .count();
        let attention = plan
            .config
            .blocks
            .iter()
            .filter(|&&b| b == NemotronBlock::Attention)
            .count();
        let cache_bytes =
            (2 * attention * kv_layer + mambas * state_slots * (STATE + WINDOW) * 4) as u64;
        let required = plan.bytes
            + cache_bytes
            + Scratch::sizes_for_rows(max_batch, page_stride, MLX_CHUNK, true)
                .iter()
                .sum::<usize>() as u64;
        // Room for the largest transient BF16->F32 raw-plane cast, not a
        // full dequantized matrix or second model-sized copy.
        let device = MetalDevice::new_planned(budget, required + (WIDTH * 128 * 2 + 4) as u64)?;
        let embedding = plan.packed(&device, "backbone.embeddings", WIDTH, VOCAB)?;
        let head = plan.packed(&device, "lm_head", WIDTH, VOCAB)?;
        let output_norm = plan.raw(&device, "backbone.norm_f.weight")?;
        let mut layers = Vec::new();
        for (i, kind) in plan.config.blocks.iter().enumerate() {
            let b = format!("backbone.layers.{i}");
            let norm = plan.raw(&device, &format!("{b}.norm.weight"))?;
            let b = format!("{b}.mixer");
            let raw = |n: &str| plan.raw(&device, &format!("{b}.{n}"));
            let w = |n: &str, k, o| plan.packed(&device, &format!("{b}.{n}"), k, o);
            let mixer = match kind {
                NemotronBlock::Mamba => Mixer::Mamba(Mamba {
                    input: w("in_proj", WIDTH, PROJECTED)?,
                    output: w("out_proj", INNER, WIDTH)?,
                    conv_w: raw("conv1d.weight")?,
                    conv_b: raw("conv1d.bias")?,
                    a: raw("A_log")?,
                    d: raw("D")?,
                    dt: raw("dt_bias")?,
                    norm: raw("norm.weight")?,
                    state: device.alloc(state_slots * STATE * 4)?,
                    window: device.alloc(state_slots * WINDOW * 4)?,
                }),
                NemotronBlock::Attention => Mixer::Attention(Attention {
                    q: w("q_proj", WIDTH, 4096)?,
                    k: w("k_proj", WIDTH, 256)?,
                    v: w("v_proj", WIDTH, 256)?,
                    out: w("o_proj", 4096, WIDTH)?,
                    keys: device.alloc(kv_layer)?,
                    values: device.alloc(kv_layer)?,
                }),
                NemotronBlock::Moe => Mixer::Moe(Experts {
                    router: raw("gate.weight")?,
                    bias: raw("gate.e_score_correction_bias")?,
                    up: w("switch_mlp.fc1", WIDTH, 128 * FF)?,
                    down: w("switch_mlp.fc2", FF, 128 * WIDTH)?,
                    shared_up: w("shared_experts.up_proj", WIDTH, SHARED)?,
                    shared_down: w("shared_experts.down_proj", SHARED, WIDTH)?,
                }),
            };
            layers.push(Layer { norm, mixer });
        }
        assert_eq!(device.allocated_bytes(), plan.bytes + cache_bytes);
        let scratch = Scratch::new_for_rows(&device, max_batch, page_stride, MLX_CHUNK, true)?;
        assert_eq!(device.allocated_bytes(), required);
        tracing::info!(
            weight_bytes = plan.bytes,
            cache_bytes,
            allocated = required,
            "native Nemotron MLX affine-8, BF16 KV, F32 recurrent state; speculation off"
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
            reply_trim_tokens,
            pool: KvPool::with_blocks(blocks as u32),
            context,
            page_stride,
            weight_bytes: plan.bytes,
            cache_bytes,
            last_gpu_seconds: 0.,
            #[cfg(test)]
            scan_only: false,
            #[cfg(test)]
            single_snapshot: false,
            #[cfg(test)]
            reference_inputs: vec![],
        })
    }
}
