use super::*;
impl Whisper {
    /// Peak GPU envelope for this fixed-shape backend, including alignment.
    /// Metadata-only: also used before an on-demand runner opens its listener.
    pub fn residency_bytes(map: &MappedGguf, context: usize, slots: usize) -> Result<u64> {
        if !(1..=MAX_BATCH).contains(&slots) || context < 8 {
            return Err(error(
                "Whisper requires 1..16 slots and at least 8 context tokens",
            ));
        }
        let mut source = 0u64;
        let mut widened = 0u64;
        let mut peak = 0u64;
        for tensor in &map.gguf().tensors {
            let (_, bytes) = map
                .tensor_bytes(&tensor.name)
                .map_err(|e| error(e.to_string()))?;
            source += bytes.len() as u64;
            if tensor.dims.len() == 1 || tensor.name.ends_with("embed_positions.weight") {
                widened += bytes.len() as u64;
                peak = peak.max(bytes.len() as u64);
            }
        }
        let ctx = context.min(448);
        let kv = (L * slots * (T + ctx) * D * 4) as u64;
        let scratch = Scratch::sizes(slots).iter().sum::<usize>() as u64;
        let admission = (slots.min(ENC_BATCH) * (3000 * 128 * 4 + T.div_ceil(32) * 16)) as u64;
        let alignment = super::timing::reservation_bytes(ctx);
        Ok(source + widened + peak.max(kv + scratch + admission + alignment))
    }

    pub fn load(path: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        Self::load_for_batch(path, context, budget, 1)
    }

    pub fn load_for_batch(
        path: &Path,
        context: usize,
        budget: Option<u64>,
        slots: usize,
    ) -> Result<Self> {
        if !(1..=MAX_BATCH).contains(&slots) {
            return Err(error("decode slots must be 1..16"));
        }
        if context < 8 {
            return Err(error("decoder context must be at least 8 tokens"));
        }
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        if map.gguf().architecture() != Some("whisper") {
            return Err(error("requires architecture=whisper"));
        }
        for (key, n) in [
            ("d_model", D),
            ("encoder.layer_count", L),
            ("encoder.head_count", 20),
            ("encoder.ffn_length", FF),
            ("decoder.layer_count", L),
            ("decoder.head_count", 20),
            ("decoder.ffn_length", FF),
            ("vocab_size", V),
            ("max_source_positions", T),
            ("max_target_positions", 448),
            ("mel.bins", 128),
            ("mel.n_fft", 400),
            ("mel.hop_length", 160),
            ("mel.chunk_length_s", 30),
            ("mel.sampling_rate", 16000),
            ("token.sot", 50258),
            ("token.eot", 50257),
            ("token.no_timestamps", 50364),
            ("token.transcribe", 50360),
        ] {
            if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(n as u64) {
                return Err(error(format!("unsupported {key} (requires {n})")));
            }
        }
        for (key, n) in [
            ("token.nospeech", 50363),
            ("token.sot_prev", 50362),
            ("token.timestamp_begin", 50365),
            ("token.translate", 50359),
        ] {
            if let Some(v) = map.gguf().arch_field(key)
                && v.as_u64() != Some(n)
            {
                return Err(error(format!("invalid {key}")));
            }
        }
        let langs = parse_langs(
            map.gguf()
                .arch_field("lang_to_id_json")
                .and_then(Value::as_str)
                .ok_or_else(|| error("missing language map"))?,
        )?;
        let schema = schema(map.tensor_info("proj_out.weight").is_some());
        if map.gguf().tensors.len() != schema.len() {
            return Err(error("unexpected tensor inventory"));
        }
        for (name, dims) in &schema {
            let (t, b) = map.tensor_bytes(name).map_err(|e| error(e.to_string()))?;
            if t.raw_type != 1
                || t.dims.iter().map(|&v| v as usize).collect::<Vec<_>>() != *dims
                || b.len() != dims.iter().product::<usize>() * 2
            {
                return Err(error(format!("{name}: expected F16 {dims:?}")));
            }
            if b.chunks_exact(2)
                .any(|x| u16::from_le_bytes([x[0], x[1]]) & 0x7c00 == 0x7c00)
            {
                return Err(error(format!("{name}: nonfinite weight")));
            }
        }
        let ctx = context.min(448);
        // residency_bytes already includes the weight copy. The GGUF mapping is
        // file cache and is already counted as reclaimable by the host sample.
        let required = Self::residency_bytes(&map, ctx, slots)?;
        let host = paddock_engine::host_memory::sample()
            .ok_or_else(|| MetalError::Memory("cannot read physical memory availability".into()))?;
        let available = host.available.saturating_sub(1 << 30);
        paddock_engine::host_memory::check(required, available, budget)
            .map_err(MetalError::Memory)?;
        let device = MetalDevice::new_planned(budget, required)?;
        let conv1 = linear(&device, &map, "encoder.conv1", 3 * 128, D)?;
        let conv2 = linear(&device, &map, "encoder.conv2", 3 * D, D)?;
        let enc_pos = wide(&device, &map, "encoder.embed_positions.weight")?;
        let mut enc = Vec::new();
        let mut dec = Vec::new();
        for i in 0..L {
            let p = format!("encoder.layers.{i}");
            enc.push(EncoderLayer {
                attn: attention(&device, &map, &p)?,
                mlp: mlp(&device, &map, &p)?,
            });
            let p = format!("decoder.layers.{i}");
            dec.push(DecoderLayer {
                attn: attention(&device, &map, &p)?,
                cross_norm: norm(&device, &map, &format!("{p}.encoder_attn_layer_norm"))?,
                q: linear(&device, &map, &format!("{p}.encoder_attn.q_proj"), D, D)?,
                kv: concat(
                    &device,
                    &map,
                    &[
                        format!("{p}.encoder_attn.k_proj.weight"),
                        format!("{p}.encoder_attn.v_proj.weight"),
                    ],
                )?,
                vb: wide(&device, &map, &format!("{p}.encoder_attn.v_proj.bias"))?,
                out: linear(&device, &map, &format!("{p}.encoder_attn.out_proj"), D, D)?,
                mlp: mlp(&device, &map, &p)?,
            });
        }
        let enc_ln = norm(&device, &map, "encoder.layer_norm")?;
        let dec_ln = norm(&device, &map, "decoder.layer_norm")?;
        let embedding = raw(&device, &map, "decoder.embed_tokens.weight")?;
        // Retain the declared head if present; never assume arbitrary imported
        // GGUFs tied it merely because the three authors did.
        let head = if map.tensor_info("proj_out.weight").is_some() {
            Some(raw(&device, &map, "proj_out.weight")?)
        } else {
            None
        };
        let dec_pos = wide(&device, &map, "decoder.embed_positions.weight")?;
        let weights_bytes = device.allocated_bytes();
        tracing::info!(
            weights_bytes,
            ctx,
            "Whisper Metal: F16 weights/KV, F32 residuals; chunked word alignment"
        );
        Ok(Self {
            device,
            conv1,
            conv2,
            enc_pos,
            enc,
            enc_ln,
            embedding,
            head,
            dec_pos,
            dec,
            dec_ln,
            langs,
            ctx,
            capacity: 0,
            cache: Vec::new(),
            scratch: None,
            lengths: Vec::new(),
            last_rows: 0,
            weights_bytes,
        })
    }
}
fn schema(head: bool) -> Vec<(String, Vec<usize>)> {
    let mut s = vec![
        ("encoder.conv1.weight".into(), vec![3, 128, D]),
        ("encoder.conv1.bias".into(), vec![D]),
        ("encoder.conv2.weight".into(), vec![3, D, D]),
        ("encoder.conv2.bias".into(), vec![D]),
        ("encoder.embed_positions.weight".into(), vec![D, T]),
        ("decoder.embed_tokens.weight".into(), vec![D, V]),
        ("decoder.embed_positions.weight".into(), vec![D, 448]),
    ];
    if head {
        s.push(("proj_out.weight".into(), vec![D, V]));
    }
    for side in ["encoder", "decoder"] {
        for tail in ["weight", "bias"] {
            s.push((format!("{side}.layer_norm.{tail}"), vec![D]));
        }
        for i in 0..L {
            let p = format!("{side}.layers.{i}");
            let attns = if side == "decoder" {
                vec!["self_attn", "encoder_attn"]
            } else {
                vec!["self_attn"]
            };
            for a in attns {
                for tail in ["weight", "bias"] {
                    s.push((format!("{p}.{a}_layer_norm.{tail}"), vec![D]));
                }
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    s.push((format!("{p}.{a}.{proj}.weight"), vec![D, D]));
                    if proj != "k_proj" {
                        s.push((format!("{p}.{a}.{proj}.bias"), vec![D]));
                    }
                }
            }
            for tail in ["weight", "bias"] {
                s.push((format!("{p}.final_layer_norm.{tail}"), vec![D]));
            }
            for (name, k, n) in [("fc1", D, FF), ("fc2", FF, D)] {
                s.push((format!("{p}.{name}.weight"), vec![k, n]));
                s.push((format!("{p}.{name}.bias"), vec![n]));
            }
        }
    }
    s
}
fn raw(d: &MetalDevice, m: &MappedGguf, n: &str) -> Result<Buffer> {
    d.upload(m.tensor_bytes(n).map_err(|e| error(e.to_string()))?.1)
}
fn concat(d: &MetalDevice, m: &MappedGguf, n: &[String]) -> Result<Buffer> {
    let parts = n
        .iter()
        .map(|n| {
            m.tensor_bytes(n)
                .map(|t| t.1)
                .map_err(|e| error(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    d.upload_parts(&parts)
}
fn wide(d: &MetalDevice, m: &MappedGguf, n: &str) -> Result<Buffer> {
    widen(d, raw(d, m, n)?)
}
fn widen(d: &MetalDevice, src: Buffer) -> Result<Buffer> {
    let n = src.len() / 2;
    let out = d.alloc(n * 4)?;
    let c = d.begin()?;
    c.dispatch(
        "wh_widen",
        &[&src, &out],
        &[n as u32],
        [n.div_ceil(256), 1, 1],
        256,
    );
    c.finish()?;
    Ok(out)
}
fn norm(d: &MetalDevice, m: &MappedGguf, p: &str) -> Result<Norm> {
    Ok(Norm {
        w: wide(d, m, &format!("{p}.weight"))?,
        b: wide(d, m, &format!("{p}.bias"))?,
    })
}
fn linear(d: &MetalDevice, m: &MappedGguf, p: &str, k: usize, n: usize) -> Result<Linear> {
    Ok(Linear {
        w: raw(d, m, &format!("{p}.weight"))?,
        b: wide(d, m, &format!("{p}.bias"))?,
        k,
        n,
    })
}
fn attention(d: &MetalDevice, m: &MappedGguf, p: &str) -> Result<Attention> {
    Ok(Attention {
        norm: norm(d, m, &format!("{p}.self_attn_layer_norm"))?,
        qkv: concat(
            d,
            m,
            &["q", "k", "v"].map(|q| format!("{p}.self_attn.{q}_proj.weight")),
        )?,
        bias: widen(
            d,
            concat(
                d,
                m,
                &["q", "v"].map(|q| format!("{p}.self_attn.{q}_proj.bias")),
            )?,
        )?,
        out: linear(d, m, &format!("{p}.self_attn.out_proj"), D, D)?,
    })
}
fn mlp(d: &MetalDevice, m: &MappedGguf, p: &str) -> Result<Mlp> {
    Ok(Mlp {
        norm: norm(d, m, &format!("{p}.final_layer_norm"))?,
        up: linear(d, m, &format!("{p}.fc1"), D, FF)?,
        down: linear(d, m, &format!("{p}.fc2"), FF, D)?,
    })
}
// Our converter writes this flat object. Refuse malformed, duplicate and
// out-of-vocabulary entries; no new parser dependency for a fixed schema.
fn parse_langs(s: &str) -> Result<Vec<(String, u32)>> {
    let body = s
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| error("invalid language map"))?;
    let mut out = Vec::new();
    for item in body.split(',') {
        let (k, v) = item
            .split_once(':')
            .ok_or_else(|| error("invalid language entry"))?;
        let code = k
            .trim()
            .strip_prefix("\"<|")
            .and_then(|s| s.strip_suffix("|>\""))
            .ok_or_else(|| error("invalid language key"))?;
        let id = v
            .trim()
            .parse::<u32>()
            .map_err(|_| error("invalid language id"))?;
        if !(50259..=50358).contains(&id)
            || code.is_empty()
            || !code.bytes().all(|b| b.is_ascii_lowercase())
            || out.iter().any(|(c, i)| c == code || *i == id)
        {
            return Err(error("invalid/duplicate language entry"));
        }
        out.push((code.to_owned(), id));
    }
    if out.is_empty() {
        return Err(error("empty language map"));
    }
    Ok(out)
}
