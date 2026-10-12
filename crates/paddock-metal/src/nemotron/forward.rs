use super::*;
use crate::projection::project;

impl Nemotron {
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, output_rows))
    }
    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        let m = rows.len();
        let mlx = self.embedding.ty == mlx::AFFINE8;
        let mlx_contract = mlx && crate::projection::affine_mlx();
        let reuse_decode = mlx && crate::projection::affine_packed_experts();
        if m == 0
            || m > self.scratch.ids.len() / 4
            || output_rows.len() > self.slots.len()
            || output_rows.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid Nemotron execution/output rows".into(),
            ));
        }
        let mut lengths = self
            .slots
            .iter()
            .map(|s| s.history.len())
            .collect::<Vec<_>>();
        for &(slot, token, pos) in rows {
            if slot >= self.slots.len()
                || token as usize >= VOCAB
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid Nemotron row slot={slot} token={token} pos={pos}"
                )));
            }
            lengths[slot] += 1;
        }
        let mut sequences = Vec::new();
        let mut ssd_tiles = Vec::new();
        let mut attention_tiles = Vec::new();
        let mut decodes = Vec::new();
        // Dense/expert arithmetic follows each sequence's own work, never the
        // total mixed batch width. Coalesce adjacent rows of the same contract
        // for reuse, even when callers interleave decode and prefill slots.
        let mut projection_spans: Vec<(usize, usize, bool)> = Vec::new();
        let mut seen = vec![false; self.slots.len()];
        let mut first = 0;
        let mut longest = 0;
        while first < m {
            let slot = rows[first].0;
            if seen[slot] {
                return Err(MetalError::Model(
                    "Nemotron requires contiguous rows per slot".into(),
                ));
            }
            seen[slot] = true;
            let mut end = first + 1;
            while end < m && rows[end].0 == slot {
                end += 1;
            }
            let count = end - first;
            let seq = sequences.len() / 4;
            // MLX's initial token and multi-token calls use SSD, even below
            // the GGUF throughput threshold. A warm single-token call uses
            // the reference's distinct F32 state-update arithmetic.
            let tiled = count >= 16 || (mlx && (count > 1 || rows[first].2 == 0));
            let vector = count < 16;
            if let Some((start, len, prior)) = projection_spans.last_mut()
                && *prior == vector
                && *start + *len == first
                && (!mlx_contract || vector)
            {
                *len += count;
            } else {
                projection_spans.push((first, count, vector));
            }
            sequences.extend([
                slot as u32,
                first as u32,
                count as u32,
                if mlx && !tiled {
                    u32::MAX
                } else {
                    (ssd_tiles.len() / 4) as u32
                },
            ]);
            longest = longest.max(count);
            if tiled {
                for t in (first..end).step_by(32) {
                    let n = (end - t).min(32);
                    ssd_tiles.extend([slot as u32, t as u32, n as u32, seq as u32]);
                    attention_tiles.extend([t as u32, n as u32]);
                }
            } else {
                decodes.extend((first..end).map(|r| r as u32));
            }
            first = end;
        }
        for &(slot, _, pos) in rows {
            while self.slots[slot]
                .table
                .ensure(pos as usize, &mut self.pool)
                .is_err()
            {
                if !self.evict_prefix() {
                    return Err(MetalError::Memory("Nemotron KV pool exhausted".into()));
                }
            }
            loop {
                match self.slots[slot].table.cow_at(pos as usize, &mut self.pool) {
                    Ok(Some((source, destination))) => {
                        // Fence this copy before another allocation can evict
                        // the prefix's reference to source and recycle its id.
                        self.copy_attention_page(source, destination)?;
                        break;
                    }
                    Ok(None) => break,
                    Err(_) if self.evict_prefix() => continue,
                    Err(_) => {
                        return Err(MetalError::Memory(
                            "Nemotron KV pool exhausted during prefix copy-on-write".into(),
                        ));
                    }
                }
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, s) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + s.table.blocks().len()]
                .copy_from_slice(s.table.blocks());
        }
        let s = &self.scratch;
        // Empty/aborted slots never inherit an old recurrent state, including
        // callers that use forward(token) after reset instead of prefill.
        for (slot, &present) in seen.iter().enumerate() {
            if present && self.slots[slot].history.is_empty() {
                self.copy_state(slot, None)?;
            }
        }
        // SAFETY: previous command buffers have completed; host metadata
        // describes validated, disjoint slot/row/page ownership only.
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &rows
                    .iter()
                    .flat_map(|r| [r.0 as u32, r.2])
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&pages);
            s.output_rows
                .write_u32(&output_rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.sequences.write_u32(&sequences);
            s.ssd_tiles.write_u32(&ssd_tiles);
            s.attention_tiles.write_u32(&attention_tiles);
            s.decode_rows.write_u32(&decodes);
        }
        let cmd = self.device.begin()?;
        let project = |cmd: &crate::device::Commands<'_>,
                       planes: &[(&Weight, &Buffer)],
                       input: &Buffer,
                       rows: usize| {
            if mlx {
                // Expert-up storage is dead during every backbone projection:
                // Mamba/attention do not use it; shared-expert projections run
                // only after routed down-projection has consumed it.
                project_mlx_spans(cmd, planes, input, &s.up, &projection_spans);
            } else {
                project(cmd, planes, input, rows);
            }
        };
        let round = |buffer: &Buffer, count: usize| {
            if mlx {
                cmd.dispatch(
                    "gmlx_round",
                    &[buffer],
                    &[count as u32],
                    [count.div_ceil(256), 1, 1],
                    256,
                );
            }
        };
        let seq_count = sequences.len() / 4;
        let tile_count = ssd_tiles.len() / 4;
        #[cfg(test)]
        let scan_only = self.scan_only;
        #[cfg(not(test))]
        let scan_only = false;
        cmd.dispatch(
            if mlx { "nemo_mlx_embed" } else { "embed" },
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[WIDTH as u32, m as u32, 8, 1f32.to_bits()],
            [(m * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        #[cfg(test)]
        let trace_dir = std::env::var_os("PADDOCK_NEMOTRON_MLX_TRACE").filter(|_| mlx && m > 1);
        #[cfg(test)]
        let mut traces = Vec::new();
        #[cfg(test)]
        let mut trace = |name: String, buffer: &Buffer, count: usize| -> Result<()> {
            if trace_dir.is_some() {
                let saved = self.device.alloc(count * 4)?;
                cmd.dispatch(
                    "spec_copy",
                    &[buffer, &saved],
                    &[0, 0, count as u32],
                    [count.div_ceil(256), 1, 1],
                    256,
                );
                traces.push((name, saved, count));
            }
            Ok(())
        };
        #[cfg(test)]
        trace("embed".into(), &s.x, m * WIDTH)?;
        for l in &self.layers {
            #[cfg(test)]
            if !self.reference_inputs.is_empty() {
                let i = self.layers.iter().position(|v| std::ptr::eq(v, l)).unwrap();
                let input = &self.reference_inputs[i];
                assert_eq!(input.len(), m * WIDTH * 4);
                cmd.dispatch(
                    "spec_copy",
                    &[input, &s.x],
                    &[0, 0, (m * WIDTH) as u32],
                    [(m * WIDTH).div_ceil(256), 1, 1],
                    256,
                );
            }
            cmd.dispatch(
                if mlx { "gmlx_norm" } else { "rms" },
                &[&s.x, &l.norm.buffer, &s.norm],
                &[WIDTH as u32, 0, 1e-5f32.to_bits()],
                [m, 1, 1],
                if mlx { WIDTH.div_ceil(128) * 32 } else { 256 },
            );
            #[cfg(test)]
            let first_layer = std::ptr::eq(l, &self.layers[0]);
            #[cfg(test)]
            if first_layer {
                trace("norm".into(), &s.norm, m * WIDTH)?;
            }
            match &l.mixer {
                Mixer::Mamba(w) => {
                    project(&cmd, &[(&w.input, &s.proj)], &s.norm, m);
                    #[cfg(test)]
                    if first_layer {
                        trace("proj".into(), &s.proj, m * PROJECTED)?;
                    }
                    cmd.dispatch(
                        if mlx { "nemo_mlx_conv" } else { "nemo_conv" },
                        &[
                            &s.proj,
                            &w.window,
                            &w.conv_w.buffer,
                            &w.conv_b.buffer,
                            &s.sequences,
                            &s.conv,
                        ],
                        &[],
                        [(longest * CONV).div_ceil(256), seq_count, 1],
                        256,
                    );
                    #[cfg(test)]
                    if first_layer {
                        trace("conv".into(), &s.conv, m * CONV)?;
                    }
                    cmd.dispatch(
                        "nemo_conv_commit",
                        &[&s.proj, &w.window, &s.sequences],
                        &[],
                        [CONV.div_ceil(256), seq_count, 1],
                        256,
                    );
                    if !decodes.is_empty() || scan_only {
                        cmd.dispatch(
                            if mlx { "nemo_mlx_scan" } else { "nemo_scan" },
                            &[
                                &s.conv,
                                &s.proj,
                                &w.a.buffer,
                                &w.d.buffer,
                                &w.dt.buffer,
                                &w.state,
                                &s.sequences,
                                &s.y,
                            ],
                            &[u32::from(scan_only)],
                            [16, 64, seq_count],
                            128,
                        );
                    }
                    if tile_count > 0 && !scan_only {
                        cmd.dispatch(
                            if mlx {
                                "nemo_mlx_ssd_prepare"
                            } else {
                                "nemo_ssd_prepare"
                            },
                            &[
                                &s.proj,
                                &w.a.buffer,
                                &w.dt.buffer,
                                &s.ssd_tiles,
                                &s.decay,
                                &s.dt,
                            ],
                            &[],
                            [64, tile_count, 1],
                            32,
                        );
                        cmd.dispatch(
                            if mlx_contract {
                                "nemo_mlx_ssd_matrix_reference"
                            } else if mlx {
                                "nemo_mlx_ssd_matrix"
                            } else {
                                "nemo_ssd_matrix"
                            },
                            &[&s.conv, &s.ssd_tiles, &s.decay, &s.dt, &s.matrix],
                            &[],
                            [64, tile_count, 1],
                            128,
                        );
                        cmd.dispatch(
                            if mlx_contract {
                                "nemo_mlx_ssd_delta_reference"
                            } else {
                                "nemo_ssd_delta"
                            },
                            &[&s.conv, &s.ssd_tiles, &s.decay, &s.dt, &s.state_delta],
                            &[],
                            [16, 64, tile_count],
                            128,
                        );
                        cmd.dispatch(
                            if mlx {
                                "nemo_mlx_ssd_states"
                            } else {
                                "nemo_ssd_states"
                            },
                            &[
                                &w.state,
                                &s.state_delta,
                                &s.decay,
                                &s.sequences,
                                &s.ssd_tiles,
                                &s.state_in,
                            ],
                            &[],
                            [STATE.div_ceil(256), seq_count, 1],
                            256,
                        );
                        if mlx_contract {
                            cmd.dispatch(
                                "nemo_mlx_ssd_output_reference",
                                &[
                                    &s.conv,
                                    &s.ssd_tiles,
                                    &s.decay,
                                    &s.matrix,
                                    &s.state_in,
                                    &w.d.buffer,
                                    &s.dt,
                                    &s.y,
                                ],
                                &[],
                                [4, 64, tile_count],
                                128,
                            );
                        } else {
                            cmd.dispatch(
                                if mlx {
                                    "nemo_mlx_ssd_output"
                                } else {
                                    "nemo_ssd_output"
                                },
                                &[
                                    &s.conv,
                                    &s.ssd_tiles,
                                    &s.decay,
                                    &s.matrix,
                                    &s.state_in,
                                    &w.d.buffer,
                                    &s.y,
                                ],
                                &[],
                                [4, 64, tile_count],
                                128,
                            );
                        }
                    }
                    #[cfg(test)]
                    if first_layer {
                        trace("y".into(), &s.y, m * INNER)?;
                        trace("state".into(), &w.state, STATE)?;
                    }
                    cmd.dispatch(
                        if mlx {
                            "nemo_mlx_gated_norm"
                        } else {
                            "nemo_gated_norm"
                        },
                        &[&s.y, &s.proj, &w.norm.buffer, &s.yn],
                        &[1e-5f32.to_bits()],
                        [8, m, 1],
                        256,
                    );
                    project(&cmd, &[(&w.output, &s.delta)], &s.yn, m);
                    #[cfg(test)]
                    if first_layer {
                        trace("yn".into(), &s.yn, m * INNER)?;
                        trace("delta".into(), &s.delta, m * WIDTH)?;
                    }
                }
                Mixer::Attention(w) => {
                    project(
                        &cmd,
                        &[(&w.q, &s.q), (&w.k, &s.k), (&w.v, &s.v)],
                        &s.norm,
                        m,
                    );
                    cmd.dispatch(
                        if mlx { "nemo_mlx_store" } else { "nemo_store" },
                        &[&s.k, &s.v, &w.keys, &w.values, &s.meta, &s.pages],
                        &[m as u32, self.page_stride as u32],
                        [(m * 256).div_ceil(256), 1, 1],
                        256,
                    );
                    let ap = [32, 2, self.page_stride as u32, 0, 0, SPLITS as u32];
                    if !attention_tiles.is_empty() {
                        let register_attention = mlx_contract && self.device.tensor_accelerated();
                        if mlx_contract {
                            cmd.dispatch(
                                "mlx_input_compact",
                                &[&s.q, &s.up],
                                &[4096, m as u32],
                                [(m * 4096).div_ceil(256), 1, 1],
                                256,
                            );
                        }
                        cmd.dispatch(
                            if register_attention {
                                "nemo_mlx_prefill_nax"
                            } else if mlx_contract {
                                "nemo_mlx_prefill_tensor"
                            } else if mlx {
                                "llama_mlx_prefill"
                            } else {
                                "laguna_prefill"
                            },
                            &[
                                if mlx_contract { &s.up } else { &s.q },
                                &w.keys,
                                &w.values,
                                &s.meta,
                                &s.pages,
                                &s.attention,
                                &s.attention_tiles,
                            ],
                            &ap,
                            [32, attention_tiles.len() / 2, 1],
                            if register_attention { 64 } else { 128 },
                        );
                    }
                    if !decodes.is_empty() {
                        cmd.dispatch(
                            if mlx {
                                "nemo_mlx_attention_decode"
                            } else {
                                "nemo_attention_decode"
                            },
                            &[
                                &s.q,
                                &w.keys,
                                &w.values,
                                &s.meta,
                                &s.pages,
                                &s.decode_rows,
                                &s.parts,
                            ],
                            &ap,
                            [2, decodes.len(), SPLITS],
                            128,
                        );
                        cmd.dispatch(
                            "gemma_merge",
                            &[&s.parts, &s.attention, &s.decode_rows],
                            &[32, SPLITS as u32, 128],
                            [decodes.len() * 32, 1, 1],
                            32,
                        );
                    }
                    round(&s.attention, m * 4096);
                    project(&cmd, &[(&w.out, &s.delta)], &s.attention, m);
                }
                Mixer::Moe(w) => {
                    // Router planes are F32. The projection emits F32 logits
                    // and no activation quantization or host expert selection.
                    // Reuse amortizes F32 weights during prefill; narrow
                    // decode rows need the original higher-occupancy kernel.
                    let reuse_router = reuse_decode && m >= 32;
                    cmd.dispatch(
                        if reuse_router {
                            "nemo_mlx_router4"
                        } else {
                            "linear"
                        },
                        &[&w.router.buffer, &s.norm, &s.router],
                        &[WIDTH as u32, 128, m as u32, 0, 1f32.to_bits()],
                        [
                            if reuse_router { 16 } else { 32 },
                            m.div_ceil(if reuse_router { 4 } else { 1 }),
                            1,
                        ],
                        128,
                    );
                    round(&s.router, m * 128);
                    cmd.dispatch(
                        "nemo_route",
                        &[&s.router, &w.bias.buffer, &s.picks, &s.probabilities],
                        &[],
                        [m, 1, 1],
                        32,
                    );
                    let grouped = if mlx {
                        projection_spans.iter().any(|s| !s.2)
                    } else {
                        m >= 16
                    };
                    let grouped = if mlx_contract {
                        projection_spans.iter().any(|s| s.1 * 6 >= 128 * 4)
                    } else {
                        grouped
                    };
                    let wide = mlx && m >= 256;
                    let wide = mlx && (wide || mlx_contract);
                    #[cfg(test)]
                    let wide = wide && !crate::projection::affine_reference();
                    let packed_experts = mlx_contract && crate::projection::affine_packed_experts();
                    // True row reuse, not merely sorted dispatch order. Below
                    // 64 rows the alignment/underfilled groups can lose; large
                    // batches retain their existing bounded tile arena.
                    // The full 128-row mixed wave wins with eight-row reuse
                    // on Apple10; sparse/smaller and wider waves measured worse.
                    let vector_tile = if cmd.tensor_accelerated() && m == 128 {
                        8
                    } else {
                        4
                    };
                    let vector_group = reuse_decode
                        && !grouped
                        && m >= 64
                        && (1 + 2 * ((m * 6).div_ceil(vector_tile) + 128)) * 4 <= s.tiles.len();
                    let expert_tile = if packed_experts
                        && m * 6 / 128 >= if cmd.tensor_accelerated() { 48 } else { 64 }
                    {
                        64
                    } else if wide {
                        32
                    } else {
                        16
                    };
                    if grouped || vector_group {
                        cmd.dispatch(
                            "moe_align",
                            &[&s.picks, &s.lists, &s.counts],
                            &[(m * 6) as u32],
                            [128, 1, 1],
                            256,
                        );
                    }
                    if grouped || vector_group {
                        cmd.dispatch(
                            "moe_tiles",
                            &[&s.counts, &s.tiles],
                            &[
                                128,
                                if vector_group {
                                    vector_tile as u32
                                } else {
                                    expert_tile as u32
                                },
                            ],
                            [1, 1, 1],
                            256,
                        );
                        if grouped && packed_experts {
                            // Routing logits are dead after expert selection.
                            cmd.dispatch(
                                "nemo_mlx_expert_offsets",
                                &[&s.counts, &s.router],
                                &[],
                                [1, 1, 1],
                                128,
                            );
                        }
                    }
                    for (down, w, input, output, k, n) in [
                        (false, &w.up, &s.norm, &s.up, WIDTH, FF),
                        (true, &w.down, &s.up, &s.expert_out, FF, WIDTH),
                    ] {
                        let p = [k as u32, n as u32, m as u32];
                        if vector_group {
                            cmd.dispatch(
                                if vector_tile == 8 && down {
                                    "nemo_mlx_down_vector_group8"
                                } else if vector_tile == 8 {
                                    "nemo_mlx_up_vector_group8"
                                } else if down {
                                    "nemo_mlx_down_vector_group4"
                                } else {
                                    "nemo_mlx_up_vector_group4"
                                },
                                &[&w.buffer, input, &s.lists, &s.counts, &s.tiles, output],
                                &p,
                                [n.div_ceil(8), (m * 6).div_ceil(vector_tile) + 128, 1],
                                128,
                            );
                        } else if grouped && packed_experts {
                            // Mamba projection scratch is dead throughout MoE.
                            assert!(m * 6 * k * 2 <= s.proj.len());
                            cmd.dispatch(
                                "nemo_mlx_expert_pack",
                                &[input, &s.lists, &s.counts, &s.router, &s.tiles, &s.proj],
                                &[k as u32, m as u32, u32::from(down), expert_tile as u32],
                                [
                                    (expert_tile * k / 4).div_ceil(256),
                                    (m * 6).div_ceil(expert_tile) + 128,
                                    1,
                                ],
                                256,
                            );
                            cmd.dispatch(
                                if expert_tile == 64 && cmd.tensor_accelerated() && down {
                                    "nemo_mlx_down_reuse64_k128"
                                } else if expert_tile == 64 && cmd.tensor_accelerated() {
                                    "nemo_mlx_up_reuse64_k128"
                                } else if expert_tile == 64 && down {
                                    "nemo_mlx_down_reuse64"
                                } else if expert_tile == 64 {
                                    "nemo_mlx_up_reuse64"
                                } else if down {
                                    "nemo_mlx_down_reuse"
                                } else {
                                    "nemo_mlx_up_reuse"
                                },
                                &[
                                    &w.buffer, &s.proj, &s.lists, &s.counts, &s.tiles, &s.router,
                                    output,
                                ],
                                &p,
                                [n.div_ceil(64), (m * 6).div_ceil(expert_tile) + 128, 1],
                                128,
                            );
                        } else if grouped {
                            #[cfg(test)]
                            if mlx && crate::projection::affine_reference() {
                                cmd.dispatch(
                                    if down {
                                        "nemo_mlx_down_f32_reference"
                                    } else {
                                        "nemo_mlx_up_f32_reference"
                                    },
                                    &[&w.buffer, input, &s.lists, &s.counts, &s.tiles, output],
                                    &p,
                                    [n.div_ceil(32), (m * 6).div_ceil(16) + 128, 1],
                                    128,
                                );
                                continue;
                            }
                            cmd.dispatch(
                                if mlx_contract && down {
                                    "nemo_mlx_down_bf16"
                                } else if mlx_contract {
                                    "nemo_mlx_up_bf16"
                                } else if wide && down {
                                    "nemo_mlx_down_grouped32"
                                } else if wide {
                                    "nemo_mlx_up_grouped32"
                                } else if mlx && down {
                                    "nemo_mlx_down_grouped"
                                } else if mlx {
                                    "nemo_mlx_up_grouped"
                                } else if down {
                                    "nemo_down_grouped"
                                } else {
                                    "nemo_up_grouped"
                                },
                                &[&w.buffer, input, &s.lists, &s.counts, &s.tiles, output],
                                &p,
                                [
                                    n.div_ceil(if wide { 64 } else { 32 }),
                                    (m * 6).div_ceil(expert_tile) + 128,
                                    1,
                                ],
                                128,
                            );
                        } else {
                            cmd.dispatch(
                                if reuse_decode && down {
                                    "nemo_mlx_down_decode_reuse"
                                } else if reuse_decode {
                                    "nemo_mlx_up_decode_reuse"
                                } else if mlx && down {
                                    "nemo_mlx_down_decode"
                                } else if mlx {
                                    "nemo_mlx_up_decode"
                                } else if down {
                                    "nemo_down_decode"
                                } else {
                                    "nemo_up_decode"
                                },
                                &[&w.buffer, input, &s.picks, output],
                                &p,
                                [n.div_ceil(if reuse_decode { 16 } else { 4 }), m * 6, 1],
                                128,
                            );
                        }
                        if mlx && grouped {
                            // The gathered prefill tile has a different affine
                            // contraction from decode. Restore live rows before
                            // the down projection consumes their up activations.
                            for &(first, count, vector) in &projection_spans {
                                if if mlx_contract {
                                    count * 6 >= 128 * 4
                                } else {
                                    !vector
                                } {
                                    continue;
                                }
                                cmd.dispatch_at(
                                    if reuse_decode && down {
                                        "nemo_mlx_down_decode_reuse"
                                    } else if reuse_decode {
                                        "nemo_mlx_up_decode_reuse"
                                    } else if down {
                                        "nemo_mlx_down_decode"
                                    } else {
                                        "nemo_mlx_up_decode"
                                    },
                                    &[&w.buffer, input, &s.picks, output],
                                    &[
                                        0,
                                        first * if down { 6 } else { 1 } * k * 4,
                                        first * 6 * 4,
                                        first * 6 * n * 4,
                                    ],
                                    &[k as u32, n as u32, count as u32],
                                    [n.div_ceil(if reuse_decode { 16 } else { 4 }), count * 6, 1],
                                    128,
                                );
                            }
                        }
                    }
                    project(&cmd, &[(&w.shared_up, &s.shared)], &s.norm, m);
                    cmd.dispatch(
                        if mlx { "nemo_mlx_relu2" } else { "nemo_relu2" },
                        &[&s.shared],
                        &[(m * SHARED) as u32],
                        [(m * SHARED).div_ceil(256), 1, 1],
                        256,
                    );
                    project(&cmd, &[(&w.shared_down, &s.delta)], &s.shared, m);
                    cmd.dispatch(
                        if mlx { "nemo_mlx_fold" } else { "nemo_fold" },
                        &[&s.expert_out, &s.probabilities, &s.delta],
                        &[m as u32],
                        [(m * WIDTH).div_ceil(256), 1, 1],
                        256,
                    );
                }
            }
            cmd.dispatch(
                "residual",
                &[&s.x, &s.delta],
                &[(m * WIDTH) as u32, 1f32.to_bits()],
                [(m * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            round(&s.x, m * WIDTH);
            #[cfg(test)]
            trace(
                format!(
                    "layer-{}",
                    self.layers.iter().position(|v| std::ptr::eq(v, l)).unwrap()
                ),
                &s.x,
                m * WIDTH,
            )?;
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                if mlx {
                    "nemo_mlx_rms_selected"
                } else {
                    "rms_selected"
                },
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[WIDTH as u32, 0, 1e-5f32.to_bits()],
                [output_rows.len(), 1, 1],
                if mlx { WIDTH.div_ceil(128) * 32 } else { 256 },
            );
            // Output rows are a compact selection, not the input row plan.
            // Always use the decode contraction for the vocabulary head.
            if mlx {
                project_mlx_spans(
                    &cmd,
                    &[(&self.head, &s.logits)],
                    &s.norm,
                    &s.up,
                    &[(0, output_rows.len(), true)],
                );
            } else {
                crate::projection::project(
                    &cmd,
                    &[(&self.head, &s.logits)],
                    &s.norm,
                    output_rows.len(),
                );
            }
        }
        self.last_gpu_seconds = cmd.finish()?;
        #[cfg(test)]
        if let Some(dir) = trace_dir {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            for (name, buffer, count) in &traces {
                let bytes = unsafe { buffer.read_f32(0, *count) }
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                std::fs::write(dir.join(format!("{name}-{m}.f32")), bytes).unwrap();
            }
        }
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        // SAFETY: GPU work finished; only the requested final logits escape.
        let logits = unsafe { s.logits.read_f32(0, output_rows.len() * VOCAB) };
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(MetalError::Model("nonfinite Nemotron logits".into()));
        }
        Ok(logits)
    }
}

fn project_mlx_spans(
    cmd: &crate::device::Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    compact: &Buffer,
    spans: &[(usize, usize, bool)],
) {
    // Only combine full tensor contractions with the same K reduction. Small
    // QMM changes split count with M, so merging those would change BF16 graph
    // boundaries. Coalescing here does not merge recurrent sequences or KV.
    let combined = if crate::projection::affine_mlx() && crate::projection::affine_packed_experts()
    {
        coalesce_tensor_spans(spans, |count| {
            planes
                .iter()
                .all(|(w, _)| mlx_split_parts(w.k, w.n, count) == 1)
        })
    } else {
        std::borrow::Cow::Borrowed(spans)
    };
    let spans = combined.as_ref();
    for &(first, count, vector) in spans {
        let mut compact_k = None;
        let (kernel, columns, tile) = if vector {
            if count == 1 {
                ("nemo_mlx_vector1", 4, 1)
            } else {
                ("nemo_mlx_vector4", 4, 4)
            }
        } else if count < 32 {
            ("nemo_mlx_project_exact", 32, 16)
        } else if count < 64 {
            ("nemo_mlx_project_exact_mixed", 64, 32)
        } else {
            ("nemo_mlx_project_exact_large", 64, 64)
        };
        #[cfg(test)]
        let (kernel, columns, tile) = if crate::projection::affine_reference() {
            if vector {
                ("dg_affine_vector", 4, 1)
            } else {
                ("dg_project", 32, 16)
            }
        } else {
            (kernel, columns, tile)
        };
        let (kernel, columns, tile) = if crate::projection::affine_mlx() && !vector {
            if count < 32 {
                ("nemo_mlx_project", 32, 16)
            } else {
                ("nemo_mlx_project_wide", 64, 32)
            }
        } else {
            (kernel, columns, tile)
        };
        for &(w, out) in planes {
            if crate::projection::affine_mlx() {
                // Small-QMM split reduction rounds each part AND each addition
                // to BF16. F32 carriers must not elide those graph boundaries.
                let limit = if w.k <= 4096 && w.n <= 4096 { 25 } else { 13 };
                let parts = mlx_split_parts(w.k, w.n, count);
                if !vector && count >= limit && parts > 1 {
                    // Keep partials beyond the shared input plane. The routed
                    // expert-up scratch is much larger than both bounded views.
                    let offset = (count * w.k * 2).next_multiple_of(256);
                    assert!(offset + parts * count * w.n * 2 <= compact.len());
                    cmd.dispatch_at(
                        "nemo_mlx_project_split",
                        &[&w.buffer, input, compact],
                        &[0, first * w.k * 4, offset],
                        &[w.k as u32, w.n as u32, count as u32, w.ty, 1, parts as u32],
                        [w.n.div_ceil(32), count.div_ceil(16), parts],
                        128,
                    );
                    cmd.dispatch_at(
                        "nemo_mlx_project_split_sum",
                        &[compact, out],
                        &[offset, first * w.n * 4],
                        &[count as u32, w.n as u32, parts as u32],
                        [(count * w.n).div_ceil(256), 1, 1],
                        256,
                    );
                    continue;
                }
                if !vector && count >= 128 {
                    if compact_k != Some(w.k) {
                        assert!(count * w.k * 2 <= compact.len());
                        cmd.dispatch_at(
                            "mlx_input_compact",
                            &[input, compact],
                            &[first * w.k * 4, 0],
                            &[w.k as u32, count as u32],
                            [(count * w.k).div_ceil(256), 1, 1],
                            256,
                        );
                        compact_k = Some(w.k);
                    }
                    // The wider M tile wins only with enough occupied work.
                    // Narrow K/V and older GPUs retain the qualified 64 tile.
                    let wide = cmd.tensor_accelerated()
                        && crate::projection::affine_packed_experts()
                        && count >= 1024
                        && w.n >= WIDTH;
                    let kernel = if wide {
                        "nemo_mlx_project_reuse128"
                    } else {
                        "nemo_mlx_project_reuse"
                    };
                    #[cfg(test)]
                    let kernel = if crate::projection::affine_packed_experts() {
                        kernel
                    } else {
                        "nemo_mlx_project_reuse_baseline"
                    };
                    cmd.dispatch_at(
                        kernel,
                        &[&w.buffer, compact, out],
                        &[0, 0, first * w.n * 4],
                        &[w.k as u32, w.n as u32, count as u32, w.ty, 1],
                        [
                            w.n.div_ceil(64),
                            count.div_ceil(if wide { 128 } else { 64 }),
                            1,
                        ],
                        128,
                    );
                    continue;
                }
                let (name, columns, tile) = if vector || count < limit {
                    if crate::projection::affine_packed_experts() {
                        ("nemo_mlx_vector4_reuse", 8, 4)
                    } else {
                        ("nemo_mlx_vector4", 4, 4)
                    }
                } else {
                    (kernel, columns, tile)
                };
                cmd.dispatch_at(
                    name,
                    &[&w.buffer, input, out],
                    &[0, first * w.k * 4, first * w.n * 4],
                    &[
                        w.k as u32,
                        w.n as u32,
                        count as u32,
                        w.ty,
                        1,
                        parts as u32,
                        1,
                    ],
                    [w.n.div_ceil(columns), count.div_ceil(tile), 1],
                    128,
                );
                continue;
            }
            // Both views remain within the model's fixed CHUNK scratch.
            cmd.dispatch_at(
                kernel,
                &[&w.buffer, input, out],
                &[0, first * w.k * 4, first * w.n * 4],
                &[w.k as u32, w.n as u32, count as u32, w.ty, 1],
                [w.n.div_ceil(columns), count.div_ceil(tile), 1],
                128,
            );
        }
    }
}

fn mlx_split_parts(k: usize, n: usize, rows: usize) -> usize {
    let mut parts = (512 / (n.div_ceil(32) * rows.div_ceil(32)))
        .max(1)
        .min(k / 64);
    while parts > 1 && !k.is_multiple_of(parts * 64) {
        parts -= 1;
    }
    parts
}

fn coalesce_tensor_spans(
    spans: &[(usize, usize, bool)],
    same_reduction: impl Fn(usize) -> bool,
) -> std::borrow::Cow<'_, [(usize, usize, bool)]> {
    if spans.len() < 2 {
        return std::borrow::Cow::Borrowed(spans);
    }
    let mut result: Vec<(usize, usize, bool)> = Vec::with_capacity(spans.len());
    let mut prior_compatible = false;
    for &(first, count, vector) in spans {
        let compatible = !vector && count >= 128 && same_reduction(count);
        if compatible
            && prior_compatible
            && let Some((start, len, _)) = result.last_mut()
            && *start + *len == first
        {
            *len += count;
        } else {
            result.push((first, count, vector));
        }
        prior_compatible = compatible;
    }
    if result.len() == spans.len() {
        std::borrow::Cow::Borrowed(spans)
    } else {
        std::borrow::Cow::Owned(result)
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    #[test]
    fn coalescing_keeps_vector_and_splitk_boundaries() {
        let spans = [
            (0, 1, true),
            (1, 129, false),
            (130, 257, false),
            (387, 33, false),
            (420, 128, false),
            (548, 128, false),
        ];
        let dense = coalesce_tensor_spans(&spans, |rows| mlx_split_parts(2688, 3712, rows) == 1);
        assert_eq!(
            dense.as_ref(),
            &[
                (0, 1, true),
                (1, 386, false),
                (387, 33, false),
                (420, 256, false)
            ]
        );
        let kv = coalesce_tensor_spans(&spans, |rows| mlx_split_parts(2688, 256, rows) == 1);
        assert_eq!(kv.as_ref(), &spans);
        let gap = [(0, 128, false), (129, 128, false)];
        assert_eq!(coalesce_tensor_spans(&gap, |_| true).as_ref(), &gap);
        for rows in [128, 129, 257, 512, 1024] {
            for k in [64, 1856, 2688, 4096] {
                for n in [256, 2688, 3712, 4096, 10304] {
                    if mlx_split_parts(k, n, rows) == 1 {
                        assert_eq!(mlx_split_parts(k, n, rows * 4), 1);
                    }
                }
            }
        }
    }
}
