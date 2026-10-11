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
        if m == 0
            || m > CHUNK
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
            sequences.extend([
                slot as u32,
                first as u32,
                count as u32,
                (ssd_tiles.len() / 4) as u32,
            ]);
            longest = longest.max(count);
            if count >= 16 {
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
        let seq_count = sequences.len() / 4;
        let tile_count = ssd_tiles.len() / 4;
        #[cfg(test)]
        let scan_only = self.scan_only;
        #[cfg(not(test))]
        let scan_only = false;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[WIDTH as u32, m as u32, 8, 1f32.to_bits()],
            [(m * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        for l in &self.layers {
            cmd.dispatch(
                "rms",
                &[&s.x, &l.norm.buffer, &s.norm],
                &[WIDTH as u32, 0, 1e-5f32.to_bits()],
                [m, 1, 1],
                256,
            );
            match &l.mixer {
                Mixer::Mamba(w) => {
                    project(&cmd, &[(&w.input, &s.proj)], &s.norm, m);
                    cmd.dispatch(
                        "nemo_conv",
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
                    cmd.dispatch(
                        "nemo_conv_commit",
                        &[&s.proj, &w.window, &s.sequences],
                        &[],
                        [CONV.div_ceil(256), seq_count, 1],
                        256,
                    );
                    if !decodes.is_empty() || scan_only {
                        cmd.dispatch(
                            "nemo_scan",
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
                            "nemo_ssd_prepare",
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
                            "nemo_ssd_matrix",
                            &[&s.conv, &s.ssd_tiles, &s.decay, &s.dt, &s.matrix],
                            &[],
                            [64, tile_count, 1],
                            128,
                        );
                        cmd.dispatch(
                            "nemo_ssd_delta",
                            &[&s.conv, &s.ssd_tiles, &s.decay, &s.dt, &s.state_delta],
                            &[],
                            [16, 64, tile_count],
                            128,
                        );
                        cmd.dispatch(
                            "nemo_ssd_states",
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
                        cmd.dispatch(
                            "nemo_ssd_output",
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
                    cmd.dispatch(
                        "nemo_gated_norm",
                        &[&s.y, &s.proj, &w.norm.buffer, &s.yn],
                        &[1e-5f32.to_bits()],
                        [8, m, 1],
                        256,
                    );
                    project(&cmd, &[(&w.output, &s.delta)], &s.yn, m);
                }
                Mixer::Attention(w) => {
                    project(
                        &cmd,
                        &[(&w.q, &s.q), (&w.k, &s.k), (&w.v, &s.v)],
                        &s.norm,
                        m,
                    );
                    cmd.dispatch(
                        "nemo_store",
                        &[&s.k, &s.v, &w.keys, &w.values, &s.meta, &s.pages],
                        &[m as u32, self.page_stride as u32],
                        [(m * 256).div_ceil(256), 1, 1],
                        256,
                    );
                    let ap = [32, 2, self.page_stride as u32, 0, 0, SPLITS as u32];
                    if !attention_tiles.is_empty() {
                        cmd.dispatch(
                            "laguna_prefill",
                            &[
                                &s.q,
                                &w.keys,
                                &w.values,
                                &s.meta,
                                &s.pages,
                                &s.attention,
                                &s.attention_tiles,
                            ],
                            &ap,
                            [32, attention_tiles.len() / 2, 1],
                            128,
                        );
                    }
                    if !decodes.is_empty() {
                        cmd.dispatch(
                            "nemo_attention_decode",
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
                    project(&cmd, &[(&w.out, &s.delta)], &s.attention, m);
                }
                Mixer::Moe(w) => {
                    // Router planes are F32. The projection emits F32 logits
                    // and no activation quantization or host expert selection.
                    cmd.dispatch(
                        "linear",
                        &[&w.router.buffer, &s.norm, &s.router],
                        &[WIDTH as u32, 128, m as u32, 0, 1f32.to_bits()],
                        [32, m, 1],
                        128,
                    );
                    cmd.dispatch(
                        "nemo_route",
                        &[&s.router, &w.bias.buffer, &s.picks, &s.probabilities],
                        &[],
                        [m, 1, 1],
                        32,
                    );
                    let grouped = m >= 16;
                    if grouped {
                        cmd.dispatch(
                            "moe_align",
                            &[&s.picks, &s.lists, &s.counts],
                            &[(m * 6) as u32],
                            [128, 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            "moe_tiles",
                            &[&s.counts, &s.tiles],
                            &[128, 16],
                            [1, 1, 1],
                            256,
                        );
                    }
                    for (down, w, input, output, k, n) in [
                        (false, &w.up, &s.norm, &s.up, WIDTH, FF),
                        (true, &w.down, &s.up, &s.expert_out, FF, WIDTH),
                    ] {
                        let p = [k as u32, n as u32, m as u32];
                        if grouped {
                            cmd.dispatch(
                                if down {
                                    "nemo_down_grouped"
                                } else {
                                    "nemo_up_grouped"
                                },
                                &[&w.buffer, input, &s.lists, &s.counts, &s.tiles, output],
                                &p,
                                [n.div_ceil(32), (m * 6).div_ceil(16) + 128, 1],
                                128,
                            );
                        } else {
                            cmd.dispatch(
                                if down {
                                    "nemo_down_decode"
                                } else {
                                    "nemo_up_decode"
                                },
                                &[&w.buffer, input, &s.picks, output],
                                &p,
                                [n.div_ceil(4), m * 6, 1],
                                128,
                            );
                        }
                    }
                    project(&cmd, &[(&w.shared_up, &s.shared)], &s.norm, m);
                    cmd.dispatch(
                        "nemo_relu2",
                        &[&s.shared],
                        &[(m * SHARED) as u32],
                        [(m * SHARED).div_ceil(256), 1, 1],
                        256,
                    );
                    project(&cmd, &[(&w.shared_down, &s.delta)], &s.shared, m);
                    cmd.dispatch(
                        "nemo_fold",
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
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                "rms_selected",
                &[&s.x, &self.output_norm.buffer, &s.output_rows, &s.norm],
                &[WIDTH as u32, 0, 1e-5f32.to_bits()],
                [output_rows.len(), 1, 1],
                256,
            );
            project(&cmd, &[(&self.head, &s.logits)], &s.norm, output_rows.len());
        }
        self.last_gpu_seconds = cmd.finish()?;
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
