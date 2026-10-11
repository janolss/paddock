//! Device row sampling for Flash-Next: the per-row plans packed into the
//! shared `pd_sample_rows` family's words, the reasoning stop mask (pack slot
//! 850) launched ahead of the samplers on the rows the service masked, and
//! the readbacks.

use super::*;

impl Qwen4ExpGpu {
    /// Pack the per-row sampling plans into the device param words. Mode codes
    /// match the shared `pd_sample_rows` family every other lane uses:
    /// 1 = greedy, 2 = temperature-categorical, 5/6 = truncation (5 when the
    /// top-k head fits the 64-wide superset the device selection walks).
    pub(super) fn pack_samp_par(
        plans: &[crate::generator::RowSample],
    ) -> (Vec<u32>, Option<Vec<u32>>) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let mut any_trunc = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, any_trunc.then_some(tpar))
    }

    /// Sample `d_out` rows 0..r on device; only `Host`-plan rows pay a
    /// vocab-row readback. `d_out` already holds this tick's logits.
    pub(super) fn sample_rows_from_logits(
        &mut self,
        rows: &[(usize, u32)],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        // `d_out` holds one row per DECODED row, in `rows` order - hole rows
        // never reached the walk. Plans are indexed by the scheduler's slot,
        // so they are compacted the same way before packing and the ids are
        // scattered back at the end.
        let live: Vec<RowSample> = rows.iter().map(|&(i, _)| plans[i]).collect();
        let at: Vec<usize> = rows.iter().map(|&(i, _)| i).collect();
        let (packed, host) = self.sample_out_rows(&live, &at)?;
        let mut ids = vec![0u32; plans.len()];
        for (j, &(i, _)) in rows.iter().enumerate() {
            ids[i] = packed[j];
        }
        let host_rows = host.into_iter().map(|(j, row)| (rows[j].0, row)).collect();
        Ok(SampledStep { ids, host_rows })
    }

    /// Sample `d_out` rows `[0, plans.len())` on device, row j by `plans[j]`:
    /// the ids in row order, and the `Host`-plan rows' logits read back as
    /// (row, logits). Row j is the call's row `at[j]` - where the service's
    /// stop masks for it sit.
    pub(super) fn sample_out_rows(
        &mut self,
        plans: &[crate::generator::RowSample],
        at: &[usize],
    ) -> Result<(Vec<u32>, Vec<(usize, Vec<f32>)>), GpuModelError> {
        use crate::generator::RowSample;
        let vocab = self.cfg.vocab;
        let exec = self.exec.clone();
        let r = plans.len();
        if r == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (par, tpar) = Self::pack_samp_par(plans);
        {
            let sc = &mut self.sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| crate::gpu::GpuError::Driver("d_par slice".into()))?;
            exec.stream
                .memcpy_htod(&par, &mut v)
                .map_err(crate::gpu::from_driver)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| crate::gpu::GpuError::Driver("d_tpar slice".into()))?;
                exec.stream
                    .memcpy_htod(t, &mut v)
                    .map_err(crate::gpu::from_driver)?;
            }
            self.stop.launch(&exec, &mut sc.d_out, 0, vocab, at)?;
            exec.sample_rows_at(&sc.d_out, &sc.d_par, 0, &mut sc.d_ids, 0, r, vocab)?;
            if tpar.is_some() {
                exec.sample_rows_t_at(
                    &sc.d_out,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_ids,
                    0,
                    r,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.d_out,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_ids,
                    0,
                    r,
                    vocab,
                )?;
            }
        }
        let ids_view = self
            .sc
            .d_ids
            .try_slice(0..r)
            .ok_or_else(|| crate::gpu::GpuError::Driver("d_ids slice".into()))?;
        let packed = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(crate::gpu::from_driver)?;
        let mut host_rows = Vec::new();
        for (j, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                host_rows.push((j, self.out_row(j)?));
            }
        }
        Ok((packed, host_rows))
    }

    /// `d_out` row `j` read back.
    pub(super) fn out_row(&self, j: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.cfg.vocab;
        let v = self
            .sc
            .d_out
            .try_slice(j * vocab..(j + 1) * vocab)
            .ok_or_else(|| crate::gpu::GpuError::Driver("d_out row slice".into()))?;
        Ok(self
            .exec
            .stream
            .clone_dtoh(&v)
            .map_err(crate::gpu::from_driver)?)
    }
}
