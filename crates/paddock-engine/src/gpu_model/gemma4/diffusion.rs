//! DiffusionGemma's block-diffusion lane on the gemma4 body.
//!
//! The model is Gemma 4 26B-A4B generating by iterative denoising of a
//! CANVAS: W (= 256) positions at `[L, L+W)` past the committed prefix,
//! initialised with uniformly random vocab ids, run through the same layers
//! with bidirectional attention over prefix + canvas, sampled, partly
//! accepted, re-noised, and fed back to itself through a self-conditioning
//! MLP until the argmax canvas is stable and confident (or 48 steps), then
//! committed by a plain causal prefill of the block. The reference for the
//! loop is transformers' `generation_diffusion_gemma.py`; for the attention
//! shape, the HF modular file (decoder attention `is_causal = False`, sliding
//! layers symmetric around each row) and vLLM's `diffusion_gemma.py` (per-
//! request causal flag). Read, not copied - every kernel here is ours.
//!
//! How it rides the family:
//!
//! - the canvas step is a prefill-shaped pass over W rows per slot through
//!   `prefill_layers`, rope at the TRUE positions (`pf_pos = L + i`) and the
//!   attention BOUND at the block end (`pf_attn_pos = L + W - 1`) - the same
//!   two arrays the multimodal image spans and the DFlash block use. The
//!   canvas' K/V land past the committed cursor and are overwritten by the
//!   next step (position-keyed scratch, exactly the spec-verify rows'
//!   contract), so nothing is rolled back;
//! - the head runs over ALL W rows (`Plane::gemm`, as the DFlash drafter's
//!   head does), the logits stay on device, and `canvas_sample` /
//!   `canvas_accept` (slots 648/649) do the sampler math in place;
//! - self-conditioning = `softmax(prev logits) @ E * sqrt(embd)` through a
//!   gated GELU MLP (`self_cond_*`), added to the canvas embeddings and
//!   RMS-normed weightlessly before layer 0. The `@ E` is an NN product the NT
//!   weight kernels cannot express, so the embedding is held TRANSPOSED as a
//!   bf16 plane (`E^T`, 1.48 GB on the 262k vocab) built by slot 647 at load
//!   and multiplied with `bf16_gemm` - whose tensor-core arm casts the f32
//!   probs to bf16 in smem, the reference's own `.to(embed dtype)`. SOTA
//!   target recorded, not built: an NN kernel over the raw Q8_0 rows (zero
//!   extra bytes) for the 24 GB cards where the Q4 lane + this plane is tight.
//!
//! The sliding-window START on the SWA layers comes from each row's TRUE
//! position (`pf_pos`, through the pack's `win_pos` entries), the ceiling
//! from its bound (`pf_attn_pos`) - the canvas' first rows keep their oldest
//! 1024 keys exactly as the reference reads them. The lane refuses a pack
//! without those entries (`has_canvas_ops`): with the floor derived from the
//! bound, rows lost up to W-1 = 255 keys once the block end passed key
//! 1024, and the loss showed as 5 flipped positions / worst KL 3.3 on a
//! 1066-token prompt against the HF reference.

use cudarc::driver::CudaSlice;

use super::GpuGemma4;
use super::load::LoadError;
use crate::gpu::{CanvasStatus, GluAct, GpuError, GpuExecutor, QuantTensor};

use super::{EmbdTable, Plane, kq_rows};
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

/// The sampler constants the model ships (`generation_config.json`) - the
/// authors' own settings, elected as the serving defaults. `t_max`/`t_min`
/// bound the linear temperature decay over the step budget; `entropy_bound`
/// is the joint-MI bound of the accept step; a canvas is done when its mean
/// entropy is under `confidence` AND its argmax was stable for `stability`
/// steps; `max_steps` caps the loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionConfig {
    pub max_steps: u32,
    pub t_max: f32,
    pub t_min: f32,
    pub entropy_bound: f32,
    pub confidence: f32,
    pub stability: u32,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        // google/diffusiongemma-26B-A4B-it generation_config.json
        Self {
            max_steps: 48,
            t_max: 0.8,
            t_min: 0.4,
            entropy_bound: 0.1,
            confidence: 0.005,
            stability: 1,
        }
    }
}

impl DiffusionConfig {
    /// The reference's `LinearTemperatureScheduleLogitsProcessor`: the loop
    /// counts `cur_step` DOWN from `max_steps` to 1, and `t = t_min + (t_max
    /// - t_min) * cur_step / max_steps` - so the first step runs at t_max and
    /// the last one just above t_min. `step` here is 0-based ascending.
    pub fn temperature(&self, step: u32) -> f32 {
        let cur = self.max_steps.saturating_sub(step).max(1) as f32;
        self.t_min + (self.t_max - self.t_min) * (cur / self.max_steps as f32)
    }
}

/// The weights DiffusionGemma adds to the gemma4 body, plus the transposed
/// embedding the self-conditioning matmul needs.
pub(crate) struct DiffusionLane {
    /// `self_cond_pre_norm.weight` [n_embd] - RMS norm of the soft embedding
    pub sc_pre_norm: CudaSlice<f32>,
    /// the gated MLP: gate/up [n_embd -> n_ff], down [n_ff -> n_embd], in
    /// the file's class and padded to the family's served FFN width like
    /// the shared FFN (`load::ffn_plane`)
    pub sc_gate: Plane,
    pub sc_up: Plane,
    pub sc_down: Plane,
    /// E^T as a bf16 `[in = vocab, out = embd]` plane (dims in the family's
    /// `[in, out]` convention): `bf16_gemm(embd_t, probs) = probs @ E`.
    pub embd_t: QuantTensor,
    /// all-ones [n_embd] for the weightless post-norm of the combined
    /// embedding (the rmsnorm kernels are `x * inv_rms * w`)
    pub ones: CudaSlice<f32>,
    /// canvas width the file declares
    pub canvas_len: usize,
    /// the MLP's SERVED hidden width (2112 on the 26B-A4B, padded to 2176 /
    /// 2304 like the shared FFN - the pad rows are exact zeros)
    pub n_ff: usize,
    pub cfg: DiffusionConfig,
    /// device bytes this lane holds beyond the body
    pub bytes: u64,
}

impl DiffusionLane {
    /// Load the lane's planes off the file and build E^T from the embedding
    /// the body already holds: the raw Q8_0 `[vocab][embd]` upload, or the
    /// repacked k-quant head (a k-quant file keeps no raw table). `ffn_align`
    /// is the shared FFN's pad alignment, so the MLP lands at the same
    /// served width the scratch planes are sized for.
    pub(crate) fn load(
        exec: &GpuExecutor,
        map: &MappedGguf,
        embd: EmbdTable<'_>,
        n_embd: usize,
        n_vocab: usize,
        canvas_len: usize,
        ffn_align: usize,
    ) -> Result<Self, LoadError> {
        if !exec.has_canvas_ops() {
            return Err(LoadError::Gpu(GpuError::MissingOp(
                "canvas ops (pack too old)",
            )));
        }
        let sc_pre_norm = exec.upload(map, "self_cond_pre_norm.weight")?.buf;
        let sc_gate = super::load::ffn_plane(exec, map, "self_cond_gate.weight", true, ffn_align)?;
        let sc_up = super::load::ffn_plane(exec, map, "self_cond_up.weight", true, ffn_align)?;
        let sc_down = super::load::ffn_plane(exec, map, "self_cond_down.weight", false, ffn_align)?;
        let n_ff = sc_gate.dims()[1];
        if sc_gate.dims()[0] != n_embd
            || sc_up.dims() != sc_gate.dims()
            || sc_down.dims() != [n_ff, n_embd]
        {
            return Err(LoadError::Tensor(
                "self_cond_*".into(),
                format!(
                    "shapes gate {:?} up {:?} down {:?} do not form an [{n_embd} -> ff -> {n_embd}] MLP",
                    sc_gate.dims(),
                    sc_up.dims(),
                    sc_down.dims()
                ),
            ));
        }
        // E^T: vocab * embd bf16 - one kernel straight off the embedding's
        // resident form (Q8_0 rows, or the repacked k-quant plane through
        // the lanes' own window unpack)
        let mut bytes = exec.alloc_u8(n_vocab * n_embd * 2)?;
        match embd {
            EmbdTable::Raw(t) => {
                if t.ty != GgmlType::Q8_0 {
                    // a bf16-embedding file (UD mixes) would need its own
                    // transpose - not shipped for this arch, so say so
                    return Err(LoadError::Tensor(
                        "token_embd.weight".into(),
                        format!(
                            "{:?}: the diffusion lane transposes a Q8_0 or k-quant embedding",
                            t.ty
                        ),
                    ));
                }
                exec.q8_embed_transpose_bf16(t, &mut bytes, n_vocab, n_embd)?;
            }
            EmbdTable::Kq(w) => exec.kq_embed_transpose_bf16(w, &mut bytes, n_vocab, n_embd)?,
            // the loader keeps the raw rows until this lane is built
            EmbdTable::Q8r(_) => {
                return Err(LoadError::Tensor(
                    "token_embd.weight".into(),
                    "the diffusion lane transposes the raw Q8_0 rows".into(),
                ));
            }
        }
        let embd_t = QuantTensor {
            bytes,
            ty: GgmlType::Bf16,
            dims: vec![n_vocab, n_embd],
        };
        let ones = exec
            .stream
            .clone_htod(&vec![1.0f32; n_embd])
            .map_err(crate::gpu::from_driver)?;
        let bytes = (n_vocab * n_embd * 2) as u64
            + sc_gate.bytes()
            + sc_up.bytes()
            + sc_down.bytes()
            + (2 * n_embd * 4) as u64;
        Ok(Self {
            sc_pre_norm,
            sc_gate,
            sc_up,
            sc_down,
            embd_t,
            ones,
            canvas_len,
            n_ff,
            cfg: DiffusionConfig::default(),
            bytes,
        })
    }
}

/// One canvas's device state: the ids the next step reads, the sampler's
/// per-position outputs, the argmax history the stopping rule compares
/// against, and the `[w][vocab]` plane the head writes and the sampler turns
/// into the probs the NEXT step's self-conditioning reads. `host` mirrors
/// `canvas` so pinning and emission never wait on a readback of their own.
pub struct CanvasState {
    pub w: usize,
    pub canvas: CudaSlice<u32>,
    pub host: Vec<u32>,
    pub sampled: CudaSlice<u32>,
    pub argmax: CudaSlice<u32>,
    pub entropy: CudaSlice<f32>,
    pub inv_t: CudaSlice<f32>,
    pub hist: CudaSlice<u32>,
    pub status: CudaSlice<u32>,
    pub probs: CudaSlice<f32>,
    /// `probs` holds the previous step's normalized distribution - the
    /// self-conditioning input. False before the first step (the reference
    /// passes `None` and the signal is exactly zero).
    pub have_probs: bool,
    /// 0-based ascending step count (the reference's `cur_step` counts down)
    pub step: u32,
    /// the last step's argmax canvas, the block a converged canvas emits
    pub last_argmax: Vec<u32>,
    /// the last step's per-position entropy (the structured read's
    /// confidence signal comes from here)
    pub last_entropy: Vec<f32>,
}

impl GpuGemma4 {
    /// The diffusion lane, or the honest refusal for a next-token file.
    fn lane(&self) -> Result<&DiffusionLane, GpuError> {
        self.diffusion
            .as_ref()
            .ok_or_else(|| GpuError::Unsupported("this file is not a block-diffusion model".into()))
    }

    /// The lane's sampler constants (the file's authors' settings).
    pub fn diffusion_config(&self) -> Option<DiffusionConfig> {
        self.diffusion.as_ref().map(|l| l.cfg)
    }

    /// The canvas width the file declares (0 = not a diffusion model).
    pub fn canvas_len(&self) -> usize {
        self.hp.canvas_len
    }

    /// Allocate a canvas of `w` positions (1..=canvas_len; a structured read
    /// takes a narrower one). The argmax history starts at 0xFFFFFFFF, which
    /// no real id equals, so the first step is never "stable".
    pub fn canvas_new(&self, w: usize) -> Result<CanvasState, GpuError> {
        let lane = self.lane()?;
        if w == 0 || w > lane.canvas_len {
            return Err(GpuError::Unsupported(format!(
                "canvas of {w} positions: this file denoises 1..={}",
                lane.canvas_len
            )));
        }
        if w > self.pf_rows {
            return Err(GpuError::Unsupported(format!(
                "canvas of {w} positions exceeds the {}-row prefill scratch",
                self.pf_rows
            )));
        }
        let exec = &self.exec;
        let stab = lane.cfg.stability as usize;
        Ok(CanvasState {
            w,
            canvas: exec.alloc_u32(w)?,
            host: vec![0; w],
            sampled: exec.alloc_u32(w)?,
            argmax: exec.alloc_u32(w)?,
            entropy: exec.alloc(w)?,
            inv_t: exec.alloc(w)?,
            hist: exec.to_device_u32(&vec![u32::MAX; stab.max(1) * w])?,
            status: exec.alloc_u32(4)?,
            probs: exec.alloc(w * self.hp.n_vocab)?,
            have_probs: false,
            step: 0,
            last_argmax: vec![0; w],
            last_entropy: vec![0.0; w],
        })
    }

    /// Set the canvas ids for the next step (the random initial canvas, a
    /// seeded template, or a pin overwrite after `canvas_step`).
    pub fn canvas_seed(&self, st: &mut CanvasState, ids: &[u32]) -> Result<(), GpuError> {
        if ids.len() != st.w {
            return Err(GpuError::Driver(format!(
                "canvas_seed: {} ids for a {}-wide canvas",
                ids.len(),
                st.w
            )));
        }
        st.host.copy_from_slice(ids);
        self.exec
            .stream
            .memcpy_htod(ids, &mut st.canvas)
            .map_err(crate::gpu::from_driver)
    }

    /// A uniformly random canvas the way the reference draws one
    /// (`torch.randint(0, vocab)`), from the same Philox layout the
    /// re-noise draws use, so a seeded run is reproducible.
    pub fn random_canvas(&self, w: usize, seed: u64, offset: u32) -> Vec<u32> {
        let vocab = self.hp.n_vocab as u64;
        (0..w as u32)
            .map(|i| {
                let u = philox_uniform(seed, offset, i);
                ((u as f64 * vocab as f64) as u64).min(vocab - 1) as u32
            })
            .collect()
    }

    /// The canvas forward: `w` rows at `[base, base + w)` in `slot`'s KV,
    /// rope at their true positions, attention bounded at the block end
    /// (every row reads the prefix and the whole canvas), self-conditioned on
    /// the previous step's probs when there are any. Leaves the softcapped
    /// logits of every row in `st.probs` (`[w][vocab]`, still logits at this
    /// point). The canvas' K/V sit past the committed cursor and are
    /// overwritten by the next step or the commit - nothing persists.
    fn canvas_forward(
        &mut self,
        slot: usize,
        base: usize,
        st: &mut CanvasState,
    ) -> Result<(), GpuError> {
        let w = st.w;
        if base + w > self.max_ctx {
            return Err(GpuError::Driver(format!(
                "canvas at [{base}, {}) exceeds max_ctx {}",
                base + w,
                self.max_ctx
            )));
        }
        let exec = self.exec.clone();
        let (n_embd, eps, vocab) = (self.hp.n_embd, self.hp.eps, self.hp.n_vocab);
        let embd_scale = self.hp.embd_scale();
        // global-layer pool rows for the block end (pool mode; no-op dense)
        self.ensure_global_rows(&[slot as u32], &[(base + w - 1) as u32])?;
        {
            let lane = self
                .diffusion
                .as_ref()
                .ok_or_else(|| GpuError::Unsupported("not a diffusion model".into()))?;
            let sc = &mut self.scratch;
            // per-row staging: ids, rope positions, the block-end bound, slot
            let positions: Vec<u32> = (0..w).map(|i| (base + i) as u32).collect();
            let bounds = vec![(base + w - 1) as u32; w];
            let slots = vec![slot as u32; self.pf_rows];
            let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
                let mut v = dst
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver("canvas staging view".into()))?;
                exec.stream
                    .memcpy_htod(host, &mut v)
                    .map_err(crate::gpu::from_driver)
            };
            up(&st.host, &mut sc.pf_toks)?;
            up(&positions, &mut sc.pf_pos)?;
            up(&bounds, &mut sc.pf_attn_pos)?;
            up(&slots, &mut sc.pf_slots)?;
            // x = embed(ids) * sqrt(embd)
            EmbdTable::of(&self.token_embd, &self.head).gather(
                &exec,
                &sc.pf_toks,
                &mut sc.pf_x,
                n_embd,
                w,
                embd_scale,
            )?;
            // + self_cond(softmax(prev logits) @ E * sqrt(embd)), zero on the
            //   first step (the reference passes None -> zeros, and the
            //   gated MLP of a zero input is zero)
            if st.have_probs {
                exec.bf16_gemm(&lane.embd_t, None, &st.probs, &mut sc.pf_tmp, w)?;
                exec.scale(&mut sc.pf_tmp, embd_scale, w * n_embd)?;
                exec.rmsnorm_batch(
                    &sc.pf_tmp,
                    &lane.sc_pre_norm,
                    &mut sc.pf_normed,
                    n_embd,
                    eps,
                    w,
                )?;
                lane.sc_gate
                    .gemm(&exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_gate, w)?;
                lane.sc_up
                    .gemm(&exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_up, w)?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, w * lane.n_ff, GluAct::Gelu)?;
                lane.sc_down
                    .gemm(&exec, kq_rows!(sc), &sc.pf_gate, &mut sc.pf_proj, w)?;
                exec.scale_add(&mut sc.pf_x, &sc.pf_proj, 1.0, w * n_embd)?;
            }
            // the module's weightless post-norm, applied on every step
            exec.rmsnorm_batch_inplace(&mut sc.pf_x, &lane.ones, n_embd, eps, w)?;
        }
        let spans = super::forward::swa_spans(self.swa_span, &[(0, w)]);
        self.prefill_layers(w, &[(0, w)], &spans, 0)?;
        // the head over every canvas row, softcapped on device
        {
            let sc = &mut self.scratch;
            exec.rmsnorm_batch(
                &sc.pf_x,
                &self.output_norm,
                &mut sc.pf_normed,
                n_embd,
                eps,
                w,
            )?;
            self.head
                .gemm(&exec, kq_rows!(sc), &sc.pf_normed, &mut st.probs, w)?;
            super::logit_epilogue_dev(
                &exec,
                &mut st.probs,
                w * vocab,
                self.hp.logit_scale,
                self.hp.final_softcap,
            )?;
            // restore the single-stream slot-0 staging (forward_prefill's rule)
            let zeros = vec![0u32; self.pf_rows];
            exec.stream
                .memcpy_htod(&zeros, &mut sc.pf_slots)
                .map_err(crate::gpu::from_driver)?;
        }
        Ok(())
    }

    /// One denoising step of `st` at `[base, base + w)` in `slot`: the
    /// forward, the row sampler at temperature `temp` (`<= 0` = the GREEDY
    /// draw at the schedule's temperature for this step - the argmax is
    /// taken, but the entropies the accept step reads stay the schedule's;
    /// a literal zero temperature would call every position certain and
    /// accept the noise canvas whole on step one), the entropy-bounded
    /// accept and re-noise, the stability check. On
    /// return `st.host` is the next step's input (pins go on top of it
    /// via `canvas_seed`), `st.last_argmax` the block a converged canvas
    /// emits, and `st.probs` the self-conditioning input. `seed` is the
    /// request's; the Philox offsets fold in the step only (never the slot,
    /// so a seeded request reproduces wherever it was placed).
    pub fn canvas_step(
        &mut self,
        slot: usize,
        base: usize,
        st: &mut CanvasState,
        temp: f32,
        seed: u64,
    ) -> Result<CanvasStatus, GpuError> {
        let cfg = self.lane()?.cfg;
        self.canvas_forward(slot, base, st)?;
        let exec = self.exec.clone();
        let (w, vocab) = (st.w, self.hp.n_vocab);
        let inv_t = Self::canvas_inv_t(&cfg, st.step, temp);
        exec.stream
            .memcpy_htod(&vec![inv_t; w], &mut st.inv_t)
            .map_err(crate::gpu::from_driver)?;
        // keyed by (seed, step), not the slot - the batched tick's rule too
        let offset = (st.step & 0x7FFF) << 1;
        exec.canvas_sample(
            &mut st.probs,
            &st.inv_t,
            seed,
            offset,
            &mut st.sampled,
            &mut st.argmax,
            &mut st.entropy,
            w,
            vocab,
        )?;
        exec.canvas_accept(
            &st.entropy,
            &st.sampled,
            &st.argmax,
            &mut st.canvas,
            &mut st.hist,
            &mut st.status,
            w,
            vocab,
            cfg.stability as usize,
            st.step,
            cfg.entropy_bound,
            cfg.confidence,
            seed,
            offset + 1,
        )?;
        let words = exec.to_host_u32(&st.status)?;
        st.last_argmax = exec.to_host_u32(&st.argmax)?;
        st.last_entropy = exec.to_host(&st.entropy)?;
        st.host = exec.to_host_u32(&st.canvas)?;
        st.have_probs = true;
        st.step += 1;
        Ok(CanvasStatus::from_words([
            words[0], words[1], words[2], words[3],
        ]))
    }

    /// The sampler's inverse temperature for a step: `temp > 0` samples at
    /// it; `temp <= 0` is the greedy draw, encoded as the NEGATIVE inverse
    /// of the schedule's temperature for the step (the kernel's convention:
    /// argmax sample, entropy and probs at |inv_t|).
    fn canvas_inv_t(cfg: &DiffusionConfig, step: u32, temp: f32) -> f32 {
        if temp > 0.0 {
            1.0 / temp
        } else {
            -1.0 / cfg.temperature(step)
        }
    }

    /// The structured read: one forward of a seeded canvas, the probs at
    /// temperature 1, and each position's probability of every id in
    /// `label_ids` - `[w][label_ids.len()]`, row-major. Nothing is
    /// accepted, re-noised or committed; `st` is left holding the probs.
    pub fn canvas_read(
        &mut self,
        slot: usize,
        base: usize,
        st: &mut CanvasState,
        label_ids: &[u32],
    ) -> Result<Vec<f32>, GpuError> {
        self.canvas_forward(slot, base, st)?;
        let exec = self.exec.clone();
        let (w, vocab, k) = (st.w, self.hp.n_vocab, label_ids.len());
        exec.stream
            .memcpy_htod(&vec![1.0f32; w], &mut st.inv_t)
            .map_err(crate::gpu::from_driver)?;
        // temperature 1, no draw needed: inv_t 1 still runs the Gumbel pass,
        // which is cheap next to the forward; the argmax/entropy come along
        exec.canvas_sample(
            &mut st.probs,
            &st.inv_t,
            0,
            0,
            &mut st.sampled,
            &mut st.argmax,
            &mut st.entropy,
            w,
            vocab,
        )?;
        let ids = exec.to_device_u32(label_ids)?;
        let mut out = exec.alloc(w * k.max(1))?;
        if k > 0 {
            exec.gather_cols(&st.probs, &ids, &mut out, w, vocab, k)?;
        }
        st.last_argmax = exec.to_host_u32(&st.argmax)?;
        st.last_entropy = exec.to_host(&st.entropy)?;
        st.have_probs = true;
        st.step += 1;
        if k == 0 {
            return Ok(Vec::new());
        }
        exec.to_host(&out)
    }

    /// One whole block for the serial service (`Generator::canvas_block`):
    /// a random canvas of `w` at `[base, base + w)` in slot 0, denoised at
    /// `temperature` (`None` = the authors' schedule, `Some(0.0)` =
    /// deterministic) until the stopping rule fires or the step budget is
    /// spent; returns the argmax canvas and the steps it took. Nothing is
    /// committed - the caller decides once it has looked for an EOS.
    pub(crate) fn canvas_block_impl(
        &mut self,
        base: usize,
        w: usize,
        temperature: Option<f32>,
        seed: u64,
    ) -> Result<(Vec<u32>, u32), GpuError> {
        let cfg = self.lane()?.cfg;
        let mut st = self.canvas_new(w)?;
        // the block's position folded into the offset so each block's random
        // canvas and re-noise draws differ from the last block's
        let block_offset = (base as u32).wrapping_mul(2_654_435_761);
        let init = self.random_canvas(w, seed, block_offset);
        self.canvas_seed(&mut st, &init)?;
        let mut steps = 0u32;
        while steps < cfg.max_steps {
            let temp = temperature.unwrap_or_else(|| cfg.temperature(steps));
            let status =
                self.canvas_step(0, base, &mut st, temp, seed ^ u64::from(block_offset))?;
            steps += 1;
            if status.converged {
                break;
            }
        }
        Ok((st.last_argmax.clone(), steps))
    }

    /// The structured read for the serial service (`Generator::canvas_read`):
    /// the seeded ids in slot 0 at `[base, base + canvas.len())`.
    pub(crate) fn canvas_read_impl(
        &mut self,
        base: usize,
        canvas: &[u32],
        label_ids: &[u32],
    ) -> Result<crate::generator::CanvasReadOut, GpuError> {
        let mut st = self.canvas_new(canvas.len())?;
        self.canvas_seed(&mut st, canvas)?;
        let probs = self.canvas_read(0, base, &mut st, label_ids)?;
        Ok(crate::generator::CanvasReadOut {
            probs,
            entropy: st.last_entropy,
            argmax: st.last_argmax,
        })
    }

    /// The multi-step structured read for the serial service
    /// (`Generator::canvas_read_steps`): `steps - 1` ordinary denoising
    /// steps of the seeded canvas in slot 0, the `pinned` positions put back
    /// after each, then the read's forward at temperature 1 - which sees the
    /// last step's probs through the self-conditioning MLP, exactly as the
    /// batched loop's final tick does.
    pub(crate) fn canvas_read_steps_impl(
        &mut self,
        base: usize,
        canvas: &[u32],
        label_ids: &[u32],
        steps: u32,
        pinned: &[u32],
        seed: u64,
    ) -> Result<crate::generator::CanvasReadOut, GpuError> {
        let cfg = self.lane()?.cfg;
        let mut st = self.canvas_new(canvas.len())?;
        self.canvas_seed(&mut st, canvas)?;
        for _ in 1..steps.max(1) {
            let temp = cfg.temperature(st.step);
            self.canvas_step(0, base, &mut st, temp, seed)?;
            let mut next = st.host.clone();
            for &p in pinned {
                let p = p as usize;
                if p >= next.len() {
                    return Err(GpuError::Driver(format!(
                        "canvas_read_steps: pinned position {p} is past the {}-wide canvas",
                        next.len()
                    )));
                }
                next[p] = canvas[p];
            }
            self.canvas_seed(&mut st, &next)?;
        }
        let probs = self.canvas_read(0, base, &mut st, label_ids)?;
        Ok(crate::generator::CanvasReadOut {
            probs,
            entropy: st.last_entropy,
            argmax: st.last_argmax,
        })
    }

    // ── the batched tick: several canvases, several slots, ONE forward ──

    /// Open a canvas for the batched loop and hand back its handle (the
    /// first released handle is reused, so the table stays as wide as the
    /// live cohort).
    pub(crate) fn canvas_open_impl(&mut self, w: usize) -> Result<usize, GpuError> {
        let st = self.canvas_new(w)?;
        if let Some(h) = self.canvases.iter().position(Option::is_none) {
            self.canvases[h] = Some(st);
            return Ok(h);
        }
        self.canvases.push(Some(st));
        Ok(self.canvases.len() - 1)
    }

    fn canvas_at(&self, h: usize) -> Result<&CanvasState, GpuError> {
        self.canvases
            .get(h)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::Driver(format!("canvas handle {h} is not open")))
    }

    pub(crate) fn canvas_set_impl(&mut self, h: usize, ids: &[u32]) -> Result<(), GpuError> {
        let mut st = self
            .canvases
            .get_mut(h)
            .and_then(Option::take)
            .ok_or_else(|| GpuError::Driver(format!("canvas handle {h} is not open")))?;
        let r = self.canvas_seed(&mut st, ids);
        self.canvases[h] = Some(st);
        r
    }

    /// Hold `positions` of canvas `h` at `ids` for its next step
    /// (`Generator::canvas_pin`): the last accepting tick left the next
    /// input in `st.host` - accepted draws where the canvas settled, fresh
    /// noise elsewhere - and the pinned positions go back on top of it, so a
    /// multi-step read denoises its answer slots against a template that
    /// never moves.
    pub(crate) fn canvas_pin_impl(
        &mut self,
        h: usize,
        positions: &[u32],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        if positions.len() != ids.len() {
            return Err(GpuError::Driver(format!(
                "canvas_pin: {} positions for {} ids",
                positions.len(),
                ids.len()
            )));
        }
        let mut st = self
            .canvases
            .get_mut(h)
            .and_then(Option::take)
            .ok_or_else(|| GpuError::Driver(format!("canvas handle {h} is not open")))?;
        let mut next = st.host.clone();
        let mut bad = None;
        for (&p, &id) in positions.iter().zip(ids) {
            match next.get_mut(p as usize) {
                Some(slot) => *slot = id,
                None => bad = Some(p),
            }
        }
        let r = match bad {
            Some(p) => Err(GpuError::Driver(format!(
                "canvas_pin: position {p} is past the {}-wide canvas",
                st.w
            ))),
            None => self.canvas_seed(&mut st, &next),
        };
        self.canvases[h] = Some(st);
        r
    }

    /// One denoising step of every canvas in `ticks`, in one forward: the
    /// canvases' rows are concatenated into one prefill-shaped pass (per-row
    /// slot, true position and block-end bound - the batched chunk prefill's
    /// own staging), the self-conditioning input is assembled per canvas
    /// from its own previous probs (zero rows for a first step - the gated
    /// MLP of zero is zero, so one MLP launch over all rows is exact), the
    /// layers run once, and the head, sampler and accept step run per canvas
    /// over that canvas' rows. Every canvas' K/V land past its slot's
    /// committed cursor - scratch the next tick or the commit overwrites.
    pub(crate) fn canvas_tick_impl(
        &mut self,
        ticks: &[crate::generator::CanvasTickReq],
    ) -> Result<Vec<CanvasStatus>, GpuError> {
        let cfg = self.lane()?.cfg;
        if ticks.is_empty() {
            return Ok(Vec::new());
        }
        // geometry: each canvas' row offset in the pass, and the pass width
        let mut offs = Vec::with_capacity(ticks.len());
        let mut r = 0usize;
        for t in ticks {
            let w = self.canvas_at(t.handle)?.w;
            if t.base + w > self.max_ctx {
                return Err(GpuError::Driver(format!(
                    "canvas at [{}, {}) exceeds max_ctx {}",
                    t.base,
                    t.base + w,
                    self.max_ctx
                )));
            }
            offs.push(r);
            r += w;
        }
        if r > self.pf_rows {
            return Err(GpuError::Unsupported(format!(
                "{r} canvas rows in one tick exceed the {}-row prefill scratch",
                self.pf_rows
            )));
        }
        let exec = self.exec.clone();
        let (n_embd, eps, vocab) = (self.hp.n_embd, self.hp.eps, self.hp.n_vocab);
        let embd_scale = self.hp.embd_scale();
        let slots_v: Vec<u32> = ticks.iter().map(|t| t.slot as u32).collect();
        let ends: Vec<u32> = ticks
            .iter()
            .map(|t| (t.base + self.canvases[t.handle].as_ref().map_or(0, |c| c.w) - 1) as u32)
            .collect();
        // global-layer pool rows for every block end (pool mode; no-op dense)
        self.ensure_global_rows(&slots_v, &ends)?;
        let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
            let mut v = dst
                .try_slice_mut(0..host.len())
                .ok_or_else(|| GpuError::Driver("canvas staging view".into()))?;
            exec.stream
                .memcpy_htod(host, &mut v)
                .map_err(crate::gpu::from_driver)
        };
        {
            let lane = self
                .diffusion
                .as_ref()
                .ok_or_else(|| GpuError::Unsupported("not a diffusion model".into()))?;
            let sc = &mut self.scratch;
            let mut toks = Vec::with_capacity(r);
            let mut positions = Vec::with_capacity(r);
            let mut bounds = Vec::with_capacity(r);
            let mut slot_rows = Vec::with_capacity(r);
            for t in ticks {
                let st = self.canvases[t.handle].as_ref().expect("checked above");
                toks.extend_from_slice(&st.host);
                positions.extend((0..st.w).map(|i| (t.base + i) as u32));
                bounds.extend(std::iter::repeat_n((t.base + st.w - 1) as u32, st.w));
                slot_rows.extend(std::iter::repeat_n(t.slot as u32, st.w));
            }
            up(&toks, &mut sc.pf_toks)?;
            up(&positions, &mut sc.pf_pos)?;
            up(&bounds, &mut sc.pf_attn_pos)?;
            up(&slot_rows, &mut sc.pf_slots)?;
            EmbdTable::of(&self.token_embd, &self.head).gather(
                &exec,
                &sc.pf_toks,
                &mut sc.pf_x,
                n_embd,
                r,
                embd_scale,
            )?;
            // self-conditioning, per canvas from its own probs into its rows
            // of pf_tmp; a canvas on its first step contributes zero rows
            {
                let mut z = sc
                    .pf_tmp
                    .try_slice_mut(0..r * n_embd)
                    .ok_or_else(|| GpuError::Driver("canvas self-cond view".into()))?;
                exec.stream
                    .memset_zeros(&mut z)
                    .map_err(crate::gpu::from_driver)?;
            }
            let mut any = false;
            for (t, &off) in ticks.iter().zip(&offs) {
                let st = self.canvases[t.handle].as_ref().expect("checked above");
                if !st.have_probs {
                    continue;
                }
                any = true;
                exec.bf16_gemm(&lane.embd_t, None, &st.probs, &mut sc.pf_proj, st.w)?;
                exec.copy_region(&sc.pf_proj, 0, &mut sc.pf_tmp, off * n_embd, st.w * n_embd)?;
            }
            if any {
                exec.scale(&mut sc.pf_tmp, embd_scale, r * n_embd)?;
                exec.rmsnorm_batch(
                    &sc.pf_tmp,
                    &lane.sc_pre_norm,
                    &mut sc.pf_normed,
                    n_embd,
                    eps,
                    r,
                )?;
                lane.sc_gate
                    .gemm(&exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_gate, r)?;
                lane.sc_up
                    .gemm(&exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_up, r)?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * lane.n_ff, GluAct::Gelu)?;
                lane.sc_down
                    .gemm(&exec, kq_rows!(sc), &sc.pf_gate, &mut sc.pf_proj, r)?;
                exec.scale_add(&mut sc.pf_x, &sc.pf_proj, 1.0, r * n_embd)?;
            }
            exec.rmsnorm_batch_inplace(&mut sc.pf_x, &lane.ones, n_embd, eps, r)?;
        }
        let runs: Vec<(usize, usize)> = ticks
            .iter()
            .zip(&offs)
            .map(|(t, &off)| (off, self.canvases[t.handle].as_ref().map_or(0, |c| c.w)))
            .collect();
        let spans = super::forward::swa_spans(self.swa_span, &runs);
        self.prefill_layers(r, &runs, &spans, 0)?;
        // the head, sampler and accept step per canvas over its own rows
        let mut out = Vec::with_capacity(ticks.len());
        {
            let sc = &mut self.scratch;
            exec.rmsnorm_batch(
                &sc.pf_x,
                &self.output_norm,
                &mut sc.pf_normed,
                n_embd,
                eps,
                r,
            )?;
            for (t, &off) in ticks.iter().zip(&offs) {
                let st = self.canvases[t.handle].as_mut().expect("checked above");
                let w = st.w;
                // the head reads rows from the front of its input, so the
                // canvas' normed rows move to the front of pf_tmp (free since
                // the self-cond pass) - w x embd, nothing next to the head
                exec.copy_region(&sc.pf_normed, off * n_embd, &mut sc.pf_tmp, 0, w * n_embd)?;
                self.head
                    .gemm(&exec, kq_rows!(sc), &sc.pf_tmp, &mut st.probs, w)?;
                super::logit_epilogue_dev(
                    &exec,
                    &mut st.probs,
                    w * vocab,
                    self.hp.logit_scale,
                    self.hp.final_softcap,
                )?;
                let temp = t.temperature.unwrap_or_else(|| cfg.temperature(st.step));
                let inv_t = Self::canvas_inv_t(&cfg, st.step, temp);
                exec.stream
                    .memcpy_htod(&vec![inv_t; w], &mut st.inv_t)
                    .map_err(crate::gpu::from_driver)?;
                // the draw is keyed by (request seed, step) only - never the
                // slot - so a seeded request reproduces wherever the
                // scheduler placed it (the serial loop's slot-0 stream)
                let offset = (st.step & 0x7FFF) << 1;
                exec.canvas_sample(
                    &mut st.probs,
                    &st.inv_t,
                    t.seed,
                    offset,
                    &mut st.sampled,
                    &mut st.argmax,
                    &mut st.entropy,
                    w,
                    vocab,
                )?;
                let status = if t.accept {
                    exec.canvas_accept(
                        &st.entropy,
                        &st.sampled,
                        &st.argmax,
                        &mut st.canvas,
                        &mut st.hist,
                        &mut st.status,
                        w,
                        vocab,
                        cfg.stability as usize,
                        st.step,
                        cfg.entropy_bound,
                        cfg.confidence,
                        t.seed,
                        offset + 1,
                    )?;
                    let words = exec.to_host_u32(&st.status)?;
                    st.host = exec.to_host_u32(&st.canvas)?;
                    CanvasStatus::from_words([words[0], words[1], words[2], words[3]])
                } else {
                    CanvasStatus {
                        converged: false,
                        n_accepted: 0,
                        mean_entropy: f32::NAN,
                        stable: false,
                    }
                };
                st.last_argmax = exec.to_host_u32(&st.argmax)?;
                st.last_entropy = exec.to_host(&st.entropy)?;
                st.have_probs = true;
                st.step += 1;
                out.push(status);
            }
            // the slot staging back to zeros for the single-stream callers
            up(&vec![0u32; r], &mut sc.pf_slots)?;
        }
        Ok(out)
    }

    /// The last tick's result for canvas `h` (`Generator::canvas_result`):
    /// label probabilities gathered from the normalized plane, plus the
    /// entropies and the argmax canvas the tick read back.
    pub(crate) fn canvas_result_impl(
        &self,
        h: usize,
        label_ids: &[u32],
    ) -> Result<crate::generator::CanvasReadOut, GpuError> {
        let st = self.canvas_at(h)?;
        let (w, vocab, k) = (st.w, self.hp.n_vocab, label_ids.len());
        let probs = if k == 0 {
            Vec::new()
        } else {
            let exec = &self.exec;
            let ids = exec.to_device_u32(label_ids)?;
            let mut out = exec.alloc(w * k)?;
            exec.gather_cols(&st.probs, &ids, &mut out, w, vocab, k)?;
            exec.to_host(&out)?
        };
        Ok(crate::generator::CanvasReadOut {
            probs,
            entropy: st.last_entropy.clone(),
            argmax: st.last_argmax.clone(),
        })
    }

    /// Commit a converged block: the ids re-run CAUSALLY at `[base, base +
    /// ids.len())` in `slot`, writing the real K/V the next canvas reads.
    /// The single-stream cursor advances with it (batched slots are
    /// position-keyed by the service).
    pub fn canvas_commit_at(
        &mut self,
        slot: usize,
        base: usize,
        ids: &[u32],
    ) -> Result<(), GpuError> {
        if ids.is_empty() {
            return Ok(());
        }
        if ids.len() > self.pf_rows {
            return Err(GpuError::Driver(format!(
                "commit of {} rows exceeds the {}-row prefill scratch",
                ids.len(),
                self.pf_rows
            )));
        }
        self.ensure_global_rows(&[slot as u32], &[(base + ids.len() - 1) as u32])?;
        let fill = vec![slot as u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&fill, &mut self.scratch.pf_slots)
            .map_err(crate::gpu::from_driver)?;
        self.prefill_chunk(ids, base)?;
        let zeros = vec![0u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&zeros, &mut self.scratch.pf_slots)
            .map_err(crate::gpu::from_driver)?;
        if slot == 0 && self.pos == base {
            self.pos = base + ids.len();
        }
        Ok(())
    }
}

/// Host twin of the pack's `pd_canvas_uniform`: Philox4x32-10 with key =
/// seed (lo, hi), counter = {offset, 0, index, 0}, first word as a uniform
/// in (0, 1). Used for the initial canvas so it can be drawn without a
/// launch and matches what a device draw at the same coordinates gives.
fn philox_uniform(seed: u64, offset: u32, index: u32) -> f32 {
    let (mut c0, mut c1, mut c2, mut c3) = (offset, 0u32, index, 0u32);
    let (mut k0, mut k1) = (seed as u32, (seed >> 32) as u32);
    let round = |c0: &mut u32, c1: &mut u32, c2: &mut u32, c3: &mut u32, k0: u32, k1: u32| {
        let p0 = (*c0 as u64) * 0xD251_1F53;
        let p1 = (*c2 as u64) * 0xCD9E_8D57;
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        let n0 = hi1 ^ *c1 ^ k0;
        let n2 = hi0 ^ *c3 ^ k1;
        *c0 = n0;
        *c1 = lo1;
        *c2 = n2;
        *c3 = lo0;
    };
    for _ in 0..9 {
        round(&mut c0, &mut c1, &mut c2, &mut c3, k0, k1);
        k0 = k0.wrapping_add(0x9E37_79B9);
        k1 = k1.wrapping_add(0xBB67_AE85);
    }
    round(&mut c0, &mut c1, &mut c2, &mut c3, k0, k1);
    let inv32 = 2.328_306_4e-10f32;
    c0 as f32 * inv32 + inv32 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_schedule_runs_from_t_max_down_to_just_above_t_min() {
        let c = DiffusionConfig::default();
        assert!((c.temperature(0) - 0.8).abs() < 1e-6);
        // cur_step = 1 on the last iteration: t_min + (t_max - t_min) / 48
        assert!((c.temperature(47) - (0.4 + 0.4 / 48.0)).abs() < 1e-6);
        // past the budget the schedule pins to that last value
        assert!((c.temperature(99) - (0.4 + 0.4 / 48.0)).abs() < 1e-6);
    }

    #[test]
    fn philox_uniform_is_in_range_and_keyed() {
        let a = philox_uniform(42, 0, 0);
        let b = philox_uniform(42, 0, 1);
        let c = philox_uniform(43, 0, 0);
        for u in [a, b, c] {
            assert!(u > 0.0 && u < 1.0, "{u}");
        }
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
