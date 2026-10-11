//! The reasoning stop mask (pack slot 850) and the lane a family carries to
//! run it.
//!
//! The service hands a backend that answers `supports_stop_mask` one
//! `Option<StopMask>` per row of its next device-sampled call (the
//! sampler's `sampler/reasoning.rs` policy: inside an open reasoning block
//! the stop tokens are masked before the draw and the row's stop pressure is
//! read). The family keeps them in a [`StopMaskLane`], launches the mask on
//! its logits rows right before the sampling kernels, and the service takes
//! the pressure flags back after the call - one small read on the stream the
//! ids came back on.

use super::error::*;
use super::*;
use crate::sampler::StopMask;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// True when the pack carries the reasoning stop mask (slot 850).
    pub fn has_stop_mask_rows(&self) -> bool {
        self.kernels.stop_mask_rows.is_some()
    }

    /// The reasoning stop mask, in place on `logits` `[rows, n]` (from element
    /// `off`) before the sampling kernels: rows whose `par` (8 u32 a row:
    /// active, inv_t bits, n_stops, 4 stop ids, pad) is active get their stop
    /// logits set to -inf and `flags[row]` = stop pressure; inactive rows are
    /// untouched and flag 0. `at`: the first row's index into the `par` and
    /// `flags` planes.
    #[allow(clippy::too_many_arguments)]
    pub fn stop_mask_rows(
        &self,
        logits: &mut CudaSlice<f32>,
        off: usize,
        par: &CudaSlice<u32>,
        at: usize,
        flags: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .stop_mask_rows
            .ok_or(GpuError::MissingOp("stop_mask_rows"))?;
        let (lp, _g1) = logits.device_ptr_mut(&self.stream);
        let lp = lp + (off * std::mem::size_of::<f32>()) as u64;
        let (pp, _g2) = par.device_ptr(&self.stream);
        let pp = pp + (at * 8 * std::mem::size_of::<u32>()) as u64;
        let (fp, _g3) = flags.device_ptr_mut(&self.stream);
        let fp = fp + (at * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                lp as *mut _,
                pp as *const _,
                fp as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// One family's stop-mask state: the masks the service set for the next
/// device-sampled call, the kernel's row/flag planes, and what the last
/// launch covered.
#[derive(Default)]
pub struct StopMaskLane {
    /// the call's masks, in its row index space (its plans)
    masks: Vec<Option<StopMask>>,
    par: Option<CudaSlice<u32>>,
    flags: Option<CudaSlice<u32>>,
    /// rows the planes hold
    cap: usize,
    /// the call index of each logits row the last launch masked
    launched: Vec<usize>,
    /// the flags read back, in the call's row index space
    pressure: Vec<bool>,
}

impl StopMaskLane {
    /// The service's masks for the next call (`Generator::set_stop_masks`).
    pub fn set(&mut self, masks: Vec<Option<StopMask>>) {
        self.masks = masks;
        self.launched.clear();
        self.pressure.clear();
    }

    /// Mask the call's rows: logits row `r` (from element `off` of `logits`,
    /// `n` wide) is call row `rows[r]` (`usize::MAX`: a row no mask can index,
    /// a verify pad). Launches nothing - false - when no row carries a mask.
    /// Call it on the stream the sampling kernels run on, right before them;
    /// a call that samples in pieces (a split tick) launches once a piece.
    pub fn launch(
        &mut self,
        exec: &GpuExecutor,
        logits: &mut CudaSlice<f32>,
        off: usize,
        n: usize,
        rows: &[usize],
    ) -> Result<bool, GpuError> {
        let mask = |i: usize| self.masks.get(i).and_then(Option::as_ref);
        if !rows.iter().any(|&i| mask(i).is_some()) {
            return Ok(false);
        }
        let words: Vec<u32> = rows
            .iter()
            .flat_map(|&i| StopMask::words(mask(i)))
            .collect();
        let (r, at) = (rows.len(), self.launched.len());
        if self.cap < at + r {
            // pieces launched so far keep their flags: read them first
            self.read(exec)?;
            self.cap = r.max(self.masks.len()).next_power_of_two().max(64);
            self.par = Some(exec.alloc_u32(self.cap * 8)?);
            self.flags = Some(exec.alloc_u32(self.cap)?);
        }
        let at = self.launched.len();
        let (par, flags) = (
            self.par.as_mut().expect("par"),
            self.flags.as_mut().expect("flags"),
        );
        let mut v = par
            .try_slice_mut(at * 8..(at + r) * 8)
            .ok_or_else(|| GpuError::Driver("stop mask par slice".into()))?;
        exec.stream.memcpy_htod(&words, &mut v).map_err(drv)?;
        exec.stop_mask_rows(logits, off, par, at, flags, r, n)?;
        self.launched.extend_from_slice(rows);
        // engagement witness (bisect-trap law): once per process
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            eprintln!("[stop-mask] engaged: {r} rows (reasoning stops masked on device)")
        });
        Ok(true)
    }

    /// Read the launched rows' flags back (after the call's own ids
    /// readback, same stream). A verify round reads them before its accept
    /// walk, for [`Self::cut`]; otherwise [`Self::take`] does.
    pub fn read(&mut self, exec: &GpuExecutor) -> Result<(), GpuError> {
        let rows = std::mem::take(&mut self.launched);
        if rows.is_empty() {
            return Ok(());
        }
        let flags = self.flags.as_ref().expect("flags");
        let v = flags
            .try_slice(0..rows.len())
            .ok_or_else(|| GpuError::Driver("stop mask flags slice".into()))?;
        let f = exec.stream.clone_dtoh(&v).map_err(drv)?;
        self.pressure.resize(self.masks.len(), false);
        for (j, &i) in rows.iter().enumerate() {
            if let Some(p) = self.pressure.get_mut(i) {
                *p = f[j] != 0;
            }
        }
        Ok(())
    }

    /// The last row a verify run (call rows `off..off + picks.len()`) may
    /// accept under its rows' masks - `sampler::run_cut`, after [`Self::read`].
    pub fn cut(&self, off: usize, picks: &[u32]) -> usize {
        let end = off + picks.len();
        match self.masks.get(off..end) {
            Some(m) => {
                crate::sampler::run_cut(m, picks, self.pressure.get(off..end).unwrap_or(&[]))
            }
            None => picks.len().saturating_sub(1),
        }
    }

    /// The last call's pressure flags in its row index space (empty when no
    /// mask rode it), for `Generator::take_stop_pressure`. Clears the masks
    /// too, so a call that launched none cannot leave them for the next.
    pub fn take(&mut self, exec: &GpuExecutor) -> Vec<bool> {
        if let Err(e) = self.read(exec) {
            tracing::warn!("stop mask: the pressure flags could not be read back: {e}");
        }
        self.masks.clear();
        std::mem::take(&mut self.pressure)
    }
}
