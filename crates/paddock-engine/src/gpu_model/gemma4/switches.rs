//! Lane switches and tuning floors the batched/chunked walks read: each a
//! once-read development variable (compiled out of shipped builds) with
//! the measurement behind its default. Split out of batch.rs, which
//! re-exports them under their old paths.

use super::forward::PF_ROWS;

// fp4 weight-class lane switch: set by load.rs only when the planes really
// were converted (fp4 ladder present + both fusions live) - see load.rs
pub(crate) fn fp4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("PADDOCK_G4_FP4_ACTIVE").is_some())
}

// fused gu-epilogue (geglu+quant in the GEMM) kill switch - the interleave
// itself is killed separately at load (PADDOCK_G4_NO_GUIL); this one keeps
// the interleaved plane but falls back to the 2-launch geglu2i chain
pub(crate) fn gu_fuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_GUFUSE").is_none())
}

// fused rmsnorm->e4m3 kill switch (v11 norms rung)
pub(crate) fn nqfuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_NQFUSE").is_none())
}

// P47.L3: width cap for the row-fused norm+quant deferral.
// The old r<=64 bound was a decode-band election; the 65..128 verify
// band paid the unfused 3-kernel chain (rmsnorm_add_scale +
// rmsnorm_batch + row1p, 37.6us of spans/boundary) where one addnorm
// does the identical math - width-conditioned A/B: -0.3ms/tick at
// rows=128, comm/slot unmoved across 10 reps. The producer
// (next_fuses) and consumer (nqf_row_attn) must read the same cap or a
// deferred post strands. Truthy gate: PADDOCK_G4_NQF_WIDE=0 reverts.
pub(crate) fn nqf_wide_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match paddock_models::dev_var!("PADDOCK_G4_NQF_WIDE") {
        Ok(v) if v == "0" => 64,
        _ => 128,
    })
}

// f8t16: bf16 wo stream on the F8T chunk walk. The c16 machinery gates on
// f8w planes the unified loader doesn't build, so the f8t chunk route runs
// f32 glue, which is where the prefill band goes. The loader publishes
// PADDOCK_TC5R_O16_DIM/_IN so the pack's
// tc5r O16 arm fires on exactly the wo plane at batch >= 129; the wo
// consumers here read bf16 through the shipped p16 twins. Probe gate:
// PADDOCK_G4_F8T16=1 (default off until the battery).
pub(crate) fn f8t16_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_F8T16").is_some())
}

// the loader-published wo in_dim the pack's O16 arm keys on - reading the
// same env keeps engine and pack elections agreeing by construction
// (per-layer hd differs: sliding vs global attention).
pub(crate) fn f8t16_wo_in() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_TC5R_O16_IN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// P50: the verify tick's last f32-D GEMM plane - qkv - onto the
// b16-D arm, read straight by the packed-bf16 nra3 twin (slot 420). Its own
// truthy gate so the plane prices alone in the A/B; same r>=65 spec-only
// floor as VB16.
pub(crate) fn vb16q_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::envset::env_on("PADDOCK_G4_VB16Q"))
}

// P53 (slot 423): fin-e4 attention - SWA verify attention
// (n_kv==1: the fin CTA owns whole output rows) writes the wo-in e4m3
// rows + per-row scales in-kernel, bit-identical to fin followed by
// quantize_e4m3_row, and the standalone row-quant launch disappears.
// Truthy gate (=0 reverts once defaulted); scheduling-only otherwise.
pub(crate) fn fae4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_on!("PADDOCK_G4_FAE4"))
}

/// P54: fin-e4s - fin stores the wo input as e4m3 at STATIC
/// scale 1.0 (ones xrs), killing the standalone wo-in row quantize.
/// Numerics class change (the per-row scale only moves the clip/denorm
/// cliffs), so opt-in and comm/PPL-gated.
pub(crate) fn fae4s_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_on!("PADDOCK_G4_FAE4S"))
}

// Chunk-band 16-bit streams kill switch: o16 GEMM epilogues +
// bf16-in glue twins on the prefill chunk walk. Class change (bf16-rounded
// intermediates, the rival's own stream class) - acceptance-gated, not
// bit-parity; the kill restores the f32 streams exactly.
/// Attention streams kill (PADDOCK_G4_NO_ATTN16): absent = the
/// f16 pf_qn/pf_attn planes ride every eligible mixed/prefill pass.
pub(crate) fn attn16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_ATTN16").is_none())
}

pub(crate) fn chunk16_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_CHUNK16").is_none())
}

// spec-verify in-kernel finalize kill switch (door 3): FA at one split
// writing pf_attn directly, combine skipped. Chunk floor guards the dc4
// regression (few-chunk grids starve the die; splits still pay there).
pub(super) fn spec_fin_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_SPEC_FIN").is_none())
}

// sm_120 wide-band split election - FIN is the default. The older sp2/sp4
// election was measured on the pre-krs F8/SB route; with the krs-VR arm fin
// beats splits+combine at both classes on the bench (SWA 207.6 vs 224.4,
// GLB 307.8 vs 324.5) and never loses in a serve. The combine mass the
// splits carried was largely PDL-overlapped, so the win is small but
// uniform, and the round sheds 72 combine launches. PADDOCK_G4_SPEC_SP=1
// restores the split election for A/B; the legacy NO_SPEC_SP kill is
// subsumed (fin is the default it forced).
pub(super) fn spec_sp_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_G4_SPEC_SP").is_some()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_SPEC_SP").is_none()
    })
}

// Attention at depth: the fin flip above is a SWA verdict, not a global-layer
// one. Re-benched at fin, the split arm still wins on GLB (R2-sp4+combine
// 289.3 vs fin 308.2us - the fin GLB grid is 128 CTAs / 0.68 waves with the
// walk latency exposed per CTA), and the fin penalty GROWS with ctx depth
// while a split GLB stays flat. SWA keeps FIN, its own measured winner.
// Kill: PADDOCK_NO_GLB_SP4 restores fin on both arms.
pub(super) fn glb_sp4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GLB_SP4").is_none())
}

// Seen-once lazy capture: decode-tick graphs capture on the
// SECOND sight of a gkey, not the first - the fresh-capture path costs
// ~6.4ms (pre-capture sync serializes the stream, then record +
// instantiate) and wave transitions mint one-off row counts that never
// replay. Kill: PADDOCK_NO_LAZY_CAP restores capture-on-first-sight.
pub(super) fn lazy_cap_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_LAZY_CAP").is_none())
}

/// debug: PADDOCK_NO_DECODE_GRAPH=1 runs every decode tick
/// EAGER - no capture, no replay. The qkv f8cut corruption reproduces
/// only where decode graphs capture (weights/GEMM/PDL all exonerated);
/// this is the final serve-element discriminator.
pub(super) fn decode_graphs_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DECODE_GRAPH").is_none())
}

/// A/B kill for the r=1 ffn gate|up rung (see the FFN arm ladder): puts the
/// single-row decode back on `q8_0_gemm_mma_ks`, the older route.
/// Process-constant, so a captured decode graph bakes one arm and stays valid.
pub(super) fn r1_gu_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_R1GU").is_some())
}

///  prefill-pass capture: PADDOCK_PF_CAP=<N> enables graph capture of
/// forward_prefill_batch_impl's embed+prefill_layers for SINGLE-RUN chunks up
/// to N rows (a steady-state c32 mixed tick carries one ~128-row prompt). The
/// value is the max chunk width to capture - bursts (r>N, or multi-prompt
/// chunks) stay eager, so the capture is a pure add-on that only fires on the
/// thin single-prompt ticks that dominate the c32 measurement. Off by default.
/// The PF_RUNS arm htod's a run table inside prefill_layers, which a capture
/// cannot bake by pointer - but that arm only engages when `spans.len() > 1`
/// (see `pf_runs_batched` in forward.rs), while this capture only engages on
/// single-run chunks, so the two are near-disjoint at runtime. The old global
/// `PADDOCK_PF_RUNS` refusal here was therefore far too coarse: it disabled the
/// capture on every tick just because the env was set, including the
/// single-prompt steady-state ticks where PF_RUNS does nothing. P55 measured
/// what that costs to work around - trading PF_RUNS away for the capture
/// (PF_RUNS=0 + PF_CAP=256) is **-2.6%**, because PF_RUNS earns its keep on the
/// multi-prompt burst chunks the capture never touches. The guard now lives at
/// the call site as `spans.len() <= 1`, which makes `pf_runs_batched` false by
/// construction inside the captured region (its registration is also scoped to
/// one prefill_layers call: set at forward.rs:920, cleared at :2264), so both
/// features can be on at once.
pub(super) fn pf_cap() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PF_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (1..=PF_ROWS).contains(&n))
    })
}

// LCO opt-in (bit-equal on both arms): the krs spec-FA
// arms merge their split partials IN-KERNEL (last-CTA-out) and the separate
// combine launch disappears from the tick. Mirrors the pack's
// PADDOCK_SPEC_LCO gate so the engine skips its combine call exactly when
// the pack takes the LCO route.
pub(super) fn spec_lco_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_LCO").is_some())
}

// PC route row floor: the chunk sites always clear it (r ~ 1952);
// lowering to 128 routes the r~160 pure-decode ticks through the scale-free
// twins too (their per-32 kt3/kt3g residue measured 0.15 ms/tok at churn).
// Default keeps the gated behavior; PADDOCK_G4_PC_FLOOR=128 is the A/B.
pub(super) fn pc_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_PC_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (32..=4096).contains(&n))
            .unwrap_or(256)
    })
}

// kv-epilogue fold: the chunk band's kn/vn planes were pure
// round-trip - qkv_norm_rope_batch wrote them and the paged append read
// them straight back (134 MB/layer at the 2048-row SWA shape). Default on:
// pd_kv_nra_rows norms+ropes the RAW k/v GEMM planes into the caches and
// the v2 pass shrinks to q-only. Kill: PADDOCK_G4_NO_KVF.
pub(super) fn kvf_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_KVF").is_none())
}

// FIN position floor (default 0 = fin whenever the chunk floor holds):
// fin is a small win at short KV and ~flat at long, because the combine
// saving mostly cancels against the one-split walk running slower per
// launch. The floor stays as a tuning
// hook (it rides the decode-graph key band, so both variants capture
// cleanly) in case a losing regime shows up at other geometries.
pub(super) fn spec_fin_pos_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_SPEC_FIN_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// Pos-thresholded LCO election: elect the in-kernel last-CTA combine only
// on pure ticks whose max position sits below this floor. Shallow walks win
// by the deleted launch and deep walks lose to the serialized last-CTA
// merge, which makes an unconditional LCO a wash.
// 0 = off; PADDOCK_SPEC_LCO stays the unconditional opt-in.
// Rides the decode-graph key like the FIN band (spec_shallow).
pub(super) fn spec_lco_pos_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_LCO_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// Prefill overlap-lane row floor (see prefill_layers): chunks below this
// row count run the classic serial walk - at small r the cross-stream event
// edges cost more than the tiny work they hide, so an ungated overlap loses
// on short prompts and only pays on ~1.6k-row chunks.
pub(super) fn pf_overlap_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_PF_OVERLAP_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1280)
    })
}
