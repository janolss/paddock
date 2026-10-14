//! The reasoning-stop policy: while a reply's reasoning block is open, the
//! model's own stop tokens do not end the reply.
//!
//! Reasoning models sample `<|im_end|>` mid-thought. The Qwen3.8 family does
//! it on every engine (vLLM #55420, QwenLM/Qwen3.8#216: 17-38% empty answers
//! at its top reasoning effort, typically mid-enumeration), and Flash Next at
//! ~213K did it under Claude Code - a stop the full-sequence prefill puts at
//! 0.1-0.2% drew ~10% in the live decode - so the turn ended with reasoning
//! and no answer, and Claude Code's compaction failed as "summarization
//! produced empty response". SGLang (merged 2026-09) does not finish on the
//! model's EOS while the reasoning phase is open; vLLM's `force_end` (PR
//! #55562) closes the block instead of stopping. This is both:
//!
//! - inside the block the stop tokens are MASKED before the draw (-inf, a
//!   logits processor), so every draw - greedy, categorical, truncated - is
//!   the masked row's. One draw, one uniform: the seed stream advances as it
//!   does outside the block. Speculative rounds stay exact: each verify row
//!   is drawn from the masked distribution, so accept-while-equal keeps its
//!   meaning.
//! - stop PRESSURE - a stop within ln 2 of the row's top logit after
//!   temperature (p_stop >= p_top / 2), or at the top for a greedy row - on
//!   [`STOP_RUN`] picks within [`STOP_WINDOW`] is the model asking to finish,
//!   not a noise spike, and forces the exit from the next pick on: the Qwen3
//!   report's budget phrase ending in the block's close, the recipe the
//!   thinking budget already injects, so the reply answers from what it has
//!   instead of thinking on until max_tokens.
//!
//! - a degenerate LOOP inside the block - the reply cycling one span (up to
//!   [`LOOP_MAX_PERIOD`] tokens a lap), or drawing from a handful of tokens
//!   over and over - forces the same exit,
//!   so the reply answers from the reasoning it has instead of spinning to
//!   max_tokens. Gemma 4 26B under Claude Code, reciting a 217-row ledger
//!   column from memory, narrowed to two notes and then cycled `standard, `
//!   for 50K characters. vLLM's reasoning loop break (PR #52677) is the same
//!   shape - an exact cycle in the reasoning tail flips the request into the
//!   forced end - and catches the cycle; [`degenerate`] also takes the
//!   phase before it, where a random draw over two or three items never
//!   repeats exactly. Measured on 404 Gemma 4 thoughts from that campaign:
//!   none comes within 2x of either threshold.
//!
//! The host draw and the device's stop mask (pack slot 850, launched on the
//! logits rows right before the sampling kernels) compute both the same way,
//! so a slot whose backend masks on device keeps its device plans inside the
//! block ([`Sampler::arm_stop_mask`]); every other slot picks on the host
//! there, and so does a slot while its exit is forced (no draw at all).
//!
//! The block ends at its close token, at a forced exit's last token, or at a
//! tool call's opener where the dialect writes calls inside its think block
//! (a turn may end on a call - vLLM counts a call as leaving reasoning too).
//! Stop strings the caller set are text-level and runner-side; they work as
//! before.

use super::{Sampler, TokenConstraint};

/// Pressure picks inside the block that force its exit, within
/// [`STOP_WINDOW`] picks of each other - one is a noise spike the mask
/// absorbs.
pub(super) const STOP_RUN: usize = 3;
/// The window (in picks inside the block) a run of pressure picks must fall
/// in.
pub(super) const STOP_WINDOW: u32 = 64;
/// Stop ids one device row's mask carries (its 8-word kernel row).
pub const MASK_STOPS: usize = 4;
/// Leave tokens a mask carries for the verify walk's cut ([`run_cut`]).
pub const MASK_LEAVE: usize = 2;
/// p_stop >= p_top / 2, in logits after temperature. The kernel's literal is
/// the same f32.
const PRESSURE_GAP: f32 = -std::f32::consts::LN_2;

/// Loop breaking: no check before this many picks in the block (vLLM's
/// default too) - a short thought is never a runaway.
const LOOP_MIN_PICKS: u32 = 256;
/// Picks between checks.
const LOOP_EVERY: u32 = 16;
/// The diversity check's window and n-gram: a degenerate stretch has fewer
/// than [`LOOP_MAX_DISTINCT`] distinct 4-grams per position in its last 256
/// picks. Measured: Gemma 4 thoughts bottom out at 0.225, an honest
/// enumeration of ten notes at 0.44; a random draw over two notes sits at
/// 0.04, over three at 0.08, an exact cycle under 0.01.
const LOOP_WINDOW: usize = 256;
const LOOP_GRAM: usize = 4;
const LOOP_MAX_DISTINCT: f32 = 0.10;
/// The cycle check: a span of at most this many tokens...
const LOOP_MAX_PERIOD: usize = 2048;
/// ...repeated back to back at least [`LOOP_MIN_COUNT`] times over at
/// least this many picks. The longest exact repeat in those thoughts is a
/// re-quoted 62-token code line, twice; the longest cycle a raw replay fell
/// into was a ~110-number run of the ledger's weights, ~500 tokens a lap
/// (diversity-high, so only the cycle check sees it).
const LOOP_MIN_SPAN: usize = 256;
const LOOP_MIN_COUNT: usize = 3;
/// Checks in a row that must find the loop - one is a dip.
const LOOP_HITS: u8 = 2;
/// Picks the tail keeps: the longest span the cycle check reads.
const LOOP_TAIL: usize = {
    let laps = LOOP_MAX_PERIOD * LOOP_MIN_COUNT;
    let short = LOOP_MIN_SPAN + LOOP_MAX_PERIOD;
    if laps > short { laps } else { short }
};

/// The policy for one request ([`super::SamplingParams::reasoning`]), built
/// by the runner from the dialect when the rendered prompt opened a think
/// block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningStop {
    /// tokens after which the reply has left its reasoning: the block's close
    /// (`</think>`, gemma4's `<channel|>`) and, where the dialect writes calls
    /// inside its think block, a tool call's opener
    pub leave: Vec<u32>,
    /// the forced exit, ending with the block's close
    pub exit: Vec<u32>,
}

/// One device row's stop mask (pack slot 850): the row's stops set to -inf
/// before its draw, its pressure flagged. Built per pick by
/// [`Sampler::stop_mask`]; a backend that answers `supports_stop_mask` takes
/// them per row of a device-sampled call and hands the flags back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StopMask {
    /// the row's inverse temperature, 0 for a greedy row
    pub inv_t: f32,
    pub stops: [u32; MASK_STOPS],
    pub n: u32,
    /// host side, for a verify run's walk ([`run_cut`]): the block's leave
    /// tokens (u32::MAX = none)
    pub leave: [u32; MASK_LEAVE],
    /// pressure rows, counted from the run's first row through this one,
    /// that force the exit at this row (u32::MAX = never)
    pub need: u32,
}

/// Where a device verify run must END under the masks its rows carried: the
/// first row whose pick leaves the block, or whose pressure (counted over the
/// run so far) forces the exit - later rows were drawn under a mask that no
/// longer applies, so the walk treats them as rejected; `len - 1` when no row
/// does. The families' own accept walks (each re-derives the service's to
/// commit its state) cap at this row; the service's walk reaches the same
/// one through [`Sampler::walk_device_run`].
pub fn run_cut(masks: &[Option<StopMask>], picks: &[u32], pressure: &[bool]) -> usize {
    let mut p = 0u32;
    for (j, (m, &t)) in masks.iter().zip(picks).enumerate() {
        let Some(m) = m else {
            continue;
        };
        if m.leave.contains(&t) {
            return j;
        }
        p += pressure.get(j).copied().unwrap_or(false) as u32;
        if p >= m.need {
            return j;
        }
    }
    picks.len().saturating_sub(1)
}

impl StopMask {
    /// The kernel's row: `{active, inv_t bits, n_stops, stop0..stop3, pad}`;
    /// None is an inactive row (untouched, flag 0).
    pub fn words(m: Option<&StopMask>) -> [u32; 8] {
        let mut w = [0u32; 8];
        if let Some(m) = m {
            w[0] = 1;
            w[1] = m.inv_t.to_bits();
            w[2] = m.n;
            w[3..3 + MASK_STOPS].copy_from_slice(&m.stops);
        }
        w
    }
}

/// Per-reply state while the block is open (the sampler holds `None` once
/// the reply left it, and without a policy).
#[derive(Debug, Clone, Default)]
pub(super) struct Reasoning {
    /// picks made inside the block
    picks: u32,
    /// the picks that showed stop pressure (the last few)
    pressure: Vec<u32>,
    /// forcing the exit: the next `exit` index
    forcing: Option<usize>,
    /// the backend masks stops on device: device plans stay open here
    device: bool,
    /// the block's last [`LOOP_TAIL`] picks, for [`degenerate`]
    tail: std::collections::VecDeque<u32>,
    /// consecutive checks that found a loop
    loop_hits: u8,
    /// a loop was found inside a device run: the exit is forced from the
    /// pick after the run (the run's own cut is the masks')
    loop_pending: bool,
}

impl Sampler {
    /// Inside an open reasoning block (or forcing its exit): a stop is not
    /// final here.
    pub fn in_reasoning(&self) -> bool {
        self.reasoning.is_some()
    }

    /// Inside the block with no device mask to lean on, or forcing its exit:
    /// every pick is a host pick (the device plans decline).
    pub(super) fn host_reasoning(&self) -> bool {
        self.reasoning
            .as_ref()
            .is_some_and(|r| !r.device || r.forcing.is_some())
    }

    /// The backend applies stop masks on device (pack slot 850): device plans
    /// stay open inside the block, each riding [`Self::stop_mask`]. Stops
    /// that do not fit one kernel row keep the slot on the host.
    pub fn arm_stop_mask(&mut self, stops: &[u32]) {
        let leave = self.params.reasoning.as_ref().map_or(0, |p| p.leave.len());
        if let Some(r) = self.reasoning.as_mut() {
            r.device = !stops.is_empty() && stops.len() <= MASK_STOPS && leave <= MASK_LEAVE;
        }
    }

    /// The mask the device row `ahead` picks from now must carry (0 = this
    /// pick; a verify run's rows count on from its first) - None outside the
    /// block, or where the pick is not a device pick at all.
    pub fn stop_mask(&self, stops: &[u32], ahead: u32) -> Option<StopMask> {
        let (Some(p), Some(r)) = (self.params.reasoning.as_ref(), self.reasoning.as_ref()) else {
            return None;
        };
        if self.host_reasoning() {
            return None;
        }
        let t = self.params.temperature;
        // the pressure picks still inside the window at that row: the run's
        // own pressure rows complete the forcing count from there
        let at = r.picks + 1 + ahead;
        let alive = r.pressure.iter().filter(|&&q| at - q < STOP_WINDOW).count() as u32;
        let mut m = StopMask {
            inv_t: if t <= 0.0 { 0.0 } else { 1.0 / t },
            stops: [0; MASK_STOPS],
            n: stops.len() as u32,
            leave: [u32::MAX; MASK_LEAVE],
            need: if p.exit.is_empty() {
                u32::MAX
            } else {
                (STOP_RUN as u32).saturating_sub(alive)
            },
        };
        m.stops[..stops.len()].copy_from_slice(stops);
        m.leave[..p.leave.len()].copy_from_slice(&p.leave);
        Some(m)
    }

    /// The exit id this pick is while the exit is being forced (no draw).
    fn forced_exit(&self) -> Option<u32> {
        let i = self.reasoning.as_ref()?.forcing?;
        self.params.reasoning.as_ref()?.exit.get(i).copied()
    }

    /// Commit a forced pick: the exit advances, its last id ends the block.
    fn note_forced(&mut self) {
        let len = self.params.reasoning.as_ref().map_or(0, |p| p.exit.len());
        let Some(r) = self.reasoning.as_mut() else {
            return;
        };
        match r.forcing {
            Some(i) if i + 1 < len => r.forcing = Some(i + 1),
            _ => self.reasoning = None,
        }
    }

    /// Commit a drawn pick (host or device) and its row's pressure: a leave
    /// token ends the block; [`STOP_RUN`] pressure picks within the window,
    /// or a loop ([`degenerate`]), force the exit from the next pick on.
    pub fn note_drawn(&mut self, t: u32, pressure: bool) {
        self.note_pick(t, pressure);
        self.start_loop_exit();
    }

    /// [`Self::note_drawn`] with a loop only marked: inside a device run the
    /// exit waits for the run's end.
    fn note_pick(&mut self, t: u32, pressure: bool) {
        let (Some(p), Some(r)) = (self.params.reasoning.as_ref(), self.reasoning.as_mut()) else {
            return;
        };
        if p.leave.contains(&t) {
            self.reasoning = None;
            return;
        }
        r.picks += 1;
        if !p.exit.is_empty() && r.forcing.is_none() {
            r.tail.push_back(t);
            if r.tail.len() > LOOP_TAIL {
                r.tail.pop_front();
            }
            if r.picks >= LOOP_MIN_PICKS && r.picks.is_multiple_of(LOOP_EVERY) && loop_break_on() {
                match degenerate(r.tail.make_contiguous()) {
                    Some(why) => {
                        r.loop_hits += 1;
                        if r.loop_hits >= LOOP_HITS && !r.loop_pending {
                            r.loop_pending = true;
                            tracing::info!(
                                "reasoning loop: {why} after {} picks in the block - forcing its exit",
                                r.picks
                            );
                        }
                    }
                    None => r.loop_hits = 0,
                }
            }
        }
        if !pressure {
            return;
        }
        let now = r.picks;
        r.pressure.retain(|&q| now - q < STOP_WINDOW);
        r.pressure.push(now);
        if !p.exit.is_empty() && r.pressure.len() >= STOP_RUN {
            r.forcing = Some(0);
            tracing::debug!(
                "reasoning-stop: stop pressure on {STOP_RUN} picks in the block - forcing its exit"
            );
        }
    }

    /// A loop found by [`Self::note_drawn`] forces the exit from the next
    /// pick on.
    fn start_loop_exit(&mut self) {
        if let Some(r) = self.reasoning.as_mut()
            && r.loop_pending
            && r.forcing.is_none()
        {
            r.forcing = Some(0);
        }
    }

    /// The device-sampled rows of one slot's accepted run (accepted drafts'
    /// picks, then the bonus), with each row's pressure flag (missing = no
    /// mask rode the row): returns the row the run must END at - the first
    /// pick after which the rows' mask no longer holds (the block closed, or
    /// its exit is forced from the next pick on). Later rows were drawn under
    /// a mask that no longer applies, so the walk treats them as rejected.
    pub fn walk_device_run(&mut self, picks: &[u32], pressure: &[bool]) -> usize {
        let last = picks.len().saturating_sub(1);
        if !self.in_reasoning() {
            return last;
        }
        for (r, &t) in picks.iter().enumerate() {
            self.note_pick(t, pressure.get(r).copied().unwrap_or(false));
            if self.host_reasoning() || !self.in_reasoning() {
                return r;
            }
        }
        // a loop does not cut the run: every row was drawn under a mask that
        // still held, and the families cut from the masks alone
        self.start_loop_exit();
        last
    }
}

/// The row's stop pressure, read before the mask: a stop within ln 2 of the
/// row's max after temperature, or at the max for a greedy row - the slot-850
/// kernel's test, same f32 arithmetic.
fn pressure(row: &[f32], stops: &[u32], inv_t: f32) -> bool {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    stops.iter().filter_map(|&s| row.get(s as usize)).any(|&l| {
        if inv_t > 0.0 {
            (l - m) * inv_t >= PRESSURE_GAP
        } else {
            l >= m
        }
    })
}

/// One host draw under the policy: a forced exit's id (no draw), else bias
/// and penalties, then inside the block the pressure read and the stops
/// masked, then the draw - the constraint's when one is set (stop tokens
/// legal only where it may end). None = the constraint has no legal
/// continuation.
fn draw(
    sampler: &mut Sampler,
    row: &mut [f32],
    history: &[u32],
    constraint: Option<&dyn TokenConstraint>,
    stops: &[u32],
) -> Option<u32> {
    if let Some(t) = sampler.forced_exit() {
        sampler.note_forced();
        return Some(t);
    }
    sampler.process_row(row, history);
    let mut press = false;
    if sampler.in_reasoning() {
        let t = sampler.params.temperature;
        // a stop the constraint forbids here asks for nothing
        press = pressure(row, stops, if t <= 0.0 { 0.0 } else { 1.0 / t })
            && constraint.is_none_or(|c| c.may_stop());
        for &s in stops {
            if let Some(l) = row.get_mut(s as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    let t = match constraint {
        None => sampler.draw_row(row, None),
        Some(c) => {
            let may_stop = c.may_stop();
            sampler.draw_row(
                row,
                Some(&mut |id: u32| {
                    if stops.contains(&id) {
                        may_stop
                    } else {
                        c.allows(id)
                    }
                }),
            )
        }
    }?;
    sampler.note_drawn(t, press);
    Some(t)
}

/// Loop breaking is on (PADDOCK_NO_REASONING_LOOP_BREAK turns it off for an
/// A/B; development builds only).
fn loop_break_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_REASONING_LOOP_BREAK").is_none())
}

/// The reply has fallen into a loop: the tail (oldest first) ends in a span
/// of at most [`LOOP_MAX_PERIOD`] tokens repeated back to back, or its last
/// [`LOOP_WINDOW`] picks hold too few distinct [`LOOP_GRAM`]-grams. Some(why)
/// for the log.
fn degenerate(tail: &[u32]) -> Option<String> {
    let n = tail.len();
    // smallest period first; each candidate is read newest-first, so a
    // period the last pick already contradicts costs one compare
    for p in 1..=LOOP_MAX_PERIOD.min(n.saturating_sub(1)) {
        let len = p * LOOP_MIN_COUNT.max(LOOP_MIN_SPAN.div_ceil(p));
        if len <= n && (n - len + p..n).rev().all(|i| tail[i] == tail[i - p]) {
            return Some(format!("a {p}-token span repeated over {len} picks"));
        }
    }
    if n >= LOOP_WINDOW {
        let w = &tail[n - LOOP_WINDOW..];
        let grams: std::collections::HashSet<&[u32]> = w.windows(LOOP_GRAM).collect();
        let total = LOOP_WINDOW - LOOP_GRAM + 1;
        if (grams.len() as f32) < LOOP_MAX_DISTINCT * total as f32 {
            return Some(format!(
                "{} distinct {LOOP_GRAM}-grams in its last {LOOP_WINDOW} picks",
                grams.len()
            ));
        }
    }
    None
}

/// One host pick: [`draw`], then the machine advanced on what was picked.
/// Err = the constraint has no legal continuation.
pub fn pick_next(
    sampler: &mut Sampler,
    logits: &mut [f32],
    history: &[u32],
    constraint: &mut Option<Box<dyn TokenConstraint>>,
    stop_tokens: &[u32],
) -> Result<u32, String> {
    let next = draw(sampler, logits, history, constraint.as_deref(), stop_tokens)
        .ok_or("constraint deadlock: no legal next token")?;
    if let Some(c) = constraint
        && !stop_tokens.contains(&next)
    {
        c.accept(next);
    }
    Ok(next)
}

/// The host-sampled verify round's pick for one row: [`pick_next`]'s
/// semantics at the verify position, with the machine NOT advanced (the round
/// accepts its committed picks itself). `true` = grammar deadlock: the
/// unconstrained pick comes back and the round must end there - the next
/// dense step hits pick_next's own error and fails the request properly.
pub fn pick_verify_row(
    sampler: &mut Sampler,
    row: &mut [f32],
    constraint: Option<&dyn TokenConstraint>,
    stops: &[u32],
) -> (u32, bool) {
    match draw(sampler, row, &[], constraint, stops) {
        Some(t) => (t, false),
        None => (sampler.sample(row, &[]), true),
    }
}

#[cfg(test)]
mod tests {
    use super::super::SamplingParams;
    use super::*;

    const EOS: u32 = 0;
    const CLOSE: u32 = 3;
    const CALL: u32 = 4;

    fn policy(temperature: f32, seed: u64) -> Sampler {
        Sampler::new(SamplingParams {
            temperature,
            seed,
            reasoning: Some(ReasoningStop {
                leave: vec![CLOSE, CALL],
                exit: vec![5, 6, CLOSE],
            }),
            ..SamplingParams::default()
        })
    }

    fn row(eos: f32) -> Vec<f32> {
        // eos, two text tokens, close, call, exit-phrase tokens
        vec![eos, 1.0, 0.5, -9.0, -9.0, -9.0, -9.0]
    }

    fn pick(s: &mut Sampler, eos: f32) -> u32 {
        pick_next(s, &mut row(eos), &[], &mut None, &[EOS]).unwrap()
    }

    /// Greedy, EOS on top: inside the block the second-best token comes back
    /// (the mask), and without a device mask the device plans decline.
    #[test]
    fn a_stop_inside_reasoning_is_masked() {
        let mut s = policy(0.0, 1);
        assert!(s.in_reasoning());
        assert!(!s.is_device_plannable() && !s.is_pure_greedy() && s.device_plan().is_none());
        assert!(s.is_spec_safe(), "host-sampled spec rounds stay open");
        assert_eq!(pick(&mut s, 9.0), 1);
    }

    /// Pressure on three picks within the window forces the exit from the
    /// next pick on, token by token, and the block's close ends the policy -
    /// stops are final again and the device plans come back.
    #[test]
    fn a_run_of_pressure_forces_the_exit() {
        let mut s = policy(0.0, 1);
        assert_eq!(pick(&mut s, 9.0), 1);
        assert_eq!(pick(&mut s, 9.0), 1);
        assert_eq!(pick(&mut s, 9.0), 1, "the third pressure pick still draws");
        assert_eq!(pick(&mut s, 9.0), 5, "the exit starts on the next pick");
        assert_eq!(pick(&mut s, 9.0), 6);
        assert_eq!(pick(&mut s, 9.0), CLOSE);
        assert!(!s.in_reasoning());
        assert!(s.is_device_plannable());
        assert_eq!(
            pick(&mut s, 9.0),
            EOS,
            "after the block a stop ends the reply"
        );
    }

    /// Pressure is p_stop >= p_top / 2 after temperature: 0.5 nats under the
    /// top is pressure at T = 1, 1.0 under is not, and at T = 0.5 the 0.5 gap
    /// doubles to 1.0 - not pressure either.
    #[test]
    fn pressure_is_half_the_top_after_temperature() {
        let force_after_three = |t: f32, gap: f32| {
            let mut s = policy(t, 7);
            for _ in 0..3 {
                let mut r = row(1.0 - gap);
                pick_next(&mut s, &mut r, &[], &mut None, &[EOS]).unwrap();
            }
            s.forced_exit().is_some()
        };
        assert!(force_after_three(1.0, 0.5));
        assert!(!force_after_three(1.0, 1.0));
        assert!(!force_after_three(0.5, 0.5));
    }

    /// Pressure picks spread wider than the window are noise spikes: each is
    /// masked and none forces the exit.
    #[test]
    fn spread_out_pressure_never_forces() {
        let mut s = policy(0.0, 1);
        for i in 0..(3 * STOP_WINDOW) {
            let eos = if i % STOP_WINDOW == 0 { 9.0 } else { -9.0 };
            assert_eq!(pick(&mut s, eos), 1);
        }
        assert!(s.in_reasoning());
    }

    /// A tool call's opener leaves the block: the turn may end on the call.
    #[test]
    fn a_call_opener_leaves_the_block() {
        let mut s = policy(0.0, 1);
        let mut r = row(-9.0);
        r[CALL as usize] = 20.0;
        assert_eq!(
            pick_next(&mut s, &mut r, &[], &mut None, &[EOS]).unwrap(),
            CALL
        );
        assert!(!s.in_reasoning());
    }

    /// The draw is the masked distribution: EOS carries twice the first text
    /// token's weight, yet never comes out, and the two text tokens keep
    /// their own e:1 ratio.
    #[test]
    fn the_draw_is_the_masked_distribution() {
        let logits = [2.0f32.ln() + 1.0, 1.0, 0.0, -50.0, -50.0, -50.0, -50.0];
        let mut counts = [0usize; 7];
        let n = 30_000;
        for seed in 0..n {
            let mut s = policy(1.0, seed);
            let t = pick_next(&mut s, &mut logits.clone(), &[], &mut None, &[EOS]).unwrap();
            counts[t as usize] += 1;
        }
        assert_eq!(counts[0], 0);
        let ratio = counts[1] as f64 / counts[2] as f64;
        let e = std::f64::consts::E;
        assert!((ratio - e).abs() < 0.12, "ratio {ratio} vs e");
    }

    /// The verify row's pick follows the same policy.
    #[test]
    fn verify_rows_take_the_policy() {
        let mut s = policy(0.0, 1);
        for _ in 0..3 {
            assert_eq!(
                pick_verify_row(&mut s, &mut row(9.0), None, &[EOS]),
                (1, false)
            );
        }
        assert_eq!(
            pick_verify_row(&mut s, &mut row(9.0), None, &[EOS]),
            (5, false)
        );
    }

    /// Armed for the device mask, the slot keeps its device plans inside the
    /// block - each pick carrying its mask - until pressure forces the exit,
    /// which is a host affair (no mask, no plan); a finisher's peek never
    /// carries a mask, so it declines throughout.
    #[test]
    fn the_device_mask_keeps_device_plans_open() {
        let mut s = policy(0.6, 1);
        s.arm_stop_mask(&[EOS, 9]);
        assert!(s.is_device_plannable() && s.device_plan().is_some());
        assert!(s.peek_device_plan().is_none() && s.peek_device_plan_trunc().is_none());
        let m = s.stop_mask(&[EOS, 9], 0).expect("mask");
        assert_eq!((m.n, &m.stops[..2]), (2, &[EOS, 9][..]));
        assert!((m.inv_t - 1.0 / 0.6).abs() < 1e-6);
        assert_eq!(
            StopMask::words(Some(&m)),
            [1, m.inv_t.to_bits(), 2, EOS, 9, 0, 0, 0]
        );
        s.note_drawn(1, true);
        s.note_drawn(2, true);
        s.note_drawn(1, true);
        assert!(s.device_plan().is_none() && s.stop_mask(&[EOS], 0).is_none());
        assert_eq!(pick(&mut s, 9.0), 5);
        // five stops do not fit one kernel row: the host keeps the slot
        let mut s = policy(0.0, 1);
        s.arm_stop_mask(&[0, 1, 2, 3, 4]);
        assert!(!s.is_device_plannable() && s.stop_mask(&[0, 1, 2, 3, 4], 0).is_none());
    }

    /// A device spec round's accepted run ends where the mask stops holding:
    /// at the block's close, or at the pick whose pressure forced the exit.
    #[test]
    fn a_device_run_ends_where_the_mask_stops_holding() {
        let mut s = policy(0.0, 1);
        s.arm_stop_mask(&[EOS]);
        assert_eq!(s.walk_device_run(&[1, 2, CLOSE, 1, 2], &[]), 2);
        assert!(!s.in_reasoning());
        assert_eq!(s.walk_device_run(&[1, 2], &[]), 1, "outside: all of it");
        let mut s = policy(0.0, 1);
        s.arm_stop_mask(&[EOS]);
        let p = [true, false, true, true, false];
        assert_eq!(s.walk_device_run(&[1, 2, 1, 2, 1], &p), 3);
        assert_eq!(s.forced_exit(), Some(5));
    }

    /// The families' cut ([`run_cut`], from the masks the rows carried) and
    /// the service's walk ([`Sampler::walk_device_run`], from the sampler's
    /// own state) end every run at the same row, rounds after rounds, with
    /// pressure picks aging out of the window between them.
    #[test]
    fn the_run_cut_agrees_with_the_walk() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..400 {
            let mut s = policy(0.0, 1);
            s.arm_stop_mask(&[EOS]);
            while s.stop_mask(&[EOS], 0).is_some() {
                let len = 1 + (next() % 6) as usize;
                let masks: Vec<_> = (0..len as u32).map(|j| s.stop_mask(&[EOS], j)).collect();
                let picks: Vec<u32> = (0..len)
                    .map(|_| if next() % 97 == 0 { CLOSE } else { 1 })
                    .collect();
                let pressure: Vec<bool> = (0..len).map(|_| next() % 23 == 0).collect();
                let cut = run_cut(&masks, &picks, &pressure);
                assert_eq!(s.walk_device_run(&picks, &pressure), cut);
            }
        }
    }

    /// Greedy on a fixed row: the block cycles one token. The first check
    /// (pick 256) finds it, the second (272) confirms it, and the exit is
    /// forced from the next pick on.
    #[test]
    fn a_cycle_in_the_block_forces_the_exit() {
        let mut s = policy(0.0, 1);
        for i in 0..(LOOP_MIN_PICKS + LOOP_EVERY) {
            assert_eq!(pick(&mut s, -9.0), 1, "pick {i}");
        }
        assert_eq!(
            pick(&mut s, -9.0),
            5,
            "the exit starts after the second hit"
        );
        assert_eq!(pick(&mut s, -9.0), 6);
        assert_eq!(pick(&mut s, -9.0), CLOSE);
        assert!(!s.in_reasoning());
    }

    /// A row-by-row pick (one token per row, `n` tokens wide) steered by a
    /// sequence - the policy's host draw on what the sequence says.
    fn steer(s: &mut Sampler, tok: u32, width: usize) -> u32 {
        let mut r = vec![-9.0f32; width];
        r[tok as usize] = 9.0;
        pick_next(s, &mut r, &[], &mut None, &[EOS]).unwrap()
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    /// No exact cycle, only a random draw over two tokens - the phase before
    /// the collapse - still forces the exit (diversity under the floor).
    #[test]
    fn a_two_token_random_walk_forces_the_exit() {
        let mut s = policy(0.0, 1);
        let mut next = xorshift(0x2545f4914f6cdd1d);
        let mut n = 0;
        while s.forced_exit().is_none() {
            let t = if next().is_multiple_of(2) { 1 } else { 2 };
            assert_eq!(steer(&mut s, t, 8), t);
            n += 1;
            assert!(n <= 400, "never forced");
        }
        assert_eq!(n, (LOOP_MIN_PICKS + LOOP_EVERY) as usize);
    }

    /// A long lap the diversity check cannot see - 300 distinct tokens, over
    /// and over, like the ledger's weight column a raw replay cycled - is the
    /// cycle check's: three laps (900 picks), then the confirming check.
    #[test]
    fn a_long_cycle_forces_the_exit() {
        let mut s = policy(0.0, 1);
        let mut n = 0u32;
        while s.forced_exit().is_none() {
            let t = 7 + n % 300;
            assert_eq!(steer(&mut s, t, 310), t);
            n += 1;
            assert!(n <= 2000, "never forced");
        }
        assert_eq!(
            n, 928,
            "first hit at the check after 900, confirmed 16 later"
        );
    }

    /// Reasoning over a real vocabulary never trips it: 4000 picks drawn from
    /// 60 tokens keep nearly every 4-gram distinct.
    #[test]
    fn varied_reasoning_never_breaks() {
        let mut s = policy(0.0, 1);
        let mut next = xorshift(0x9e3779b97f4a7c15);
        for _ in 0..4000 {
            let t = 7 + (next() % 60) as u32;
            assert_eq!(steer(&mut s, t, 70), t);
        }
        assert!(s.in_reasoning() && s.forced_exit().is_none());
    }

    /// Inside a device run the loop does not cut it (the families cut from
    /// the masks, and every row was drawn under one that held): the run
    /// commits whole and the exit starts at the pick after it.
    #[test]
    fn a_loop_in_a_device_run_exits_after_the_run() {
        let mut s = policy(0.0, 1);
        s.arm_stop_mask(&[EOS]);
        let mut runs = 0;
        while s.stop_mask(&[EOS], 0).is_some() {
            assert_eq!(s.walk_device_run(&[1, 1, 1, 1, 1], &[]), 4, "a whole run");
            runs += 1;
            assert!(runs <= 100, "never forced");
        }
        assert_eq!(s.forced_exit(), Some(5));
        // the second hit lands at pick 272 = inside run 55 (picks 271-275)
        assert_eq!(runs, 55);
    }

    /// Without a policy nothing changes.
    #[test]
    fn no_policy_no_change() {
        let mut s = Sampler::new(SamplingParams::default());
        assert!(!s.in_reasoning());
        assert_eq!(pick(&mut s, 9.0), EOS);
    }
}
