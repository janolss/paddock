//! Bounded complete-state retention. Each key owns immutable attention pages
//! (including a partial last page) paired with its complete Mamba slab.
use super::*;

#[derive(Default)]
pub(super) struct Checkpoint {
    pub table: BlockTable,
    pub history: Vec<u32>,
    pub touched: u64,
    pub anchor: bool,
    pub reply: bool,
    pub reply_candidate: bool,
}

impl Checkpoint {
    pub(super) fn clear(&mut self, pool: &mut KvPool) {
        self.table.clear(pool);
        self.history.clear();
        self.anchor = false;
        self.reply = false;
        self.reply_candidate = false;
    }
}

fn covers(c: &Checkpoint, tokens: &[u32]) -> bool {
    !c.history.is_empty() && tokens.starts_with(&c.history)
}

pub(super) fn retain_active_anchors(cache: &mut [Checkpoint], active: &[&[u32]]) {
    // Protection is an active-conversation policy, not an immortal cache pin.
    // Keep old keys as ordinary LRU entries, but let a new cohort reclaim their
    // protection before its prefill starts competing for rolling snapshots.
    for c in cache {
        if c.anchor && !active.iter().any(|tokens| covers(c, tokens)) {
            c.anchor = false;
        }
        if c.reply && !active.iter().any(|tokens| covers(c, tokens)) {
            c.reply = false;
        }
    }
}

/// Keep two successive boundaries before anchoring; once a user-turn anchor
/// exists, keep it and a rolling tail. Retire this conversation's older cuts
/// before touching another conversation, even when its tool loop runs ahead.
pub(super) fn replacement(cache: &[Checkpoint], tokens: &[u32]) -> Option<usize> {
    if let Some((i, c)) = cache.iter().enumerate().find(|(_, c)| c.history == tokens) {
        return (!c.anchor && !c.reply).then_some(i);
    }
    let anchored = cache.iter().any(|c| c.anchor && covers(c, tokens));
    let keep = if anchored {
        tokens.len()
    } else {
        cache
            .iter()
            .filter(|c| covers(c, tokens))
            .map(|c| c.history.len())
            .max()
            .unwrap_or(0)
    };
    cache
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.anchor && !c.reply && !(c.history.len() >= keep && covers(c, tokens)))
        .min_by_key(|(_, c)| {
            let old = c.history.len() < keep && covers(c, tokens);
            // A rewind/cancellation can return to an earlier prompt. Retire
            // its now-unused descendant before stealing a peer's only cut.
            let descendant = !tokens.is_empty() && c.history.starts_with(tokens);
            (
                if old || descendant {
                    0
                } else if c.history.is_empty() {
                    1
                } else {
                    2
                },
                if old || descendant {
                    c.history.len() as u64
                } else {
                    c.touched
                },
            )
        })
        .map(|(i, _)| i)
        .or_else(|| {
            // With two entries, an unrelated conversation's held anchor
            // plus this prefill's latest cut otherwise leaves no candidate.
            // The new conversation freezes at its first cut forever. Keep
            // the covering cut and retire the least-recent unrelated anchor
            // only when no ordinary victim exists. A covering anchor is
            // never selected here, and no extra cache/state memory is added.
            cache
                .iter()
                .enumerate()
                .filter(|(_, c)| c.anchor && !covers(c, tokens))
                .min_by_key(|(_, c)| c.touched)
                .map(|(i, _)| i)
        })
}

#[cfg(test)]
fn mark_anchor(cache: &mut [Checkpoint], tokens: &[u32], upto: usize) -> Option<usize> {
    mark_anchor_limited(cache, tokens, upto, cache.len() / PREFIX_PER_SLOT)
}

pub(super) fn mark_anchor_limited(
    cache: &mut [Checkpoint],
    tokens: &[u32],
    upto: usize,
    slots: usize,
) -> Option<usize> {
    // Protect at most one user-turn cut per slot, leaving room for rolling
    // tails (and an MLX reply boundary). Partial-page tails
    // must not displace an active conversation's replayable user-turn cut.
    let cap = slots.min(cache.len().saturating_sub(1));
    if cap == 0 {
        return None;
    }
    let index = cache
        .iter()
        .enumerate()
        .filter(|(_, c)| c.history.len() <= upto && covers(c, tokens))
        .max_by_key(|(_, c)| c.history.len())
        .map(|(i, _)| i)?;
    cache[index].anchor = true;
    // A new user turn has already re-rendered the preceding reply. Its old
    // tool-boundary pins need not remain protected through the next turn.
    for c in cache.iter_mut().filter(|c| covers(c, tokens)) {
        c.reply = false;
    }
    if cache.iter().filter(|c| c.anchor).count() > cap
        && let Some(old) = cache
            .iter()
            .enumerate()
            .filter(|(i, c)| *i != index && c.anchor)
            .min_by_key(|(_, c)| c.touched)
            .map(|(i, _)| i)
    {
        cache[old].anchor = false;
    }
    Some(cache[index].history.len())
}

/// A reply snapshot must not destroy this request's reusable prompt. The first
/// tool may be reserialized from its very first byte on the next HTTP request.
fn reply_replacement(cache: &[Checkpoint], tokens: &[u32], prompt_end: usize) -> Option<usize> {
    if cache.iter().any(|c| c.reply && covers(c, tokens)) {
        return None;
    }
    cache
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            !c.anchor && !c.reply && !(covers(c, tokens) && c.history.len() <= prompt_end)
        })
        .min_by_key(|(_, c)| {
            (
                if c.reply_candidate && covers(c, tokens) {
                    0
                } else if c.history.is_empty() {
                    1
                } else {
                    2
                },
                c.touched,
            )
        })
        .map(|(i, _)| i)
}

impl Nemotron {
    pub(super) fn snapshot_reply(&mut self, slot: usize) -> Result<()> {
        let s = &self.slots[slot];
        if self.embedding.ty != mlx::AFFINE8
            || self.prefix_limit() == 1
            || s.tool_started
            || s.prompt_end == 0
            // A partial first page may contain only the leading newline.
            // Pinning that steals the one reply entry from useful narration
            // on a later tool turn. Require a full page of actual progress.
            || s.history.len().saturating_sub(s.prompt_end) < BLOCK_TOKENS
            || !s.history.len().is_multiple_of(BLOCK_TOKENS)
            || s.history.last().is_some_and(|&id| {
                self.reply_trim_tokens.get(id as usize).copied().unwrap_or(false)
            })
        {
            return Ok(());
        }
        if let Some(index) = reply_replacement(&self.prefixes, &s.history, s.prompt_end) {
            self.file_checkpoint(slot, index, true)?;
        }
        Ok(())
    }

    pub(super) fn pin_reply(&mut self, slot: usize) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        s.tool_started = true;
        if self.embedding.ty != mlx::AFFINE8 || self.prefix_limit() == 1 {
            return;
        }
        let s = &self.slots[slot];
        if self
            .prefixes
            .iter()
            .any(|c| c.reply && covers(c, &s.history))
        {
            return;
        }
        let Some(index) = self
            .prefixes
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.reply_candidate && c.history.len() > s.prompt_end && covers(c, &s.history)
            })
            .max_by_key(|(_, c)| c.history.len())
            .map(|(i, _)| i)
        else {
            return;
        };
        self.prefixes[index].reply = true;
        if self.prefixes.iter().filter(|c| c.reply).count() > self.slots.len()
            && let Some(old) = self
                .prefixes
                .iter()
                .enumerate()
                .filter(|(i, c)| *i != index && c.reply)
                .min_by_key(|(_, c)| c.touched)
                .map(|(i, _)| i)
        {
            self.prefixes[old].reply = false;
        }
    }

    pub(super) fn copy_attention_page(&self, source: u32, destination: u32) -> Result<()> {
        // Two KV heads × 128 channels, BF16 in both elected formats. Pages
        // already belong to the fixed pool: COW never expands the allocation.
        let bytes = BLOCK_TOKENS * 256 * 2;
        let regions = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.mixer {
                Mixer::Attention(a) => Some(a),
                _ => None,
            })
            .flat_map(|a| {
                [&a.keys, &a.values].map(|b| {
                    (
                        b,
                        source as usize * bytes,
                        b,
                        destination as usize * bytes,
                        bytes,
                    )
                })
            })
            .collect::<Vec<_>>();
        self.device.copy_regions(&regions)
    }

    pub(super) fn prefix_limit(&self) -> usize {
        #[cfg(test)]
        if self.single_snapshot {
            return 1;
        }
        self.prefixes.len()
    }

    pub(super) fn evict_prefix(&mut self) -> bool {
        let Some(index) = self
            .prefixes
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.history.is_empty())
            .min_by_key(|(_, c)| (c.anchor, c.reply, c.touched))
            .map(|(i, _)| i)
        else {
            return false;
        };
        self.prefixes[index].clear(&mut self.pool);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(branch: u32, n: usize, touched: u64) -> Checkpoint {
        Checkpoint {
            history: vec![branch; n],
            touched,
            ..Default::default()
        }
    }

    #[test]
    fn reply_snapshots_preserve_prompt_and_hold_the_first_tool_boundary() {
        let mut cache = [cut(1, 512, 1), cut(1, 513, 2), Checkpoint::default()];
        mark_anchor_limited(&mut cache, &[1; 513], 512, 1);
        let i = reply_replacement(&cache, &[1; 528], 513).unwrap();
        assert_eq!(i, 2);
        cache[i] = cut(1, 528, 3);
        cache[i].reply_candidate = true;
        assert_eq!(reply_replacement(&cache, &[1; 544], 513), Some(i));
        cache[i].reply = true;
        assert_eq!(reply_replacement(&cache, &[1; 560], 513), None);
        assert_eq!(replacement(&cache, &[1; 700]), Some(1));
        cache[1] = cut(1, 700, 4);
        assert_eq!(replacement(&cache, &[1; 800]), Some(1));
        assert_eq!(
            mark_anchor_limited(&mut cache, &[1; 700], 699, 1),
            Some(528)
        );
        assert!(!cache[0].anchor && cache[2].anchor);
        assert!(cache.iter().all(|c| !c.reply));
    }

    #[test]
    fn retired_cohort_anchors_cannot_evict_active_rolling_prefixes() {
        let mut cache = (1..=4)
            .flat_map(|b| [cut(b, 128, b as u64), cut(b, 256, b as u64)])
            .collect::<Vec<_>>();
        for b in 1..=4 {
            mark_anchor(&mut cache, &vec![b; 257], 128);
        }
        assert_eq!(cache.iter().filter(|c| c.anchor).count(), 4);
        retain_active_anchors(&mut cache, &[&[9; 512]]);
        assert!(cache.iter().all(|c| !c.anchor));
        assert!(
            cache.iter().all(|c| !c.history.is_empty()),
            "demotion does not delete keys"
        );
        for n in [128, 256, 384, 512] {
            for b in 5..=8 {
                let tokens = vec![b; n];
                let i = replacement(&cache, &tokens).unwrap();
                cache[i] = cut(b, n, n as u64 + b as u64);
            }
            for b in 5..=8 {
                assert!(cache.iter().any(|c| c.history == vec![b; n]));
            }
        }
        for b in 5..=8 {
            mark_anchor(&mut cache, &vec![b; 513], 384);
        }
        retain_active_anchors(&mut cache, &[&[5; 513], &[7; 1024]]);
        let held = cache
            .iter()
            .filter(|c| c.anchor)
            .map(|c| c.history[0])
            .collect::<Vec<_>>();
        assert_eq!(held, vec![5, 7]);
    }

    #[test]
    fn a_rewound_prompt_retires_its_descendant_before_a_peers_cut() {
        let mut cache = [
            cut(1, 512, 10),
            cut(1, 640, 20),
            cut(2, 512, 1),
            cut(2, 513, 2),
        ];
        assert_eq!(mark_anchor(&mut cache, &[1; 513], 512), Some(512));
        assert_eq!(replacement(&cache, &[1; 513]), Some(1));
        cache[0].anchor = false;
        assert_eq!(
            replacement(&cache, &[1; 513]),
            Some(1),
            "keep the covering cut even without an anchor"
        );
    }

    #[test]
    fn anchors_leave_a_rolling_entry_and_demote_the_oldest() {
        let mut cache = [cut(1, 512, 1), cut(1, 1024, 2)];
        assert_eq!(mark_anchor(&mut cache, &[1; 1025], 512), Some(512));
        assert_eq!(replacement(&cache, &[1; 1536]), Some(1));
        assert_eq!(mark_anchor(&mut cache, &[1; 1025], 1024), Some(1024));
        assert!(!cache[0].anchor && cache[1].anchor);
        assert_eq!(
            replacement(&cache, &[1; 1024]),
            None,
            "do not overwrite a held boundary"
        );
        assert_eq!(mark_anchor(&mut cache, &[2; 2048], 2048), None);
        assert_eq!(mark_anchor(&mut cache, &[1; 128], 1024), None);
        assert_eq!(mark_anchor(&mut [cut(1, 512, 1)], &[1; 1024], 512), None);
    }

    #[test]
    fn a_previous_conversations_anchor_cannot_freeze_new_prefill_at_512() {
        let mut cache = [cut(1, 512, 1), cut(1, 1024, 2)];
        assert_eq!(mark_anchor(&mut cache, &[1; 1025], 512), Some(512));
        for n in [512, 1024, 1536] {
            let i = replacement(&cache, &vec![2; n]).expect("new conversation must advance");
            cache[i] = cut(2, n, n as u64);
        }
        assert_eq!(cache.iter().map(|c| c.history.len()).max(), Some(1536));
        assert_eq!(mark_anchor(&mut cache, &[2; 1537], 1535), Some(1024));
        let i = replacement(&cache, &[2; 2048]).unwrap();
        assert!(!cache[i].anchor, "a covering anchor still has priority");
    }

    #[test]
    fn four_interleaved_branches_keep_current_state_with_bounded_anchors() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        let mut lengths = [512; 4];
        let mut clock = 0;
        for branch in [0, 1, 2, 3, 0, 0, 0, 2, 3, 1, 0, 1, 3] {
            let n = lengths[branch];
            let tokens = vec![branch as u32; n];
            let i = replacement(&cache, &tokens).unwrap();
            clock += 1;
            cache[i] = cut(branch as u32, n, clock);
            if n == 512 {
                mark_anchor(&mut cache, &tokens, n);
            }
            lengths[branch] += 512;
            for (peer, &next) in lengths.iter().enumerate().filter(|(_, n)| **n > 512) {
                assert!(
                    cache
                        .iter()
                        .any(|c| c.history.len() == next - 512 && c.history[0] == peer as u32)
                );
            }
            assert!(cache.iter().filter(|c| c.anchor).count() <= 4);
        }
    }

    #[test]
    fn retains_previous_cut_until_prompt_end_can_be_anchored() {
        let mut cache = [Checkpoint::default(), Checkpoint::default()];
        for n in [512, 1024, 1536] {
            let i = replacement(&cache, &vec![1; n]).unwrap();
            cache[i] = cut(1, n, n as u64);
        }
        assert_eq!(mark_anchor(&mut cache, &[1; 1536], 1535), Some(1024));
        let i = replacement(&cache, &[1; 2048]).unwrap();
        assert!(!cache[i].anchor);
        cache[i] = cut(1, 2048, 2048);
        assert!(cache.iter().any(|c| c.anchor && c.history.len() == 1024));
    }
}
