//! Bounded complete-state retention. Capture points and model arithmetic are
//! unchanged; each key owns both immutable attention pages and a Mamba slab.
use super::*;

#[derive(Default)]
pub(super) struct Checkpoint {
    pub table: BlockTable,
    pub history: Vec<u32>,
    pub touched: u64,
    pub anchor: bool,
}

impl Checkpoint {
    pub(super) fn clear(&mut self, pool: &mut KvPool) {
        self.table.clear(pool);
        self.history.clear();
        self.anchor = false;
    }
}

fn covers(c: &Checkpoint, tokens: &[u32]) -> bool {
    !c.history.is_empty() && tokens.starts_with(&c.history)
}

/// Keep two successive boundaries before anchoring; once a user-turn anchor
/// exists, keep it and a rolling tail. Retire this conversation's older cuts
/// before touching another conversation, even when its tool loop runs ahead.
pub(super) fn replacement(cache: &[Checkpoint], tokens: &[u32]) -> Option<usize> {
    if let Some((i, c)) = cache.iter().enumerate().find(|(_, c)| c.history == tokens) {
        return (!c.anchor).then_some(i);
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
        .filter(|(_, c)| !c.anchor && !(c.history.len() >= keep && covers(c, tokens)))
        .min_by_key(|(_, c)| {
            let old = c.history.len() < keep && covers(c, tokens);
            (
                if old {
                    0
                } else if c.history.is_empty() {
                    1
                } else {
                    2
                },
                if old {
                    c.history.len() as u64
                } else {
                    c.touched
                },
            )
        })
        .map(|(i, _)| i)
}

pub(super) fn mark_anchor(cache: &mut [Checkpoint], tokens: &[u32], upto: usize) -> Option<usize> {
    let cap = (cache.len() / 4).max(2).min(cache.len().saturating_sub(1));
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

impl Nemotron {
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
            .min_by_key(|(_, c)| (c.anchor, c.touched))
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
            assert!(cache.iter().filter(|c| c.anchor).count() <= 2);
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
