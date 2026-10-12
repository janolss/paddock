//! Metal-local row planning; the shared service still owns admission and FIFO
//! request order. No timers, hidden queues or customer tuning controls.

pub(crate) const MIXED_ROWS: usize = 128;

/// Never starve pending work if a wider decode batch already fills the usual
/// mixed grant. Keep the expanded grant tile-aligned and scratch-bounded.
pub(crate) fn row_cap(decodes: usize, capacity: usize) -> usize {
    if decodes == 0 {
        capacity
    } else {
        (decodes + 1)
            .next_multiple_of(32)
            .max(MIXED_ROWS)
            .min(capacity)
    }
}

/// Similar-sized cold prompts share compute in proportion to unfinished work.
/// This moves their completion ticks together, filling the decode batch instead
/// of making the first stream ride every later prefill. It trades the first
/// request's TTFT for fewer streaming stalls, so qualification measures both.
/// Short/asymmetric prompts retain FIFO; existing decodes always take priority.
pub(crate) fn grants(work: &[(usize, usize)], budget: usize, cold: bool) -> Vec<usize> {
    let total: usize = work.iter().map(|w| w.0).sum();
    let budget = budget.min(total);
    if total == budget {
        return work.iter().map(|w| w.0).collect();
    }
    let smallest = work.iter().map(|w| w.1).min().unwrap_or(0);
    let largest = work.iter().map(|w| w.1).max().unwrap_or(0);
    if !cold || work.len() < 2 || smallest < 256 || largest > smallest.saturating_mul(2) {
        let mut left = budget;
        return work
            .iter()
            .map(|w| {
                let n = w.0.min(left);
                left -= n;
                n
            })
            .collect();
    }
    balanced_grants(&work.iter().map(|w| w.0).collect::<Vec<_>>(), budget)
}

/// Checkpoint boundaries cap a grant, not a request's fair-share weight. Spend
/// leftovers on eligible peers without treating a nearby checkpoint as a short
/// prompt. Each redistribution either spends the budget or saturates a cap.
pub(crate) fn capped_grants(
    work: &[(usize, usize)],
    caps: &[usize],
    budget: usize,
    cold: bool,
) -> Vec<usize> {
    assert_eq!(work.len(), caps.len());
    let mut result = vec![0; work.len()];
    let mut left = budget;
    while left > 0 {
        let active: Vec<_> = (0..work.len())
            .filter(|&i| result[i] < caps[i].min(work[i].0))
            .collect();
        if active.is_empty() {
            break;
        }
        let remaining: Vec<_> = active
            .iter()
            .map(|&i| (work[i].0 - result[i], work[i].1))
            .collect();
        let proposed = grants(&remaining, left, cold);
        for (i, share) in active.into_iter().zip(proposed) {
            let n = share.min(caps[i] - result[i]);
            result[i] += n;
            left -= n;
        }
    }
    result
}

/// Work-conserving fair grants for already-admitted short resume tails. FIFO
/// completes one early, turning another prefill lane into a decode and leaving
/// fewer rows for its peers. Balancing keeps all tiny tails making progress.
pub(crate) fn balanced_grants(work: &[usize], budget: usize) -> Vec<usize> {
    let total: usize = work.iter().sum();
    let budget = budget.min(total);
    if budget == total {
        return work.to_vec();
    }
    // Largest-remainder apportionment spends exactly the caller's budget.
    // FIFO breaks equal fractional ties. No completed logits are held back.
    let mut shares: Vec<_> = work.iter().map(|w| w * budget / total).collect();
    let mut order: Vec<_> = (0..work.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(work[i] * budget % total));
    let extra = budget - shares.iter().sum::<usize>();
    for i in order.into_iter().take(extra) {
        shares[i] += 1;
    }
    shares
}

/// Apportion full recurrent/attention tiles before the ragged remainder.
/// The caller retains admission order and never withholds completed outputs.
#[cfg(test)]
pub(crate) fn tiled_grants(work: &[usize], budget: usize) -> Vec<usize> {
    let budget = budget.min(work.iter().sum());
    let tiles: Vec<_> = work.iter().map(|n| n / 32).collect();
    let mut shares = balanced_grants(&tiles, budget / 32)
        .into_iter()
        .map(|n| n * 32)
        .collect::<Vec<_>>();
    let remainder: Vec<_> = work.iter().zip(&shares).map(|(n, g)| n - g).collect();
    let extra = balanced_grants(&remainder, budget - shares.iter().sum::<usize>());
    for (share, extra) in shares.iter_mut().zip(extra) {
        *share += extra;
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoints_bound_work_without_distorting_fair_weights() {
        let work = [(7680, 8192), (8192, 8192), (8192, 8192), (8192, 8192)];
        assert_eq!(
            capped_grants(&work, &[7168, 512, 512, 512], 2048, true),
            [512; 4]
        );
        assert_eq!(
            capped_grants(&work, &[7168, 512, 512, 512], 128, false),
            [128, 0, 0, 0]
        );
        for cold in [false, true] {
            for a in 0..9 {
                for b in 0..9 {
                    let work = [(a, 512), (b, 512), (7, 512)];
                    for cap in 0..10 {
                        let caps = [cap, 3, 5];
                        for budget in 0..30 {
                            let got = capped_grants(&work, &caps, budget, cold);
                            let available: usize =
                                work.iter().zip(caps).map(|(w, c)| w.0.min(c)).sum();
                            assert_eq!(got.iter().sum::<usize>(), budget.min(available));
                            assert!(
                                got.iter()
                                    .zip(work.iter().zip(caps))
                                    .all(|(g, (w, c))| *g <= w.0.min(c))
                            );
                        }
                    }
                    assert_eq!(
                        capped_grants(&work, &[a, b, 7], 8, cold),
                        grants(&work, 8, cold)
                    );
                }
            }
        }
        assert!(capped_grants(&[], &[], 2048, true).is_empty());
    }
    #[test]
    fn tiled_diagnostic_grants_conserve_bounded_work() {
        for work in [
            vec![],
            vec![0, 0],
            vec![1, 31, 32, 33],
            vec![480, 512, 512, 512],
            vec![997, 1025, 2048],
        ] {
            let total: usize = work.iter().sum();
            for budget in 0..=total + 1 {
                let grants = tiled_grants(&work, budget);
                assert_eq!(grants.len(), work.len());
                assert_eq!(grants.iter().sum::<usize>(), budget.min(total));
                assert!(grants.iter().zip(&work).all(|(g, n)| g <= n));
            }
        }
        assert_eq!(tiled_grants(&[480, 512, 512, 512], 512), [128; 4]);
    }
    #[test]
    fn tiny_resume_peers_finish_without_fifo_decode_crowding() {
        let mut work = vec![12, 12, 12];
        for _ in 0..12 {
            let grants = balanced_grants(&work, 3);
            assert_eq!(grants, [1, 1, 1]);
            for (n, g) in work.iter_mut().zip(grants) {
                *n -= g;
            }
        }
        assert_eq!(work, [0, 0, 0]);
        for budget in 0..32 {
            let work = [0, 1, 5, 12];
            let grants = balanced_grants(&work, budget);
            assert_eq!(grants.iter().sum::<usize>(), budget.min(18));
            assert!(grants.iter().zip(work).all(|(g, n)| *g <= n));
        }
        assert!(balanced_grants(&[], 4).is_empty());
    }
    #[test]
    fn cold_cohorts_conserve_work_and_finish_without_holding_outputs() {
        let mut work = vec![(295, 807), (807, 807), (810, 810), (803, 803)];
        let mut ticks = 0;
        while work.iter().any(|w| w.0 != 0) {
            let before: usize = work.iter().map(|w| w.0).sum();
            let ns = grants(&work, 512, true);
            assert_eq!(ns.iter().sum::<usize>(), before.min(512));
            for (w, n) in work.iter_mut().zip(ns) {
                assert!(n <= w.0);
                w.0 -= n;
            }
            ticks += 1;
            assert!(ticks <= 6);
        }
        assert_eq!(ticks, 6);
    }
    #[test]
    fn decode_priority_short_prompts_and_small_budgets() {
        assert_eq!(row_cap(0, 512), 512);
        assert_eq!(row_cap(3, 512), 128);
        assert_eq!(row_cap(127, 512), 128);
        assert_eq!(row_cap(128, 512), 160);
        assert_eq!(row_cap(511, 512), 512);
        let long = [(700, 800), (800, 800)];
        assert_eq!(grants(&long, 125, false), [125, 0]);
        assert_eq!(grants(&[(20, 20), (800, 800)], 128, true), [20, 108]);
        assert_eq!(grants(&long, 0, true), [0, 0]);
        assert_eq!(grants(&long, 1, true), [0, 1]);
        assert!(grants(&[], 512, true).is_empty());
        for budget in 0..1600 {
            let ns = grants(&long, budget, true);
            assert_eq!(ns.iter().sum::<usize>(), budget.min(1500));
            assert!(ns[0] <= 700 && ns[1] <= 800);
        }
    }
}
