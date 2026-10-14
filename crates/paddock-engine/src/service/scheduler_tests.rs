//! Scheduler tests through `run_batched` and its failure paths: admission
//! against cache-only pool pages, and a failed tick's chunked prefills.

use super::*;

struct CachedPool {
    waves: Vec<usize>,
    reclaimable: usize,
}
impl Generator for CachedPool {
    fn reset(&mut self) {}
    fn vocab(&self) -> usize {
        2
    }
    fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
        Ok(vec![0., 1.])
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(0)
    }
    fn pool_admission_blocks(&self) -> Option<usize> {
        Some(self.reclaimable)
    }
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        self.waves.push(items.len());
        Ok(items.iter().map(|_| vec![0., 1.]).collect())
    }
}

#[test]
fn cache_only_pages_allow_concurrent_admission_but_active_pages_do_not() {
    for (reclaimable, expected) in [(128, vec![2]), (0, vec![1, 1])] {
        let (send, requests) = std::sync::mpsc::channel();
        let mut receivers = Vec::new();
        for _ in 0..2 {
            let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
            receivers.push(receiver);
            send.send(GenRequest {
                prompt: vec![0; 16],
                max_tokens: 1,
                sampler: SamplingParams::default(),
                stop_tokens: vec![],
                events,
                mm_chunks: None,
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
                user_turn: false,
            })
            .unwrap();
        }
        drop(send);
        let mut generator = CachedPool {
            waves: Vec::new(),
            reclaimable,
        };
        run_batched(
            &mut generator,
            &requests,
            2,
            2,
            &Arc::new(EngineMetrics::default()),
            &ShutdownCtl::new(),
        );
        assert_eq!(generator.waves, expected);
        assert_eq!(
            generator.pool_free_blocks(),
            Some(0),
            "do not falsify occupancy"
        );
        for mut receiver in receivers {
            let mut completed = false;
            while let Ok(event) = receiver.try_recv() {
                assert!(!matches!(event, TokenEvent::Error(_)));
                completed |= matches!(event, TokenEvent::Done(..));
            }
            assert!(completed);
        }
    }
}

/// Records the slots whose chunked prefill the scheduler dropped.
struct Aborts(Vec<usize>);
impl Generator for Aborts {
    fn reset(&mut self) {}
    fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
        panic!("no inference")
    }
    fn vocab(&self) -> usize {
        2
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.0.push(slot);
        true
    }
}

#[test]
fn failing_stuck_chunks_drops_them_in_the_backend_too() {
    let mut receivers = Vec::new();
    let mut slots: Vec<Option<Slot>> = (0..3)
        .map(|_| {
            let (events, rx) = tokio::sync::mpsc::unbounded_channel();
            receivers.push(rx);
            Some(Slot::new(
                GenRequest {
                    prompt: vec![1, 2],
                    max_tokens: 8,
                    sampler: SamplingParams::default(),
                    stop_tokens: Vec::new(),
                    events,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                    user_turn: false,
                },
                Arc::new(EngineMetrics::default()),
            ))
        })
        .collect();
    slots[2] = None;
    // every live slot is chunking, so there is no decode row to preempt
    let mut chunking: std::collections::HashSet<usize> = [0, 1].into();
    let mut preempted = Vec::new();
    let mut g = Aborts(Vec::new());
    preempt_or_fail_mixed(&mut g, &mut slots, &mut preempted, &mut chunking, false);
    g.0.sort_unstable();
    // the backend's queue must lose them too: left there, the next prompt
    // on either slot is refused as "already has a chunked prefill"
    assert_eq!(g.0, [0, 1]);
    assert!(chunking.is_empty());
    assert!(preempted.is_empty());
    assert!(slots.iter().all(Option::is_none));
    for rx in &mut receivers[..2] {
        assert!(matches!(rx.try_recv(), Ok(TokenEvent::Error(_))));
    }
}
