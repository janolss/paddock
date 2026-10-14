//! Error-correcting output codes: classification past the model's ten
//! outputs.
//!
//! The scheme is SDM's `sdm.models.ecoc.ECOC` (NVIDIA structured-data-models,
//! pinned revision in `recipe.rs`), itself the many-class extension from
//! TabPFN Extensions (Prior Labs, Apache-2.0). Each of T tasks gives nine of
//! the K classes their own output (0-8) and merges every other class into
//! the rest output (9); the model runs once per task with the context labels
//! mapped through that task's code. A class's score is its mean log
//! probability over the tasks where it was separate - softmax normalizes the
//! scores into probabilities, as upstream says to.
//!
//! What is ours: the draws. Upstream samples the codebook with
//! `torch.rand` on a `torch.Generator`, and torch's streams are not portable
//! across devices or versions (the recipe's member draws already replace
//! them for the same reason). Here every uniform comes from SplitMix64, 24
//! mantissa bits like a float32 `torch.rand`, consumed in upstream's order:
//! per code, the [draws x K] priority block, then the [draws x 9] symbol
//! block. The selection itself - least-covered classes first, a random
//! symbol permutation, best of 50 candidates by minimum then total pairwise
//! Hamming distance - is upstream's line for line.

/// Outputs the Kumo checkpoints carry.
pub const MODEL_CLASSES: usize = 10;
/// The merged-classes output of every task.
pub const REST: u8 = (MODEL_CLASSES - 1) as u8;
/// Largest class count served. Cost grows as tasks x estimators model passes
/// (256 classes: 29 tasks a member); the context's 4096 rows bound it anyway.
pub const MAX_CLASSES: usize = 256;

/// Seeds the codebook stream apart from the recipe's member draws (XOR
/// with the request seed), so a table of ten classes or fewer draws exactly
/// what it always drew.
pub const STREAM: u64 = 0xEC0C_C0DE_B00C_0001;

/// Tasks a codebook for `classes` (> [`MODEL_CLASSES`]) has: enough for every
/// class to be separate once, and at least 4 x ceil(log_10 K) for the
/// error-correcting distance.
pub fn tasks(classes: usize) -> usize {
    let rest = MODEL_CLASSES - 1;
    let log = (classes as f64).ln() / (MODEL_CLASSES as f64).ln();
    classes.div_ceil(rest).max(4 * log.ceil() as usize)
}

/// The portable uniform stream (see the module notes).
pub struct Draws(u64);

impl Draws {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// A float32 in [0, 1) on the 2^-24 grid.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}

/// Indices that sort `v` ascending; ties (measure zero for these draws) keep
/// index order.
fn argsort(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx
}

/// One codebook, `[tasks][classes]`, entries in 0..=[`REST`].
pub fn draw(classes: usize, d: &mut Draws) -> Vec<Vec<u8>> {
    let rest = MODEL_CLASSES - 1;
    let t = tasks(classes);
    // bound the quadratic distance search for large targets
    let candidates = if classes <= 200 { 50 } else { 1 };
    let mut book: Vec<Vec<Vec<u8>>> = vec![Vec::with_capacity(t); candidates];
    let mut coverage = vec![vec![0f32; classes]; candidates];
    for _ in 0..t {
        let priority: Vec<Vec<f32>> = (0..candidates)
            .map(|c| {
                (0..classes)
                    .map(|k| coverage[c][k] + 0.1 * d.uniform())
                    .collect()
            })
            .collect();
        let symbols: Vec<Vec<f32>> = (0..candidates)
            .map(|_| (0..rest).map(|_| d.uniform()).collect())
            .collect();
        for c in 0..candidates {
            let chosen = argsort(&priority[c]);
            let perm = argsort(&symbols[c]);
            let mut code = vec![REST; classes];
            for (j, &k) in chosen.iter().take(rest).enumerate() {
                code[k] = perm[j] as u8;
                coverage[c][k] += 1.0;
            }
            book[c].push(code);
        }
    }
    if candidates == 1 {
        return book.pop().expect("one candidate");
    }
    // prefer the largest minimum distance, ties broken by total distance
    // (torch.argmax keeps the first maximum)
    let pairs = (classes * (classes - 1) / 2) as u64;
    let mut best = (0usize, 0u64);
    for (c, b) in book.iter().enumerate() {
        let (mut min, mut sum) = (u64::MAX, 0u64);
        for a in 0..classes {
            for z in a + 1..classes {
                let dist = b.iter().filter(|row| row[a] != row[z]).count() as u64;
                min = min.min(dist);
                sum += dist;
            }
        }
        let score = min * (pairs * t as u64 + 1) + sum;
        if c == 0 || score > best.1 {
            best = (c, score);
        }
    }
    book.swap_remove(best.0)
}

/// Class scores `[rows][classes]` from one codebook's task outputs, each
/// `[rows][MODEL_CLASSES]` logits: the log-softmax of the class's own output,
/// averaged over the tasks where it is separate.
pub fn combine(outputs: &[&[f32]], book: &[Vec<u8>], rows: usize) -> Vec<f32> {
    debug_assert_eq!(outputs.len(), book.len());
    let classes = book.first().map_or(0, Vec::len);
    let mut sum = vec![0f64; rows * classes];
    for (out, code) in outputs.iter().zip(book) {
        for (r, logits) in out
            .as_chunks::<MODEL_CLASSES>()
            .0
            .iter()
            .take(rows)
            .enumerate()
        {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = f64::from(max)
                + logits
                    .iter()
                    .map(|&x| f64::from(x - max).exp())
                    .sum::<f64>()
                    .ln();
            for (k, &sym) in code.iter().enumerate() {
                if sym != REST {
                    sum[r * classes + k] += f64::from(logits[sym as usize]) - lse;
                }
            }
        }
    }
    let active: Vec<usize> = (0..classes)
        .map(|k| book.iter().filter(|code| code[k] != REST).count())
        .collect();
    sum.iter()
        .enumerate()
        .map(|(i, s)| (s / active[i % classes] as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_counts_follow_upstream() {
        // max(ceil(K/9), 4 ceil(log_10 K))
        assert_eq!(tasks(11), 8);
        assert_eq!(tasks(36), 8);
        assert_eq!(tasks(100), 12);
        assert_eq!(tasks(101), 12);
        assert_eq!(tasks(256), 29);
    }

    #[test]
    fn every_class_is_separate_somewhere_and_codes_are_permutations() {
        for k in [11, 17, 40, 201] {
            let book = draw(k, &mut Draws::new(7));
            assert_eq!(book.len(), tasks(k));
            for code in &book {
                // nine classes carry the nine own symbols, once each
                let mut own: Vec<u8> = code.iter().copied().filter(|&s| s != REST).collect();
                own.sort_unstable();
                assert_eq!(own, (0..REST).collect::<Vec<_>>());
            }
            for c in 0..k {
                assert!(
                    book.iter().any(|code| code[c] != REST),
                    "class {c} never separate"
                );
            }
        }
    }

    #[test]
    fn draws_are_seeded() {
        assert_eq!(draw(23, &mut Draws::new(3)), draw(23, &mut Draws::new(3)));
        assert_ne!(draw(23, &mut Draws::new(3)), draw(23, &mut Draws::new(4)));
    }

    #[test]
    fn combine_averages_own_log_probabilities() {
        // two tasks over 11 classes, one row
        let mut a = vec![REST; 11];
        let mut b = vec![REST; 11];
        for k in 0..9 {
            a[k] = k as u8;
            b[k + 2] = k as u8;
        }
        let la: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let lb: Vec<f32> = (0..10).map(|i| -(i as f32) * 0.2).collect();
        let lsm = |l: &[f32], i: usize| {
            let lse = l.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
            f64::from(l[i]) - lse
        };
        let out = combine(&[&la, &lb], &[a.clone(), b.clone()], 1);
        for k in 0..11 {
            let mut s = 0.;
            let mut n = 0.;
            if a[k] != REST {
                s += lsm(&la, a[k] as usize);
                n += 1.;
            }
            if b[k] != REST {
                s += lsm(&lb, b[k] as usize);
                n += 1.;
            }
            assert!((f64::from(out[k]) - s / n).abs() < 1e-6, "class {k}");
        }
    }
}
