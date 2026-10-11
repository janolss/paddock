//! The reasoning stop mask (`stop_mask_rows`, slot 850): active rows get
//! their stop logits set to -inf in place and a pressure flag (a stop within
//! ln 2 of the row max after temperature, or at the max for a greedy row);
//! inactive rows and every other logit are untouched, and a greedy draw
//! after the mask never lands on a stop. Light (no model load).

mod common;

use paddock_engine::gpu::GpuExecutor;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn exec() -> Option<(std::sync::MutexGuard<'static, ()>, GpuExecutor)> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let guard = LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    common::gpu().map(|e| (guard, e))
}

/// `{active, inv_t bits, n_stops, stop0..stop3, pad}` for one row.
fn par_row(active: bool, inv_t: f32, stops: &[u32]) -> [u32; 8] {
    let mut p = [0u32; 8];
    p[0] = active as u32;
    p[1] = inv_t.to_bits();
    p[2] = stops.len() as u32;
    p[3..3 + stops.len()].copy_from_slice(stops);
    p
}

#[test]
fn active_rows_mask_their_stops_and_flag_pressure() {
    let Some((_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_stop_mask_rows() {
        eprintln!("pack lacks stop_mask_rows - skipping");
        return;
    }
    let (rows, n) = (6usize, 50_000 + 7usize); // an odd vocab: rows off the 16-byte grid
    let (eos, eot) = (n as u32 - 274, n as u32 - 276);
    let mut logits = det(rows * n, 11);
    let max_at = |l: &mut [f32], r: usize, v: f32| l[r * n + 4321] = v;
    // row 0: inactive, its stop on top - untouched, flag 0
    max_at(&mut logits, 0, 9.0);
    logits[eos as usize] = 9.5;
    // row 1: greedy, the stop AT the max - pressure
    max_at(&mut logits, 1, 9.0);
    logits[n + eos as usize] = 9.0;
    // row 2: T = 1, the stop 0.5 under the max - pressure (p_stop > p_top / 2)
    max_at(&mut logits, 2, 9.0);
    logits[2 * n + eot as usize] = 8.5;
    // row 3: T = 1, the stop 2 under - no pressure
    max_at(&mut logits, 3, 9.0);
    logits[3 * n + eos as usize] = 7.0;
    // row 4: T = 0.5 (inv_t 2), the stop 0.5 under - scaled to 1.0 under: no pressure
    max_at(&mut logits, 4, 9.0);
    logits[4 * n + eos as usize] = 8.5;
    // row 5: greedy, the stop below the max - no pressure, still masked
    max_at(&mut logits, 5, 9.0);
    logits[5 * n + eos as usize] = 3.0;
    let stops = [eos, eot];
    let mut par = Vec::new();
    for (r, inv_t) in [0.0f32, 0.0, 1.0, 1.0, 2.0, 0.0].into_iter().enumerate() {
        par.extend_from_slice(&par_row(r != 0, inv_t, &stops));
    }
    let mut d_l = exec.to_device(&logits).expect("logits");
    let d_p = exec.to_device_u32(&par).expect("par");
    let mut d_f = exec.to_device_u32(&vec![u32::MAX; rows]).expect("flags");
    exec.stop_mask_rows(&mut d_l, 0, &d_p, 0, &mut d_f, rows, n)
        .expect("stop mask");
    let flags = exec.to_host_u32(&d_f).expect("flags dtoh");
    assert_eq!(flags, vec![0, 1, 1, 0, 0, 0], "pressure flags");
    let got = exec.to_host(&d_l).expect("logits dtoh");
    for r in 0..rows {
        for i in 0..n {
            let (g, w) = (got[r * n + i], logits[r * n + i]);
            if r != 0 && stops.contains(&(i as u32)) {
                assert_eq!(g, f32::NEG_INFINITY, "row {r} stop {i} masked");
            } else {
                assert_eq!(g.to_bits(), w.to_bits(), "row {r} logit {i} untouched");
            }
        }
    }
    // the draw that follows is the masked row's: greedy lands on the old max
    let greedy: Vec<u32> = (0..rows).flat_map(|_| [0u32, 0, 1, 0]).collect();
    let d_gp = exec.to_device_u32(&greedy).expect("greedy par");
    let mut d_out = exec.to_device_u32(&vec![u32::MAX; rows]).expect("out");
    exec.sample_rows(&d_l, &d_gp, &mut d_out, rows, n)
        .expect("sample");
    let ids = exec.to_host_u32(&d_out).expect("ids dtoh");
    assert_eq!(ids[0], eos, "the inactive row still draws its stop");
    for (r, &id) in ids.iter().enumerate().skip(1) {
        assert_eq!(id, 4321, "row {r} draws the top non-stop token");
    }
}
