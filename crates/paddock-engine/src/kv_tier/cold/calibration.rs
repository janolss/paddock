//! Best-effort cost metadata, not model state. Reads happen only while opening
//! the locked store; writes run on its bounded I/O worker, never in a GPU tick.
use super::super::cost::Calibration;
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const FILE: &str = "cost-model.txt";
const MAX_BYTES: u64 = 4096;
const SAVE_EVERY: Duration = Duration::from_secs(30);

pub(super) struct State {
    pub device: String,
    pub seed: Option<Calibration>,
    latest: Option<Calibration>,
    saved: Option<Calibration>,
    in_flight: Option<Calibration>,
    attempted: Option<Instant>,
}

impl State {
    pub fn load(dir: &Path, device: &str) -> Self {
        let seed = read(dir, device);
        Self {
            device: device.into(),
            seed,
            latest: None,
            saved: seed,
            in_flight: None,
            attempted: None,
        }
    }

    pub fn observe(&mut self, calibration: Calibration) {
        self.latest = Some(calibration);
    }

    pub fn next(&self, now: Instant) -> Option<Calibration> {
        if self.in_flight.is_some()
            || self.latest == self.saved
            || self
                .attempted
                .is_some_and(|at| now.duration_since(at) < SAVE_EVERY)
        {
            None
        } else {
            self.latest
        }
    }

    pub fn submitted(&mut self, now: Instant, calibration: Calibration) {
        self.in_flight = Some(calibration);
        self.attempted = Some(now);
    }

    pub fn complete(&mut self, calibration: Calibration, saved: bool) {
        self.in_flight = None;
        if saved {
            self.saved = Some(calibration);
        }
    }

    pub fn shutdown_value(&self) -> Option<Calibration> {
        self.latest
            .filter(|c| Some(*c) != self.saved && Some(*c) != self.in_flight)
    }
}

fn read(dir: &Path, device: &str) -> Option<Calibration> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(dir.join(FILE)).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES {
        return None;
    }
    let mut text = String::new();
    file.take(MAX_BYTES + 1).read_to_string(&mut text).ok()?;
    if text.len() as u64 > MAX_BYTES {
        return None;
    }
    Calibration::decode(&text, device)
}

pub(super) fn save(dir: &Path, device: &str, calibration: Calibration) -> io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temp = dir.join(format!(
        ".cost-model-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file: File = options.open(&temp)?;
    // Metadata can be discarded after a power failure: the bounded reader
    // rejects a torn file. No per-request fsync or model-state dependency.
    let result = file.write_all(calibration.encode(device).as_bytes());
    drop(file);
    let result = result.and_then(|()| std::fs::rename(&temp, dir.join(FILE)));
    if result.is_err() {
        // Only this successfully create_new-owned temporary file is removed.
        let _ = std::fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests;
