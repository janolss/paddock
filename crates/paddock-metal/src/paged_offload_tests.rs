use crate::*;
use paddock_engine::generator::Generator;
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires PADDOCK_METAL_PAGED_MODEL and PADDOCK_METAL_PAGED_FAMILY"]
fn paged_family_disk_restart_c4_all_logits() {
    let path = std::env::var("PADDOCK_METAL_PAGED_MODEL").unwrap();
    let family = std::env::var("PADDOCK_METAL_PAGED_FAMILY").unwrap();
    let dir = std::env::temp_dir().join(format!(
        "paddock-paged-parity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let cfg = KvOffloadConfig {
        ram_bytes: 2 << 30,
        disk: Some((dir.clone(), 2 << 30)),
        scope: b"paged-restart-test".to_vec(),
    };
    let load = || -> Box<dyn Generator> {
        let p = Path::new(&path);
        match family.as_str() {
            "granite" => {
                let mut m = Granite::load(p, 1024, 4, None).unwrap();
                m.enable_kv_offload(cfg.clone(), &[p]).unwrap();
                Box::new(m)
            }
            "gpt-oss" => {
                let mut m = GptOss::load(p, 1024, 4, None).unwrap();
                m.enable_kv_offload(cfg.clone(), &[p]).unwrap();
                Box::new(m)
            }
            "laguna" => {
                let mut m = Laguna::load(p, 1024, 4, None).unwrap();
                m.enable_kv_offload(cfg.clone(), &[p]).unwrap();
                Box::new(m)
            }
            _ => panic!("unknown paged test family"),
        }
    };
    let mut m = load();
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|s| (0..537).map(|i| 1000 + s * 2000 + i).collect())
        .collect();
    for (s, p) in prompts.iter().enumerate() {
        let started = Instant::now();
        m.forward_prefill(s, p).unwrap();
        m.tier_observe_prefill(p.len() as u32, started.elapsed().as_secs_f64() * 1e6);
    }
    m.reset();
    let mut reference = Vec::new();
    for (s, p) in prompts.iter().enumerate() {
        reference.push(m.forward_prefill(s, p).unwrap());
        assert_eq!(m.take_prefill_reused(s), 528);
    }
    let mut generation = Vec::new();
    for step in 0..16 {
        generation.push(
            m.forward_batch(&[9000 + step; 4], &[537 + step; 4])
                .unwrap(),
        );
    }
    let start = Instant::now();
    while m.tier_stats().unwrap().in_flight_demotes != 0 {
        assert!(start.elapsed() < Duration::from_secs(30));
        m.tier_pump();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(m.tier_stats().unwrap().io_failures, 0);
    drop(m);
    let mut m = load();
    let start = Instant::now();
    for (s, p) in prompts.iter().enumerate() {
        while m.tier_prefix_loading(s, p) {
            assert!(start.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            m.forward_prefill(s, p).unwrap(),
            reference[s],
            "all suffix logits, slot {s}"
        );
        assert_eq!(
            m.take_prefill_reused(s),
            528,
            "must restore, not silently recompute"
        );
    }
    for step in 0..16 {
        assert_eq!(
            m.forward_batch(&[9000 + step; 4], &[537 + step; 4])
                .unwrap(),
            generation[step as usize],
            "all c4 logits at step {step}"
        );
    }
    assert_eq!(m.tier_report().unwrap().decisions.served_from_nvme, 4);
    assert_eq!(m.tier_stats().unwrap().integrity_failures, 0);
    println!(
        "PAGED_RESTART family={family} c=4 reused=528 all_logits_steps=16 elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.
    );
    drop(m);
    // Only this unique test-owned directory, after both worker joins.
    std::fs::remove_dir_all(dir).unwrap();
}
