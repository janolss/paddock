//! CPU-only cold-load reservation. No GPU, weights copies or kernel compilation.
use super::*;

impl Gemma4 {
    pub fn diffusion_vision_budget() -> paddock_engine::generator::VisionBudget {
        vision::BUDGET
    }
    pub fn diffusion_residency_bytes(path: &Path, context: usize, slots: usize) -> Result<u64> {
        if !(1..=32768).contains(&context) || !(1..=8).contains(&slots) {
            return Err(MetalError::Model(
                "DiffusionGemma requires context 1..32768 and slots 1..8".into(),
            ));
        }
        let source_bytes = if path.is_dir() {
            let cfg =
                paddock_models::mlx::DiffusionConfig::read(path).map_err(MetalError::Model)?;
            if context > cfg.context {
                return Err(MetalError::Model("context exceeds the checkpoint".into()));
            }
            mlx::Source::open(path)?.bytes()
        } else {
            let map = paddock_models::mapped::MappedGguf::open(path)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if map.gguf().architecture() != Some("diffusion-gemma") {
                return Err(MetalError::Model(
                    "expected DiffusionGemma checkpoint".into(),
                ));
            }
            for (field, value) in [
                ("embedding_length", 2816),
                ("feed_forward_length", 2112),
                ("block_count", 30),
                ("attention.head_count", 16),
                ("expert_count", 128),
            ] {
                if map
                    .gguf()
                    .arch_field(field)
                    .and_then(paddock_models::gguf::Value::as_u64)
                    != Some(value)
                {
                    return Err(MetalError::Model(format!(
                        "DiffusionGemma requires {field}={value}"
                    )));
                }
            }
            map.total_len()
        };
        Ok(source_bytes
            + Self::diffusion_workspace_bytes(context, slots)
            + if path.is_dir() { 1 << 30 } else { 0 })
    }

    fn diffusion_workspace_bytes(context: usize, slots: usize) -> u64 {
        let pages = context.div_ceil(BLOCK_TOKENS);
        let ring = context.min(1024 + CHUNK).next_multiple_of(BLOCK_TOKENS);
        let global = pages * (slots * 2 + 1) * BLOCK_TOKENS * 1024 * 2;
        let sliding = ring * slots * 2 * 2048 * 2;
        let kv = (global * 5 * 2 + sliding * 25 * 2) as u64;
        // GGUF's larger shared scratch upper-bounds both native loaders.
        let gemm = (CHUNK * 8192).max((CHUNK + 32) * HEADS * 512) * 2;
        let scratch = (CHUNK * (2816 * 5 + HEADS * 512 * 2 + 4096 * 2 + 8192 * 2 + 9) * 4
            + slots * (pages + 262144 + HEADS * SPLITS * 514) * 4
            + gemm) as u64;
        kv + scratch
            + moe::Workspace::bytes(CHUNK)
            + diffusion::Lane::workspace_bytes(slots)
            + (64 << 20)
    }
}

/// Unified RAM is shared with other runners and applications, not a separate
/// VRAM pool. `required` is the resident set: the weight copy, KV, scratch and
/// the loader's fixed margin. The safetensors/GGUF mapping is file cache and
/// is already counted as reclaimable by the host sample.
pub(super) fn admit(required: u64, budget: Option<u64>) -> Result<()> {
    let host = paddock_engine::host_memory::sample()
        .ok_or_else(|| MetalError::Memory("cannot read physical memory availability".into()))?;
    let available = host.available.saturating_sub(1 << 30);
    paddock_engine::host_memory::check(required, available, budget).map_err(MetalError::Memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservation_grows_with_context_and_slots_without_gpu_allocation() {
        let one = Gemma4::diffusion_workspace_bytes(4096, 1);
        let four = Gemma4::diffusion_workspace_bytes(4096, 4);
        let long = Gemma4::diffusion_workspace_bytes(32768, 4);
        assert!(one > 1 << 30 && four > one && long > four);
        assert!(Gemma4::diffusion_residency_bytes(Path::new("/missing"), 32769, 1).is_err());
    }
}
