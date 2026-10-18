//! The one GPU memory ceiling shared by the manager's admission and the Metal
//! runner's device grant, so the two can never disagree.

/// Left below Apple's `recommendedMaxWorkingSetSize` for the driver, shader
/// compiler and transient allocations outside the buffer ledger.
pub const DRIVER_RESERVE: u64 = 2 << 30;

/// Largest working set any runner may be granted. `recommended` is already
/// Apple's performance line, so no percentage is shaved off it.
pub const fn gpu_ceiling(recommended: u64) -> u64 {
    recommended.saturating_sub(DRIVER_RESERVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_is_recommendation_minus_reserve() {
        assert_eq!(gpu_ceiling(10 << 30), 8 << 30);
        assert_eq!(gpu_ceiling(1 << 30), 0);
    }
}
