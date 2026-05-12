use std::time::Instant;

use anyhow::Result;

#[derive(Clone, Copy, Debug)]
pub struct DepthSample {
    pub sequence: Option<u64>,
    pub sensor_timestamp_us: Option<u64>,
    pub printed_at_ms: Option<u64>,
    pub resolution: Option<u8>,
    pub center_mm: Option<u16>,
    pub min_mm: Option<u16>,
    pub max_mm: Option<u16>,
    pub valid_zones: Option<u8>,
    pub zones: [u16; 64],
    pub zone_count: usize,
    pub sampled_at: Instant,
    pub received_at: Instant,
}

#[async_trait::async_trait]
pub trait DepthSource {
    async fn depth_at(&mut self, timestamp: Instant) -> Result<Option<DepthSample>>;
}
