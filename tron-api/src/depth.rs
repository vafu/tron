use std::time::Instant;

use anyhow::Result;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DepthSample {
    pub sequence: Option<u64>,
    pub sensor_timestamp_us: Option<u64>,
    pub printed_at_ms: Option<u64>,
    pub resolution: Option<u8>,
    pub center_mm: Option<u16>,
    pub min_mm: Option<u16>,
    pub max_mm: Option<u16>,
    pub valid_zones: Option<u8>,
    #[serde(serialize_with = "serialize_zones")]
    pub zones: [u16; 64],
    pub zone_count: usize,
    #[serde(skip)]
    pub sampled_at: Instant,
    #[serde(skip)]
    pub received_at: Instant,
}

fn serialize_zones<S>(zones: &[u16; 64], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serde::Serialize::serialize(zones.as_slice(), serializer)
}

#[async_trait::async_trait]
pub trait DepthSource {
    async fn depth_at(&mut self, timestamp: Instant) -> Result<Option<DepthSample>>;
}
