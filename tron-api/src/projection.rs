use std::time::Instant;

use anyhow::Result;

pub trait DepthProjectionMap {
    type Map;

    fn map(&self, depth_mm: f64) -> Result<Self::Map>;
}

#[async_trait::async_trait]
pub trait ProjectionMapSource {
    type Map;

    async fn next_map(&mut self, timestamp: Instant) -> Result<Self::Map>;
}

#[async_trait::async_trait]
impl<F, M> ProjectionMapSource for F
where
    F: FnMut(Instant) -> Result<M> + Send,
    M: Send,
{
    type Map = M;

    async fn next_map(&mut self, timestamp: Instant) -> Result<Self::Map> {
        self(timestamp)
    }
}
