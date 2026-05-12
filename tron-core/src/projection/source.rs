use std::time::Instant;

use anyhow::Result;
use tron_api::{DepthProjectionMap, DepthSource, ProjectionMapSource};

pub struct DepthProjectionMapSource<P, D>
where
    P: DepthProjectionMap,
{
    projection: P,
    depth_source: D,
    latest_depth_mm: f64,
}

impl<P, D> DepthProjectionMapSource<P, D>
where
    P: DepthProjectionMap,
{
    pub fn new(projection: P, depth_source: D) -> Result<Self> {
        Ok(Self {
            projection,
            depth_source,
            latest_depth_mm: 0.0,
        })
    }

    pub fn latest_depth_mm(&self) -> f64 {
        self.latest_depth_mm
    }
}

pub struct StaticProjectionMapSource<M> {
    map: Option<M>,
}

impl<M> StaticProjectionMapSource<M> {
    pub fn new(map: M) -> Self {
        Self { map: Some(map) }
    }
}

#[async_trait::async_trait]
impl<M> ProjectionMapSource for StaticProjectionMapSource<M>
where
    M: Send,
{
    type Map = M;

    async fn next_map(&mut self, _timestamp: Instant) -> Result<Option<Self::Map>> {
        Ok(self.map.take())
    }
}

#[async_trait::async_trait]
impl<P, D> ProjectionMapSource for DepthProjectionMapSource<P, D>
where
    P: DepthProjectionMap + Send,
    P::Map: Send,
    D: DepthSource + Send,
{
    type Map = P::Map;

    async fn next_map(&mut self, timestamp: Instant) -> Result<Option<Self::Map>> {
        if let Some(sample) = self.depth_source.depth_at(timestamp).await? {
            if let Some(min_mm) = sample.min_mm.filter(|min_mm| *min_mm > 0) {
                self.latest_depth_mm = f64::from(min_mm);
            }
        }
        self.projection.map(self.latest_depth_mm).map(Some)
    }
}
