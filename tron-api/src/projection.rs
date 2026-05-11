use anyhow::Result;

pub trait DepthProjectionMap {
    type Map;

    fn map(&self, depth_mm: f64) -> Result<Self::Map>;
}

pub trait ProjectionMapSource {
    type Map;

    fn next_map(&mut self) -> Result<Self::Map>;
}

impl<F, M> ProjectionMapSource for F
where
    F: FnMut() -> Result<M>,
{
    type Map = M;

    fn next_map(&mut self) -> Result<Self::Map> {
        self()
    }
}
