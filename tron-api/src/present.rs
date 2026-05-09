use anyhow::Result;

use crate::frame::Frame;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoContext;

#[derive(Clone, Copy, Debug)]
pub struct NamedFrame<'a> {
    pub name: &'static str,
    pub frame: Frame<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameViewModel<'a, M = NoContext> {
    pub frames: &'a [NamedFrame<'a>],
    pub metadata: M,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub acquire_us: u64,
}

pub trait Presenter<V> {
    fn present(&mut self, view: V) -> Result<()>;
}
