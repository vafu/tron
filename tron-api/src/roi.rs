use crate::{NoContext, Processor, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoiCandidate {
    pub rect: Rect,
    pub area: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoiResult {
    pub rect: Rect,
}

pub trait RoiProcessor<I, C = NoContext>: Processor<I, C, Output = Option<RoiResult>> {}

impl<T, I, C> RoiProcessor<I, C> for T where T: Processor<I, C, Output = Option<RoiResult>> {}
