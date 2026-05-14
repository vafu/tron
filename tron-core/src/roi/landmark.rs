use anyhow::Result;
use tron_api::{NoContext, Processor, RoiResult, Size};

use crate::roi::mediapipe::HandLandmarks;

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiInput<'a> {
    pub landmarks: Option<&'a HandLandmarks>,
    pub frame_size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiProcessor {
    scale: f32,
}

impl LandmarkRoiProcessor {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl Processor<LandmarkRoiInput<'_>, NoContext> for LandmarkRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(
        &mut self,
        input: LandmarkRoiInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        Ok(input
            .landmarks
            .and_then(|landmarks| landmarks.bounding_roi(input.frame_size, self.scale)))
    }
}
