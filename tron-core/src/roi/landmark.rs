use anyhow::Result;
use tron_api::{NoContext, Processor, RoiResult, Size};

use crate::roi::mediapipe::HandLandmarks;

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiInput<'a> {
    pub landmarks: Option<&'a HandLandmarks>,
    pub frame_size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiProcessor {}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkTrackingRoiProcessor {}

impl LandmarkRoiProcessor {
    pub fn new() -> Self {
        Self {}
    }
}

impl LandmarkTrackingRoiProcessor {
    pub fn new() -> Self {
        Self {}
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
            .and_then(|landmarks| landmarks.bounding_roi(input.frame_size)))
    }
}

impl Processor<LandmarkRoiInput<'_>, NoContext> for LandmarkTrackingRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(
        &mut self,
        input: LandmarkRoiInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        Ok(input
            .landmarks
            .and_then(|landmarks| landmarks.tracking_roi(input.frame_size)))
    }
}
