use serde::Serialize;
use tron_api::{DepthSample, Frame, Rect, RoiResult};
use tron_core::projection::HandProjectionOutput;
use tron_core::roi::mediapipe::HandLandmarks;

#[derive(Debug, Serialize)]
pub struct Aggregate<'a> {
    #[serde(skip)]
    pub rgb: Frame<'a>,
    #[serde(skip)]
    pub ir: Frame<'a>,
    pub sync_delta_us: i64,
    pub palm_roi: Option<RoiResult>,
    pub landmarks: Option<HandLandmarks>,
    pub rgb_roi: Option<RoiResult>,
    pub camera_roi: Option<Rect>,
    #[allow(dead_code)]
    pub depth_sample: Option<DepthSample>,
    pub projection: Option<HandProjectionOutput>,
}

impl Aggregate<'_> {
    pub fn pair_id(&self) -> (u64, u64) {
        (self.rgb.meta.id, self.ir.meta.id)
    }
}
