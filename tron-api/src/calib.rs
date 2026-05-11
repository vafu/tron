use crate::{Frame, FrameId, NoContext, Point2d, Point3d, Processor, Size, frame::FrameTimestamp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckerboardSpec {
    pub inner_corners: Size,
    pub square_size_mm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardDetection {
    pub spec: CheckerboardSpec,
    pub frame_size: Size,
    pub corners: Vec<Point2d>,
    pub score: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct CalibrationFrameSide<'a> {
    pub frame: Frame<'a>,
    pub detection: Option<&'a CheckerboardDetection>,
}

#[derive(Clone, Debug)]
pub struct CheckerboardSample {
    pub spec: CheckerboardSpec,
    pub object_points: Vec<Point3d>,
    pub left_corners: Vec<Point2d>,
    pub right_corners: Vec<Point2d>,
    pub left_frame_id: FrameId,
    pub right_frame_id: FrameId,
    pub left_frame_size: Size,
    pub right_frame_size: Size,
    pub left_timestamp: FrameTimestamp,
    pub right_timestamp: FrameTimestamp,
}

pub type CheckerboardProcessor<I, C = NoContext> =
    dyn Processor<I, C, Output = Option<CheckerboardDetection>>;

pub type StereoCalibrationProcessor<I, C = NoContext> =
    dyn Processor<I, C, Output = Option<CheckerboardSample>>;
