use crate::{FrameMeta, NoContext, Point2d, Point3d, Processor, Size};
use glam::{DMat3, DVec3};

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CheckerboardSpec {
    pub inner_corners: Size,
    pub square_size_mm: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CheckerboardDetection {
    pub spec: CheckerboardSpec,
    pub frame_size: Size,
    pub corners: Vec<Point2d>,
    pub score: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CalibrationFrameSide {
    pub frame_meta: FrameMeta,
    pub frame_size: Size,
    pub corners: Vec<Point2d>,
    pub score: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CheckerboardSample {
    pub spec: CheckerboardSpec,
    pub object_points: Vec<Point3d>,
    pub left: CalibrationFrameSide,
    pub right: CalibrationFrameSide,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CameraCalibration {
    pub image_size: Size,
    pub camera_matrix: DMat3,
    pub distortion: Vec<f64>,
    pub reprojection_error: f64,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CheckerboardStereoCalibration {
    pub spec: CheckerboardSpec,
    pub sample_count: usize,
    pub left: CameraCalibration,
    pub right: CameraCalibration,
    pub rotation: DMat3,
    pub translation: DVec3,
    pub essential: DMat3,
    pub fundamental: DMat3,
    pub stereo_reprojection_error: f64,
    pub per_view_errors: Vec<f64>,
}

pub type CheckerboardProcessor<I, C = NoContext> =
    dyn Processor<I, C, Output = Option<CheckerboardDetection>>;

pub type StereoCalibrationProcessor<I, C = NoContext> =
    dyn Processor<I, C, Output = Option<CheckerboardSample>>;
