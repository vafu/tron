use anyhow::{Context, Result};
use glam::{DMat3, DVec3};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Vector};
use opencv::prelude::*;
use tron_api::{CheckerboardStereoCalibration, DepthPointProjection, DepthProjectionMap, Point2d};

use super::FrameProjectionMap;

#[derive(Clone, Debug)]
pub struct CheckerboardDepthProjection {
    calibration: CheckerboardStereoCalibration,
}

impl CheckerboardDepthProjection {
    pub fn new(calibration: CheckerboardStereoCalibration) -> Self {
        Self { calibration }
    }
}

impl DepthProjectionMap for CheckerboardDepthProjection {
    type Map = FrameProjectionMap;

    fn map(&self, depth_mm: f64) -> Result<FrameProjectionMap> {
        anyhow::ensure!(depth_mm >= 0.0, "projection depth must be non-negative");
        build_projection_map(&self.calibration, depth_mm)
    }
}

impl DepthPointProjection for CheckerboardDepthProjection {
    fn project_points(&self, depth_mm: f64, points: &[Point2d]) -> Result<Vec<Option<Point2d>>> {
        anyhow::ensure!(depth_mm >= 0.0, "projection depth must be non-negative");
        project_points_at_depth(&self.calibration, depth_mm, points)
    }
}

fn build_projection_map(
    calibration: &CheckerboardStereoCalibration,
    depth_mm: f64,
) -> Result<FrameProjectionMap> {
    let output_size = calibration.left.image_size;
    let input_size = calibration.right.image_size;
    let pixel_count = output_size.width as usize * output_size.height as usize;

    let mut left_pixels = Vector::<Point2f>::with_capacity(pixel_count);
    for y in 0..output_size.height {
        for x in 0..output_size.width {
            left_pixels.push(Point2f::new(x as f32, y as f32));
        }
    }

    let projected = project_left_pixels(calibration, depth_mm, &left_pixels)?;
    let mut pixels = Vec::with_capacity(pixel_count);
    for point in projected {
        let x = point.x as f64;
        let y = point.y as f64;
        if pixel_center_in_size(x, y, input_size.width as f64, input_size.height as f64) {
            pixels.push(Some((x as u32, y as u32)));
        } else {
            pixels.push(None);
        }
    }

    Ok(FrameProjectionMap {
        input_size,
        output_size,
        pixels,
    })
}

fn project_points_at_depth(
    calibration: &CheckerboardStereoCalibration,
    depth_mm: f64,
    points: &[Point2d],
) -> Result<Vec<Option<Point2d>>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    let mut left_pixels = Vector::<Point2f>::with_capacity(points.len());
    for point in points {
        left_pixels.push(Point2f::new(point.x as f32, point.y as f32));
    }

    let projected = project_left_pixels(calibration, depth_mm, &left_pixels)?;
    let right_size = calibration.right.image_size;
    Ok(projected
        .into_iter()
        .map(|point| {
            let x = point.x as f64;
            let y = point.y as f64;
            point_in_size(x, y, right_size.width as f64, right_size.height as f64)
                .then(|| Point2d::new(x, y))
        })
        .collect())
}

fn project_left_pixels(
    calibration: &CheckerboardStereoCalibration,
    depth_mm: f64,
    left_pixels: &Vector<Point2f>,
) -> Result<Vector<Point2f>> {
    let left_camera = mat3(calibration.left.camera_matrix)?;
    let left_dist = mat_vec(&calibration.left.distortion)?;
    let mut normalized = Vector::<Point2f>::new();
    calib3d::undistort_points_def(&left_pixels, &mut normalized, &left_camera, &left_dist)
        .context("undistort left-frame pixels")?;

    let mut object_points = Vector::<Point3f>::with_capacity(left_pixels.len());
    for point in normalized {
        object_points.push(Point3f::new(
            point.x * depth_mm as f32,
            point.y * depth_mm as f32,
            depth_mm as f32,
        ));
    }

    let rotation = mat3(calibration.rotation)?;
    let mut rvec = Mat::default();
    calib3d::rodrigues_def(&rotation, &mut rvec).context("convert stereo rotation to rvec")?;
    let tvec = mat_vec3(calibration.translation)?;
    let right_camera = mat3(calibration.right.camera_matrix)?;
    let right_dist = mat_vec(&calibration.right.distortion)?;
    let mut projected = Vector::<Point2f>::new();
    calib3d::project_points_def(
        &object_points,
        &rvec,
        &tvec,
        &right_camera,
        &right_dist,
        &mut projected,
    )
    .context("project left depth plane into right frame")?;

    Ok(projected)
}

fn pixel_center_in_size(x: f64, y: f64, width: f64, height: f64) -> bool {
    (0.0..width).contains(&x) && (0.0..height).contains(&y)
}

fn point_in_size(x: f64, y: f64, width: f64, height: f64) -> bool {
    (0.0..=width).contains(&x) && (0.0..=height).contains(&y)
}

fn mat3(values: DMat3) -> Result<Mat> {
    let rows = [
        [values.x_axis.x, values.y_axis.x, values.z_axis.x],
        [values.x_axis.y, values.y_axis.y, values.z_axis.y],
        [values.x_axis.z, values.y_axis.z, values.z_axis.z],
    ];
    let mat = Mat::from_slice_2d(&rows).context("create OpenCV 3x3 matrix")?;
    mat.try_clone().context("clone OpenCV 3x3 matrix")
}

fn mat_vec3(values: DVec3) -> Result<Mat> {
    let values = values.to_array();
    mat_vec(&values)
}

fn mat_vec(values: &[f64]) -> Result<Mat> {
    let mat = Mat::from_slice(values).context("create OpenCV vector")?;
    mat.try_clone().context("clone OpenCV vector")
}
