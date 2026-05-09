use crate::calib::checkerboard::CheckerboardSample;
use anyhow::{Result, bail};
use opencv::calib3d;
use opencv::core::{
    self, Mat, MatExprTraitConst, Point2f, Point3f, Size, TermCriteria, TermCriteria_Type, Vector,
};
use opencv::prelude::*;

const MIN_STEREO_SAMPLES: usize = 3;

#[derive(Clone, Debug)]
pub struct StereoCalibrationSession {
    pattern: (i32, i32),
    square_size: f32,
    rgb_size: (u32, u32),
    ir_size: (u32, u32),
    samples: Vec<CheckerboardSample>,
}

#[derive(Clone, Debug)]
pub struct StereoCalibrationResult {
    pub sample_count: usize,
    pub corner_count: usize,
    pub rgb_rms: f64,
    pub ir_rms: f64,
    pub stereo_rms: f64,
    pub rgb_camera_matrix: Mat,
    pub rgb_dist_coeffs: Mat,
    pub ir_camera_matrix: Mat,
    pub ir_dist_coeffs: Mat,
    pub rotation: Mat,
    pub translation: Mat,
    pub essential: Mat,
    pub fundamental: Mat,
    pub rgb_per_view_errors: Mat,
    pub ir_per_view_errors: Mat,
    pub stereo_per_view_errors: Mat,
}

impl StereoCalibrationSession {
    pub fn new(
        pattern: (i32, i32),
        square_size: f32,
        rgb_size: (u32, u32),
        ir_size: (u32, u32),
    ) -> Self {
        Self {
            pattern,
            square_size,
            rgb_size,
            ir_size,
            samples: Vec::new(),
        }
    }

    pub fn reset(
        &mut self,
        pattern: (i32, i32),
        square_size: f32,
        rgb_size: (u32, u32),
        ir_size: (u32, u32),
    ) {
        *self = Self::new(pattern, square_size, rgb_size, ir_size);
    }

    pub fn push(&mut self, sample: CheckerboardSample) -> Result<usize> {
        if sample.pattern != self.pattern {
            bail!(
                "sample pattern {:?} does not match session pattern {:?}",
                sample.pattern,
                self.pattern
            );
        }
        self.samples.push(sample);
        Ok(self.samples.len())
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn solve(&self) -> Result<StereoCalibrationResult> {
        if self.samples.len() < MIN_STEREO_SAMPLES {
            bail!(
                "need at least {MIN_STEREO_SAMPLES} checkerboard samples, have {}",
                self.samples.len()
            );
        }

        let object_template = object_points(self.pattern, self.square_size);
        let mut object_points = Vector::<Vector<Point3f>>::new();
        let mut rgb_points = Vector::<Vector<Point2f>>::new();
        let mut ir_points = Vector::<Vector<Point2f>>::new();
        for sample in &self.samples {
            object_points.push(object_template.clone());
            rgb_points.push(Vector::<Point2f>::from_slice(&sample.rgb_corners));
            ir_points.push(Vector::<Point2f>::from_slice(&sample.ir_corners));
        }

        let criteria = TermCriteria::new(
            TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
            100,
            1e-6,
        )?;

        let mut rgb_camera = initial_camera_matrix(self.rgb_size)?;
        let mut rgb_dist = Mat::zeros(8, 1, core::CV_64F)?.to_mat()?;
        let mut rgb_rvecs = Vector::<Mat>::new();
        let mut rgb_tvecs = Vector::<Mat>::new();
        let mut rgb_std_intrinsics = Mat::default();
        let mut rgb_std_extrinsics = Mat::default();
        let mut rgb_per_view_errors = Mat::default();
        let rgb_rms = calib3d::calibrate_camera_extended(
            &object_points,
            &rgb_points,
            Size::new(self.rgb_size.0 as i32, self.rgb_size.1 as i32),
            &mut rgb_camera,
            &mut rgb_dist,
            &mut rgb_rvecs,
            &mut rgb_tvecs,
            &mut rgb_std_intrinsics,
            &mut rgb_std_extrinsics,
            &mut rgb_per_view_errors,
            calib3d::CALIB_USE_INTRINSIC_GUESS | calib3d::CALIB_ZERO_TANGENT_DIST,
            criteria,
        )?;

        let mut ir_camera = initial_camera_matrix(self.ir_size)?;
        let mut ir_dist = Mat::zeros(8, 1, core::CV_64F)?.to_mat()?;
        let mut ir_rvecs = Vector::<Mat>::new();
        let mut ir_tvecs = Vector::<Mat>::new();
        let mut ir_std_intrinsics = Mat::default();
        let mut ir_std_extrinsics = Mat::default();
        let mut ir_per_view_errors = Mat::default();
        let ir_rms = calib3d::calibrate_camera_extended(
            &object_points,
            &ir_points,
            Size::new(self.ir_size.0 as i32, self.ir_size.1 as i32),
            &mut ir_camera,
            &mut ir_dist,
            &mut ir_rvecs,
            &mut ir_tvecs,
            &mut ir_std_intrinsics,
            &mut ir_std_extrinsics,
            &mut ir_per_view_errors,
            calib3d::CALIB_USE_INTRINSIC_GUESS | calib3d::CALIB_ZERO_TANGENT_DIST,
            criteria,
        )?;

        let mut rotation = Mat::default();
        let mut translation = Mat::default();
        let mut essential = Mat::default();
        let mut fundamental = Mat::default();
        let mut stereo_rvecs = Vector::<Mat>::new();
        let mut stereo_tvecs = Vector::<Mat>::new();
        let mut stereo_per_view_errors = Mat::default();
        let stereo_rms = calib3d::stereo_calibrate_extended(
            &object_points,
            &rgb_points,
            &ir_points,
            &mut rgb_camera,
            &mut rgb_dist,
            &mut ir_camera,
            &mut ir_dist,
            Size::new(self.rgb_size.0 as i32, self.rgb_size.1 as i32),
            &mut rotation,
            &mut translation,
            &mut essential,
            &mut fundamental,
            &mut stereo_rvecs,
            &mut stereo_tvecs,
            &mut stereo_per_view_errors,
            calib3d::CALIB_FIX_INTRINSIC,
            criteria,
        )?;

        Ok(StereoCalibrationResult {
            sample_count: self.samples.len(),
            corner_count: (self.pattern.0 * self.pattern.1) as usize,
            rgb_rms,
            ir_rms,
            stereo_rms,
            rgb_camera_matrix: rgb_camera,
            rgb_dist_coeffs: rgb_dist,
            ir_camera_matrix: ir_camera,
            ir_dist_coeffs: ir_dist,
            rotation,
            translation,
            essential,
            fundamental,
            rgb_per_view_errors,
            ir_per_view_errors,
            stereo_per_view_errors,
        })
    }
}

impl StereoCalibrationResult {
    pub fn to_text(&self) -> String {
        format!(
            "samples={}\ncorners_per_sample={}\nrgb_rms={:.8}\nir_rms={:.8}\nstereo_rms={:.8}\nrgb_per_view_errors={}\nir_per_view_errors={}\nstereo_per_view_errors={}\nrgb_camera_matrix={}\nrgb_dist_coeffs={}\nir_camera_matrix={}\nir_dist_coeffs={}\nrotation={}\ntranslation={}\nessential={}\nfundamental={}\n",
            self.sample_count,
            self.corner_count,
            self.rgb_rms,
            self.ir_rms,
            self.stereo_rms,
            mat_values(&self.rgb_per_view_errors),
            mat_values(&self.ir_per_view_errors),
            mat_values(&self.stereo_per_view_errors),
            mat_values(&self.rgb_camera_matrix),
            mat_values(&self.rgb_dist_coeffs),
            mat_values(&self.ir_camera_matrix),
            mat_values(&self.ir_dist_coeffs),
            mat_values(&self.rotation),
            mat_values(&self.translation),
            mat_values(&self.essential),
            mat_values(&self.fundamental),
        )
    }

    pub fn error_summary(&self) -> String {
        format!(
            "rgb_view={} ir_view={} stereo_view={}",
            error_stats(&self.rgb_per_view_errors),
            error_stats(&self.ir_per_view_errors),
            error_stats(&self.stereo_per_view_errors)
        )
    }
}

fn object_points(pattern: (i32, i32), square_size: f32) -> Vector<Point3f> {
    let mut out = Vector::<Point3f>::new();
    for y in 0..pattern.1 {
        for x in 0..pattern.0 {
            out.push(Point3f::new(
                x as f32 * square_size,
                y as f32 * square_size,
                0.0,
            ));
        }
    }
    out
}

fn initial_camera_matrix(size: (u32, u32)) -> Result<Mat> {
    let mut mat = Mat::eye(3, 3, core::CV_64F)?.to_mat()?;
    let f = size.0.max(size.1) as f64;
    *mat.at_2d_mut::<f64>(0, 0)? = f;
    *mat.at_2d_mut::<f64>(1, 1)? = f;
    *mat.at_2d_mut::<f64>(0, 2)? = size.0 as f64 * 0.5;
    *mat.at_2d_mut::<f64>(1, 2)? = size.1 as f64 * 0.5;
    Ok(mat)
}

fn mat_values(mat: &Mat) -> String {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut values = Vec::new();
    for r in 0..rows {
        let mut row = Vec::new();
        for c in 0..cols {
            let value = mat.at_2d::<f64>(r, c).copied().unwrap_or(f64::NAN);
            row.push(format!("{value:.10}"));
        }
        values.push(format!("[{}]", row.join(",")));
    }
    format!("[{}]", values.join(","))
}

fn error_stats(mat: &Mat) -> String {
    let mut values = Vec::new();
    for r in 0..mat.rows() {
        for c in 0..mat.cols() {
            if let Ok(value) = mat.at_2d::<f64>(r, c) {
                if value.is_finite() {
                    values.push(*value);
                }
            }
        }
    }
    if values.is_empty() {
        return "none".to_string();
    }
    let sum = values.iter().sum::<f64>();
    let mean = sum / values.len() as f64;
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("mean={mean:.4} max={max:.4}")
}
