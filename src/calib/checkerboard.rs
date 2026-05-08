use crate::calib::AffineCalib;
use crate::types::Image;
use anyhow::{Context, Result, anyhow, bail};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Size, TermCriteria, TermCriteria_Type, Vector};
use opencv::imgproc;
use opencv::prelude::*;

#[derive(Clone, Debug)]
pub struct CheckerboardResult {
    pub calib: AffineCalib,
    pub corners: usize,
    pub rms_error: f32,
}

#[derive(Clone, Debug)]
pub struct CheckerboardSample {
    pub rgb_corners: Vec<Point2f>,
    pub ir_corners: Vec<Point2f>,
    pub pattern: (i32, i32),
}

pub fn calibrate_affine(
    rgb: &Image,
    ir: &Image,
    pattern: (i32, i32),
) -> Result<CheckerboardResult> {
    let sample = capture_sample(rgb, ir, pattern)?;
    calibrate_affine_from_sample(&sample, rgb, ir)
}

pub fn capture_sample(rgb: &Image, ir: &Image, pattern: (i32, i32)) -> Result<CheckerboardSample> {
    if pattern.0 < 2 || pattern.1 < 2 {
        bail!("checkerboard pattern must be inner corners, e.g. 9x6");
    }

    let rgb_corners = detect_corners(rgb, pattern).context("detect RGB checkerboard")?;
    let ir_corners = detect_corners(ir, pattern).context("detect IR checkerboard")?;
    if rgb_corners.len() != ir_corners.len() {
        bail!(
            "corner count mismatch: rgb={} ir={}",
            rgb_corners.len(),
            ir_corners.len()
        );
    }
    if rgb_corners.len() < 4 {
        bail!("need at least 4 matching checkerboard corners");
    }

    Ok(CheckerboardSample {
        rgb_corners,
        ir_corners,
        pattern,
    })
}

pub fn calibrate_affine_from_sample(
    sample: &CheckerboardSample,
    rgb: &Image,
    ir: &Image,
) -> Result<CheckerboardResult> {
    let rgb_norm = sample
        .rgb_corners
        .iter()
        .map(|p| (p.x / rgb.width as f32, p.y / rgb.height as f32))
        .collect::<Vec<_>>();
    let ir_norm = sample
        .ir_corners
        .iter()
        .map(|p| (p.x / ir.width as f32, p.y / ir.height as f32))
        .collect::<Vec<_>>();

    let (scale_x, offset_x) = fit_line(&ir_norm, &rgb_norm, Axis::X)?;
    let (scale_y, offset_y) = fit_line(&ir_norm, &rgb_norm, Axis::Y)?;
    let calib = AffineCalib {
        scale_x,
        scale_y,
        offset_x,
        offset_y,
        use_binary: false,
    };
    let rms_error = reprojection_rms(&ir_norm, &rgb_norm, calib);

    Ok(CheckerboardResult {
        calib,
        corners: sample.rgb_corners.len(),
        rms_error,
    })
}

pub fn detect_corners(img: &Image, pattern: (i32, i32)) -> Result<Vec<Point2f>> {
    let gray = gray_mat(img)?;
    let pattern_size = Size::new(pattern.0, pattern.1);
    let mut corners = Vector::<Point2f>::default();
    let mut found =
        calib3d::find_chessboard_corners_sb_def(&gray, pattern_size, &mut corners).unwrap_or(false);
    if !found {
        corners.clear();
        found = calib3d::find_chessboard_corners_def(&gray, pattern_size, &mut corners)?;
    }
    if !found {
        bail!(
            "checkerboard {}x{} not found in {}x{} frame",
            pattern.0,
            pattern.1,
            img.width,
            img.height
        );
    }

    let criteria = TermCriteria::new(
        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
        30,
        0.01,
    )?;
    imgproc::corner_sub_pix(
        &gray,
        &mut corners,
        Size::new(11, 11),
        Size::new(-1, -1),
        criteria,
    )?;
    Ok(corners.to_vec())
}

fn gray_mat(img: &Image) -> Result<Mat> {
    let mut gray = img.grey_iter().collect::<Vec<_>>();
    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            img.height as i32,
            img.width as i32,
            opencv::core::CV_8UC1,
            gray.as_mut_ptr() as *mut _,
        )?
    };
    Ok(mat.clone())
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

fn fit_line(src: &[(f32, f32)], dst: &[(f32, f32)], axis: Axis) -> Result<(f32, f32)> {
    let n = src.len().min(dst.len());
    if n < 2 {
        bail!("need at least 2 points to fit affine axis");
    }

    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for i in 0..n {
        sx += coord(src[i], axis);
        sy += coord(dst[i], axis);
    }
    let mean_x = sx / n as f32;
    let mean_y = sy / n as f32;

    let mut var_x = 0.0f32;
    let mut cov = 0.0f32;
    for i in 0..n {
        let dx = coord(src[i], axis) - mean_x;
        let dy = coord(dst[i], axis) - mean_y;
        var_x += dx * dx;
        cov += dx * dy;
    }
    if var_x.abs() < f32::EPSILON {
        return Err(anyhow!("degenerate checkerboard fit"));
    }
    let scale = cov / var_x;
    let offset = mean_y - scale * mean_x;
    Ok((scale, offset))
}

fn coord(p: (f32, f32), axis: Axis) -> f32 {
    match axis {
        Axis::X => p.0,
        Axis::Y => p.1,
    }
}

fn reprojection_rms(src: &[(f32, f32)], dst: &[(f32, f32)], calib: AffineCalib) -> f32 {
    let n = src.len().min(dst.len()).max(1);
    let mut sum = 0.0f32;
    for i in 0..n {
        let px = src[i].0 * calib.scale_x + calib.offset_x;
        let py = src[i].1 * calib.scale_y + calib.offset_y;
        let dx = px - dst[i].0;
        let dy = py - dst[i].1;
        sum += dx * dx + dy * dy;
    }
    (sum / n as f32).sqrt()
}
