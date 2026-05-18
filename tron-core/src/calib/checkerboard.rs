use anyhow::{Context, Result};
use glam::{DMat3, DVec3};
use opencv::boxed_ref::BoxedRef;
use opencv::calib3d;
use opencv::core::{self, Mat, Point2f, Point3f, Size as CvSize, TermCriteria, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use tron_api::{
    CalibrationFrameSide, CameraCalibration, CheckerboardDetection, CheckerboardSample,
    CheckerboardSpec, CheckerboardStereoCalibration, Frame, FrameMeta, NoContext, PixelFormat,
    Point2d, Point3d, Processor, Size,
};

#[derive(Clone, Copy, Debug)]
pub struct OpenCvCheckerboardConfig {
    pub spec: CheckerboardSpec,
    pub refine_corners: bool,
}

impl OpenCvCheckerboardConfig {
    pub fn new(spec: CheckerboardSpec) -> Self {
        Self {
            spec,
            refine_corners: true,
        }
    }
}

pub struct OpenCvCheckerboardDetector {
    config: OpenCvCheckerboardConfig,
    packed: Vec<u8>,
}

impl OpenCvCheckerboardDetector {
    pub fn new(config: OpenCvCheckerboardConfig) -> Self {
        Self {
            config,
            packed: Vec::new(),
        }
    }

    fn gray_mat<'a>(&'a mut self, frame: Frame<'a>) -> Result<BoxedRef<'a, Mat>> {
        match frame.format {
            PixelFormat::Gray8 => {
                if can_wrap_gray8(frame) {
                    // SAFETY: direct OpenCV wrapping only uses raw storage when
                    // logical and physical Gray8 rows are identical.
                    let raw = unsafe { frame.buffer.raw() };
                    let len = frame.meta.size.width as usize * frame.meta.size.height as usize;
                    Mat::new_rows_cols_with_data(
                        frame.meta.size.height as i32,
                        frame.meta.size.width as i32,
                        &raw[..len],
                    )
                    .context("wrap checkerboard Gray8 input as OpenCV Mat")
                } else {
                    pack_view(frame, 1, &mut self.packed)?;
                    Mat::new_rows_cols_with_data(
                        frame.meta.size.height as i32,
                        frame.meta.size.width as i32,
                        &self.packed,
                    )
                    .context("wrap packed checkerboard Gray8 input as OpenCV Mat")
                }
            }
            PixelFormat::Bgra8 => {
                pack_gray_from_bgra(frame, &mut self.packed)?;
                Mat::new_rows_cols_with_data(
                    frame.meta.size.height as i32,
                    frame.meta.size.width as i32,
                    &self.packed,
                )
                .context("wrap checkerboard BGRA8-derived Gray8 input as OpenCV Mat")
            }
        }
    }
}

impl Processor<Frame<'_>> for OpenCvCheckerboardDetector {
    type Output = Option<CheckerboardDetection>;

    fn process(&mut self, input: Frame<'_>, _context: NoContext) -> Result<Self::Output> {
        let refine_corners = self.config.refine_corners;
        let pattern = cv_size(self.config.spec.inner_corners)?;
        let gray = self.gray_mat(input)?;
        let mut corners = Vector::<Point2f>::new();
        let found = calib3d::find_chessboard_corners_def(&gray, pattern, &mut corners)
            .context("find checkerboard corners")?;
        if !found {
            return Ok(None);
        }

        if refine_corners {
            let criteria =
                TermCriteria::new(core::TermCriteria_COUNT + core::TermCriteria_EPS, 30, 0.001)
                    .context("create corner refinement criteria")?;
            imgproc::corner_sub_pix(
                &gray,
                &mut corners,
                CvSize::new(11, 11),
                CvSize::new(-1, -1),
                criteria,
            )
            .context("refine checkerboard corners")?;
        }

        Ok(Some(CheckerboardDetection {
            spec: self.config.spec,
            frame_size: input.meta.size,
            corners: corners
                .into_iter()
                .map(|corner| Point2d::new(corner.x as f64, corner.y as f64))
                .collect(),
            score: None,
        }))
    }
}

pub struct CheckerboardSampleBuilder {
    spec: CheckerboardSpec,
}

impl CheckerboardSampleBuilder {
    pub fn new(spec: CheckerboardSpec) -> Self {
        Self { spec }
    }
}

impl Processor<(Option<CalibrationFrameSide>, Option<CalibrationFrameSide>)>
    for CheckerboardSampleBuilder
{
    type Output = Option<CheckerboardSample>;

    fn process(
        &mut self,
        (left, right): (Option<CalibrationFrameSide>, Option<CalibrationFrameSide>),
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(left) = left else {
            return Ok(None);
        };
        let Some(right) = right else {
            return Ok(None);
        };
        checkerboard_sample(left, right, self.spec).map(Some)
    }
}

pub fn calibration_frame_side(
    frame_meta: FrameMeta,
    detection: &CheckerboardDetection,
) -> CalibrationFrameSide {
    CalibrationFrameSide {
        frame_meta,
        frame_size: detection.frame_size,
        corners: detection.corners.clone(),
        score: detection.score,
    }
}

pub fn checkerboard_sample(
    left: CalibrationFrameSide,
    right: CalibrationFrameSide,
    spec: CheckerboardSpec,
) -> Result<CheckerboardSample> {
    anyhow::ensure!(
        left.corners.len() == right.corners.len(),
        "checkerboard sample sides have different corner counts"
    );
    anyhow::ensure!(
        left.corners.len() == (spec.inner_corners.width * spec.inner_corners.height) as usize,
        "checkerboard sample corner count does not match board spec"
    );
    Ok(CheckerboardSample {
        spec,
        object_points: checkerboard_object_points(spec),
        left,
        right,
    })
}

pub fn checkerboard_object_points(spec: CheckerboardSpec) -> Vec<Point3d> {
    let mut points =
        Vec::with_capacity((spec.inner_corners.width * spec.inner_corners.height) as usize);
    for y in 0..spec.inner_corners.height {
        for x in 0..spec.inner_corners.width {
            points.push(Point3d::new(
                x as f64 * spec.square_size_mm,
                y as f64 * spec.square_size_mm,
                0.0,
            ));
        }
    }
    points
}

pub fn calibrate_stereo_checkerboard(
    samples: &[CheckerboardSample],
) -> Result<CheckerboardStereoCalibration> {
    anyhow::ensure!(
        !samples.is_empty(),
        "at least one checkerboard sample is required"
    );
    let spec = samples[0].spec;
    let left_size = samples[0].left.frame_size;
    let right_size = samples[0].right.frame_size;

    let expected_corners = (spec.inner_corners.width * spec.inner_corners.height) as usize;
    for (index, sample) in samples.iter().enumerate() {
        anyhow::ensure!(
            sample.spec == spec,
            "sample {index} uses a different checkerboard spec"
        );
        anyhow::ensure!(
            sample.left.frame_size == left_size,
            "sample {index} uses a different left frame size"
        );
        anyhow::ensure!(
            sample.right.frame_size == right_size,
            "sample {index} uses a different right frame size"
        );
        anyhow::ensure!(
            sample.object_points.len() == expected_corners
                && sample.left.corners.len() == expected_corners
                && sample.right.corners.len() == expected_corners,
            "sample {index} has an unexpected corner count"
        );
    }

    let object_points = cv_object_points(samples);
    let left_points = cv_image_points(samples, |sample| &sample.left.corners);
    let right_points = cv_image_points(samples, |sample| &sample.right.corners);
    let left_image_size = cv_size(left_size)?;
    let right_image_size = cv_size(right_size)?;

    let mut left_camera = Mat::eye(3, 3, core::CV_64F)
        .context("create left camera matrix")?
        .to_mat()
        .context("materialize left camera matrix")?;
    let mut right_camera = Mat::eye(3, 3, core::CV_64F)
        .context("create right camera matrix")?
        .to_mat()
        .context("materialize right camera matrix")?;
    let mut left_dist = Mat::zeros(8, 1, core::CV_64F)
        .context("create left distortion matrix")?
        .to_mat()
        .context("materialize left distortion matrix")?;
    let mut right_dist = Mat::zeros(8, 1, core::CV_64F)
        .context("create right distortion matrix")?
        .to_mat()
        .context("materialize right distortion matrix")?;
    let mut rvecs = Vector::<Mat>::new();
    let mut tvecs = Vector::<Mat>::new();

    let criteria = TermCriteria::new(core::TermCriteria_COUNT + core::TermCriteria_EPS, 100, 1e-6)
        .context("create stereo calibration criteria")?;
    let left_error = calib3d::calibrate_camera(
        &object_points,
        &left_points,
        left_image_size,
        &mut left_camera,
        &mut left_dist,
        &mut rvecs,
        &mut tvecs,
        0,
        criteria,
    )
    .context("calibrate left camera")?;
    let right_error = calib3d::calibrate_camera(
        &object_points,
        &right_points,
        right_image_size,
        &mut right_camera,
        &mut right_dist,
        &mut rvecs,
        &mut tvecs,
        0,
        criteria,
    )
    .context("calibrate right camera")?;

    let mut rotation = Mat::eye(3, 3, core::CV_64F)
        .context("create stereo rotation matrix")?
        .to_mat()
        .context("materialize stereo rotation matrix")?;
    let mut translation = Mat::zeros(3, 1, core::CV_64F)
        .context("create stereo translation vector")?
        .to_mat()
        .context("materialize stereo translation vector")?;
    let mut essential = Mat::default();
    let mut fundamental = Mat::default();
    let mut per_view_errors = Mat::default();

    let stereo_error = calib3d::stereo_calibrate_1(
        &object_points,
        &left_points,
        &right_points,
        &mut left_camera,
        &mut left_dist,
        &mut right_camera,
        &mut right_dist,
        // OpenCV's pinhole stereo API accepts a single image size. Intrinsics
        // are already calibrated with per-camera sizes above and fixed here.
        left_image_size,
        &mut rotation,
        &mut translation,
        &mut essential,
        &mut fundamental,
        &mut per_view_errors,
        calib3d::CALIB_FIX_INTRINSIC,
        criteria,
    )
    .context("calibrate stereo camera pair")?;

    Ok(CheckerboardStereoCalibration {
        spec,
        sample_count: samples.len(),
        left: CameraCalibration {
            image_size: left_size,
            camera_matrix: mat3(&left_camera).context("read left camera matrix")?,
            distortion: mat_vec(&left_dist).context("read left distortion")?,
            reprojection_error: left_error,
        },
        right: CameraCalibration {
            image_size: right_size,
            camera_matrix: mat3(&right_camera).context("read right camera matrix")?,
            distortion: mat_vec(&right_dist).context("read right distortion")?,
            reprojection_error: right_error,
        },
        rotation: mat3(&rotation).context("read stereo rotation")?,
        translation: mat_vec3(&translation).context("read stereo translation")?,
        essential: mat3(&essential).context("read essential matrix")?,
        fundamental: mat3(&fundamental).context("read fundamental matrix")?,
        stereo_reprojection_error: stereo_error,
        per_view_errors: mat_vec(&per_view_errors).context("read per-view errors")?,
    })
}

fn cv_size(size: Size) -> Result<CvSize> {
    anyhow::ensure!(
        size.width > 0 && size.height > 0,
        "checkerboard pattern size must be non-empty"
    );
    Ok(CvSize::new(size.width as i32, size.height as i32))
}

fn pack_view(frame: Frame<'_>, channels: usize, packed: &mut Vec<u8>) -> Result<()> {
    let row_len = frame.meta.size.width as usize * channels;
    let len = row_len
        .checked_mul(frame.meta.size.height as usize)
        .ok_or_else(|| anyhow::anyhow!("checkerboard input size overflow"))?;
    packed.resize(len, 0);
    let view = frame.view()?;
    anyhow::ensure!(
        view.shape()[2] == channels,
        "checkerboard expected {} channels, got {}",
        channels,
        view.shape()[2]
    );
    for y in 0..frame.meta.size.height as usize {
        for x in 0..frame.meta.size.width as usize {
            for channel in 0..channels {
                packed[y * row_len + x * channels + channel] = view[[y, x, channel]];
            }
        }
    }
    Ok(())
}

fn pack_gray_from_bgra(frame: Frame<'_>, packed: &mut Vec<u8>) -> Result<()> {
    let width = frame.meta.size.width as usize;
    let height = frame.meta.size.height as usize;
    let len = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("checkerboard input size overflow"))?;
    packed.resize(len, 0);
    let view = frame.view()?;
    for y in 0..height {
        let dst_row_start = y * width;
        for x in 0..width {
            let b = view[[y, x, 0]] as u16;
            let g = view[[y, x, 1]] as u16;
            let r = view[[y, x, 2]] as u16;
            packed[dst_row_start + x] = ((r + g + b) / 3) as u8;
        }
    }
    Ok(())
}

fn can_wrap_gray8(frame: Frame<'_>) -> bool {
    frame.format == PixelFormat::Gray8
        && frame.buffer.stride() == frame.meta.size.width as usize
        && !frame.buffer.is_horizontally_mirrored()
        && !frame.buffer.is_vertically_mirrored()
}

fn cv_object_points(samples: &[CheckerboardSample]) -> Vector<Vector<Point3f>> {
    let mut all = Vector::<Vector<Point3f>>::new();
    for sample in samples {
        let mut points = Vector::<Point3f>::new();
        for point in &sample.object_points {
            points.push(Point3f::new(point.x as f32, point.y as f32, point.z as f32));
        }
        all.push(points);
    }
    all
}

fn cv_image_points(
    samples: &[CheckerboardSample],
    corners: impl Fn(&CheckerboardSample) -> &[Point2d],
) -> Vector<Vector<Point2f>> {
    let mut all = Vector::<Vector<Point2f>>::new();
    for sample in samples {
        let mut points = Vector::<Point2f>::new();
        for point in corners(sample) {
            points.push(Point2f::new(point.x as f32, point.y as f32));
        }
        all.push(points);
    }
    all
}

fn mat3(mat: &Mat) -> Result<DMat3> {
    anyhow::ensure!(
        mat.rows() == 3 && mat.cols() == 3,
        "expected 3x3 matrix, got {}x{}",
        mat.rows(),
        mat.cols()
    );
    Ok(DMat3::from_cols(
        DVec3::new(
            *mat.at_2d::<f64>(0, 0)?,
            *mat.at_2d::<f64>(1, 0)?,
            *mat.at_2d::<f64>(2, 0)?,
        ),
        DVec3::new(
            *mat.at_2d::<f64>(0, 1)?,
            *mat.at_2d::<f64>(1, 1)?,
            *mat.at_2d::<f64>(2, 1)?,
        ),
        DVec3::new(
            *mat.at_2d::<f64>(0, 2)?,
            *mat.at_2d::<f64>(1, 2)?,
            *mat.at_2d::<f64>(2, 2)?,
        ),
    ))
}

fn mat_vec3(mat: &Mat) -> Result<DVec3> {
    let values = mat_vec(mat)?;
    anyhow::ensure!(
        values.len() == 3,
        "expected 3-vector, got {} values",
        values.len()
    );
    Ok(DVec3::new(values[0], values[1], values[2]))
}

fn mat_vec(mat: &Mat) -> Result<Vec<f64>> {
    if mat.empty() {
        return Ok(Vec::new());
    }
    Ok(mat.data_typed::<f64>()?.to_vec())
}
