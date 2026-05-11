use anyhow::{Context, Result};
use opencv::boxed_ref::BoxedRef;
use opencv::calib3d;
use opencv::core::{self, Mat, Point2f, Size as CvSize, TermCriteria, Vector};
use opencv::imgproc;
use tron_api::{
    CalibrationFrameSide, CheckerboardDetection, CheckerboardSample, CheckerboardSpec, NoContext,
    PixelFormat, Point2d, Point3d, Processor, Size, View,
};

use crate::view::ViewExt;

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

    fn gray_mat(&mut self, view: View<'_>) -> Result<BoxedRef<'_, Mat>> {
        match view.format {
            PixelFormat::Gray8 => {
                pack_view(view, 1, &mut self.packed)?;
                Mat::new_rows_cols_with_data(
                    view.size.height as i32,
                    view.size.width as i32,
                    &self.packed,
                )
                .context("wrap checkerboard Gray8 input as OpenCV Mat")
            }
            PixelFormat::Bgra8 => {
                pack_gray_from_bgra(view, &mut self.packed)?;
                Mat::new_rows_cols_with_data(
                    view.size.height as i32,
                    view.size.width as i32,
                    &self.packed,
                )
                .context("wrap checkerboard BGRA8-derived Gray8 input as OpenCV Mat")
            }
            PixelFormat::Yuyv422 => {
                anyhow::bail!("checkerboard detector does not support YUYV422 yet")
            }
        }
    }
}

impl Processor<View<'_>> for OpenCvCheckerboardDetector {
    type Output = Option<CheckerboardDetection>;

    fn process(&mut self, input: View<'_>, _context: NoContext) -> Result<Self::Output> {
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
            frame_size: input.size,
            corners: corners
                .into_iter()
                .map(|corner| Point2d {
                    x: corner.x as f64,
                    y: corner.y as f64,
                })
                .collect(),
            score: None,
        }))
    }
}

#[derive(Default)]
pub struct CheckerboardSampleBuilder;

impl CheckerboardSampleBuilder {
    pub fn new() -> Self {
        Self
    }
}

impl<'a> Processor<(CalibrationFrameSide<'a>, CalibrationFrameSide<'a>)>
    for CheckerboardSampleBuilder
{
    type Output = Option<CheckerboardSample>;

    fn process(
        &mut self,
        (left, right): (CalibrationFrameSide<'a>, CalibrationFrameSide<'a>),
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(left_detection) = left.detection else {
            return Ok(None);
        };
        let Some(right_detection) = right.detection else {
            return Ok(None);
        };
        anyhow::ensure!(
            left_detection.spec == right_detection.spec,
            "checkerboard sample sides use different board specs"
        );
        anyhow::ensure!(
            left_detection.corners.len() == right_detection.corners.len(),
            "checkerboard sample sides have different corner counts"
        );

        Ok(Some(CheckerboardSample {
            spec: left_detection.spec,
            object_points: checkerboard_object_points(left_detection.spec),
            left_corners: left_detection.corners.clone(),
            right_corners: right_detection.corners.clone(),
            left_frame_id: left.frame.meta.id,
            right_frame_id: right.frame.meta.id,
            left_frame_size: left_detection.frame_size,
            right_frame_size: right_detection.frame_size,
            left_timestamp: left.frame.meta.timestamp,
            right_timestamp: right.frame.meta.timestamp,
        }))
    }
}

pub fn checkerboard_object_points(spec: CheckerboardSpec) -> Vec<Point3d> {
    let mut points =
        Vec::with_capacity((spec.inner_corners.width * spec.inner_corners.height) as usize);
    for y in 0..spec.inner_corners.height {
        for x in 0..spec.inner_corners.width {
            points.push(Point3d {
                x: x as f64 * spec.square_size_mm,
                y: y as f64 * spec.square_size_mm,
                z: 0.0,
            });
        }
    }
    points
}

fn cv_size(size: Size) -> Result<CvSize> {
    anyhow::ensure!(
        size.width > 0 && size.height > 0,
        "checkerboard pattern size must be non-empty"
    );
    Ok(CvSize::new(size.width as i32, size.height as i32))
}

fn pack_view(view: View<'_>, bytes_per_pixel: usize, packed: &mut Vec<u8>) -> Result<()> {
    let row_len = view.size.width as usize * bytes_per_pixel;
    let len = row_len
        .checked_mul(view.size.height as usize)
        .ok_or_else(|| anyhow::anyhow!("checkerboard input size overflow"))?;
    packed.resize(len, 0);
    for (y, row) in view.rows().enumerate() {
        let start = y * row_len;
        packed[start..start + row_len].copy_from_slice(row);
    }
    Ok(())
}

fn pack_gray_from_bgra(view: View<'_>, packed: &mut Vec<u8>) -> Result<()> {
    let width = view.size.width as usize;
    let height = view.size.height as usize;
    let len = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("checkerboard input size overflow"))?;
    packed.resize(len, 0);
    for (y, row) in view.rows().enumerate() {
        let dst_row_start = y * width;
        for x in 0..width {
            let src = x * 4;
            let b = row[src] as u16;
            let g = row[src + 1] as u16;
            let r = row[src + 2] as u16;
            packed[dst_row_start + x] = ((r + g + b) / 3) as u8;
        }
    }
    Ok(())
}
