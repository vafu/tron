use anyhow::{Context, Result};
use opencv::core::{self, Mat};
use opencv::imgproc;
use opencv::prelude::*;
use tron_api::{Frame, NoContext, PixelFormat, Processor, Rect, RoiCandidate, RoiResult, Size};

#[derive(Clone, Copy, Debug)]
pub struct OpenCvRoiConfig {
    pub threshold: u8,
    pub min_area: i32,
    pub max_area: Option<i32>,
    pub padding: u32,
}

impl Default for OpenCvRoiConfig {
    fn default() -> Self {
        Self {
            threshold: 24,
            min_area: 128,
            max_area: None,
            padding: 16,
        }
    }
}

pub struct OpenCvRoiDetector {
    config: OpenCvRoiConfig,
    packed: Vec<u8>,
    thresholded: Mat,
    labels: Mat,
    stats: Mat,
    centroids: Mat,
}

impl OpenCvRoiDetector {
    pub fn new(config: OpenCvRoiConfig) -> Self {
        Self {
            config,
            packed: Vec::new(),
            thresholded: Mat::default(),
            labels: Mat::default(),
            stats: Mat::default(),
            centroids: Mat::default(),
        }
    }

    pub fn detect_candidates(&mut self, input: Frame<'_>) -> Result<Vec<RoiCandidate>> {
        anyhow::ensure!(
            input.format == PixelFormat::Gray8,
            "OpenCV ROI detector requires Gray8 input, got {:?}",
            input.format
        );
        let width = input.meta.size.width as usize;
        let height = input.meta.size.height as usize;
        anyhow::ensure!(
            width > 0 && height > 0,
            "OpenCV ROI detector got empty input"
        );
        pack_gray8(input, &mut self.packed)?;

        let src = Mat::new_rows_cols_with_data(height as i32, width as i32, &self.packed)
            .context("wrap ROI detector input as OpenCV Mat")?;
        imgproc::threshold(
            &src,
            &mut self.thresholded,
            self.config.threshold as f64,
            255.0,
            imgproc::THRESH_BINARY,
        )
        .context("threshold ROI detector input")?;

        let components = imgproc::connected_components_with_stats(
            &self.thresholded,
            &mut self.labels,
            &mut self.stats,
            &mut self.centroids,
            8,
            core::CV_32S,
        )
        .context("connected components for ROI detector")?;

        let mut candidates = Vec::new();
        for label in 1..components {
            let candidate = self.candidate(label, input.meta.size)?;
            if candidate.is_plausible(self.config) {
                candidates.push(RoiCandidate {
                    rect: candidate.rect,
                    area: candidate.area as u32,
                });
            }
        }

        Ok(candidates)
    }
}

impl Processor<Frame<'_>> for OpenCvRoiDetector {
    type Output = Option<RoiResult>;

    fn process(&mut self, input: Frame<'_>, _context: NoContext) -> Result<Self::Output> {
        let best = self
            .detect_candidates(input)?
            .into_iter()
            .max_by_key(|candidate| candidate.area);
        Ok(best.map(|candidate| RoiResult {
            rect: candidate.rect,
            oriented_box: None,
        }))
    }
}

impl OpenCvRoiDetector {
    fn candidate(&self, label: i32, bounds: Size) -> Result<BlobCandidate> {
        let x = *self
            .stats
            .at_2d::<i32>(label, imgproc::CC_STAT_LEFT)
            .context("read component x")?;
        let y = *self
            .stats
            .at_2d::<i32>(label, imgproc::CC_STAT_TOP)
            .context("read component y")?;
        let width = *self
            .stats
            .at_2d::<i32>(label, imgproc::CC_STAT_WIDTH)
            .context("read component width")?;
        let height = *self
            .stats
            .at_2d::<i32>(label, imgproc::CC_STAT_HEIGHT)
            .context("read component height")?;
        let area = *self
            .stats
            .at_2d::<i32>(label, imgproc::CC_STAT_AREA)
            .context("read component area")?;
        let rect = padded_rect(
            Rect {
                x: x.max(0) as u32,
                y: y.max(0) as u32,
                size: Size {
                    width: width.max(0) as u32,
                    height: height.max(0) as u32,
                },
            },
            self.config.padding,
            bounds,
        );
        Ok(BlobCandidate { rect, area })
    }
}

#[derive(Clone, Copy, Debug)]
struct BlobCandidate {
    rect: Rect,
    area: i32,
}

impl BlobCandidate {
    fn is_plausible(self, config: OpenCvRoiConfig) -> bool {
        self.area >= config.min_area
            && config
                .max_area
                .map(|max_area| self.area <= max_area)
                .unwrap_or(true)
            && self.rect.size.width > 0
            && self.rect.size.height > 0
    }
}

fn pack_gray8(frame: Frame<'_>, packed: &mut Vec<u8>) -> Result<()> {
    let width = frame.meta.size.width as usize;
    let height = frame.meta.size.height as usize;
    let len = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("ROI detector input size overflow"))?;
    packed.resize(len, 0);
    let pixels = frame.view()?;
    for y in 0..height {
        let start = y * width;
        for x in 0..width {
            packed[start + x] = pixels[[y, x, 0]];
        }
    }
    Ok(())
}

fn padded_rect(rect: Rect, padding: u32, bounds: Size) -> Rect {
    let x0 = rect.x.saturating_sub(padding);
    let y0 = rect.y.saturating_sub(padding);
    let x1 = rect
        .x
        .saturating_add(rect.size.width)
        .saturating_add(padding)
        .min(bounds.width);
    let y1 = rect
        .y
        .saturating_add(rect.size.height)
        .saturating_add(padding)
        .min(bounds.height);
    Rect {
        x: x0,
        y: y0,
        size: Size {
            width: x1.saturating_sub(x0),
            height: y1.saturating_sub(y0),
        },
    }
}
