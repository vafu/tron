use super::RoiHinter;
use crate::calib;
use crate::inference::{OrtConfig, load_ort_session};
use crate::types::{FrameContext, Image, PixelFormat, RectNorm};
use anyhow::{Context, Result, anyhow};
use ndarray::Array4;
use opencv::core::{CV_8UC1, Mat, Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::borrow::Cow;
use std::path::Path;

pub struct PalmDetector {
    session: Session,
    input_name: String,
    input_size: u32,
    anchors: Vec<Anchor>,
    pad: f32,
    frame: u64,
    source: DetectorSource,
    last_timing_log: std::time::Instant,
    timing: DetectorTiming,
    /// Most recent IR-diff seen — reused when the current frame has none
    /// (e.g. flashlight just toggled off). Without this, the detector falls
    /// through to the next hinter on every flashlight-off frame and the ROI
    /// geometry flips between detector- and tracker-sized rects.
    last_diff: Option<Image>,
}

#[derive(Default)]
struct DetectorTiming {
    frames: u32,
    prep_us: u64,
    run_us: u64,
    decode_us: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum DetectorSource {
    Rgb,
    IrForeground,
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    x_center: f32,
    y_center: f32,
}

impl PalmDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_source(model_path, DetectorSource::Rgb)
    }

    pub fn new_ir<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_source(model_path, DetectorSource::IrForeground)
    }

    pub fn with_source<P: AsRef<Path>>(model_path: P, source: DetectorSource) -> Result<Self> {
        let session = load_ort_session(model_path.as_ref(), OrtConfig::cpu(1))
            .with_context(|| format!("load palm detector {}", model_path.as_ref().display()))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("no inputs"))?;
        let input_name = input.name().to_string();
        let input_size = input
            .dtype()
            .tensor_shape()
            .and_then(|s| {
                let dims: &[i64] = s.as_ref();
                if dims.len() >= 4 {
                    Some(dims[2] as u32)
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("invalid input shape"))?;

        let anchors = generate_anchors_mp(input_size);

        Ok(Self {
            session,
            input_name,
            input_size,
            anchors,
            pad: 0.20,
            frame: 0,
            source,
            last_timing_log: std::time::Instant::now(),
            timing: DetectorTiming::default(),
            last_diff: None,
        })
    }

    fn run_model(&mut self, signal: &ModelInput) -> Result<Option<RectNorm>> {
        let w = signal.cols() as f32;
        let h = signal.rows() as f32;
        let scale = (self.input_size as f32 / w).min(self.input_size as f32 / h);
        let nw = (w * scale) as i32;
        let nh = (h * scale) as i32;
        let dx = (self.input_size as i32 - nw) / 2;
        let dy = (self.input_size as i32 - nh) / 2;

        let t_prep = std::time::Instant::now();
        let mut input;
        {
            let _span =
                tracing::debug_span!("roi.palm_detector.prep", source = ?self.source).entered();
            input =
                Array4::<f32>::zeros((1, 3, self.input_size as usize, self.input_size as usize));
            for oy in 0..self.input_size as i32 {
                let ly = oy - dy;
                if !(0..nh).contains(&ly) {
                    continue;
                }
                let sy = ((ly as f32 + 0.5) / scale).floor().clamp(0.0, h - 1.0) as usize;
                for ox in 0..self.input_size as i32 {
                    let lx = ox - dx;
                    if !(0..nw).contains(&lx) {
                        continue;
                    }
                    let sx = ((lx as f32 + 0.5) / scale).floor().clamp(0.0, w - 1.0) as usize;
                    let px = signal.sample_rgb(sx, sy);
                    let y = oy as usize;
                    let x = ox as usize;
                    input[[0, 0, y, x]] = px[0] as f32 / 127.5 - 1.0;
                    input[[0, 1, y, x]] = px[1] as f32 / 127.5 - 1.0;
                    input[[0, 2, y, x]] = px[2] as f32 / 127.5 - 1.0;
                }
            }
        }
        self.timing.prep_us += t_prep.elapsed().as_micros() as u64;

        let tensor = Tensor::from_array(input)?;
        let input_map: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            vec![(self.input_name.as_str().into(), tensor.into())];
        let t_run = std::time::Instant::now();
        let outputs = {
            let _span =
                tracing::debug_span!("roi.palm_detector.ort", source = ?self.source).entered();
            self.session
                .run(input_map)
                .map_err(|e| anyhow!("ort run: {e}"))?
        };
        self.timing.run_us += t_run.elapsed().as_micros() as u64;

        let t_decode = std::time::Instant::now();
        let _span =
            tracing::debug_span!("roi.palm_detector.decode", source = ?self.source).entered();
        let scores_out = outputs
            .iter()
            .find(|o| {
                let shape = o.1.dtype().tensor_shape();
                shape.map_or(false, |s| s.as_ref().last() == Some(&1))
            })
            .context("scores not found")?;

        let boxes_out = outputs
            .iter()
            .find(|o| {
                let shape = o.1.dtype().tensor_shape();
                shape.map_or(false, |s| s.as_ref().last() == Some(&18))
            })
            .context("boxes not found")?;

        let (_, raw_scores) = scores_out
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("ort extract: {e}"))?;
        let (_, raw_boxes) = boxes_out
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("ort extract: {e}"))?;
        self.timing.decode_us += t_decode.elapsed().as_micros() as u64;
        self.timing.frames += 1;

        let scores: &[f32] = &raw_scores;
        let boxes: &[f32] = &raw_boxes;

        let mut best_score = -1e10f32;
        let mut best_idx = None;

        for (i, &score) in scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let best_conf = best_idx
            .map(|_| 1.0 / (1.0 + (-best_score).exp()))
            .unwrap_or(0.0);
        if self.frame % 30 == 0 {
            eprintln!("palm: best_conf={best_conf:.3} (thresh 0.5)");
        }
        if let Some(i) = best_idx {
            let conf = best_conf;
            if conf > 0.5 {
                let b = &boxes[i * 18..(i + 1) * 18];
                let anchor = &self.anchors[i];

                let s = self.input_size as f32;
                let cx_norm = b[0] / s + anchor.x_center;
                let cy_norm = b[1] / s + anchor.y_center;
                let w_norm = b[2] / s;
                let h_norm = b[3] / s;

                let cx = (cx_norm - dx as f32 / s) * (s / nw as f32);
                let cy = (cy_norm - dy as f32 / s) * (s / nh as f32);
                let rw = w_norm * (s / nw as f32);
                let rh = h_norm * (s / nh as f32);

                let expand = 2.4;
                let fw = rw * expand;
                let fh = rh * expand;

                return Ok(Some(RectNorm {
                    x: (cx - fw / 2.0).clamp(0.0, 1.0),
                    y: (cy - fh / 2.0).clamp(0.0, 1.0),
                    w: fw.clamp(0.0, 1.0),
                    h: fh.clamp(0.0, 1.0),
                }));
            }
        }

        Ok(None)
    }

    fn log_timing(&mut self) {
        if self.last_timing_log.elapsed() < std::time::Duration::from_secs(2) {
            return;
        }
        let n = self.timing.frames.max(1) as f32;
        tracing::debug!(
            target: "tron::roi",
            source = ?self.source,
            fps = self.timing.frames as f32 / self.last_timing_log.elapsed().as_secs_f32(),
            prep_ms = self.timing.prep_us as f32 / n / 1000.0,
            ort_ms = self.timing.run_us as f32 / n / 1000.0,
            decode_ms = self.timing.decode_us as f32 / n / 1000.0,
            "palm detector timing"
        );
        self.last_timing_log = std::time::Instant::now();
        self.timing = DetectorTiming::default();
    }
}

impl RoiHinter for PalmDetector {
    fn hint(&mut self, ctx: &FrameContext) -> (Option<RectNorm>, Option<Image>) {
        self.frame = self.frame.wrapping_add(1);
        let input = match self.source {
            DetectorSource::Rgb => match ModelInput::from_rgb(&ctx.rgb) {
                Ok(input) => input,
                Err(e) => {
                    eprintln!("palm rgb input error: {e}");
                    return (None, None);
                }
            },
            DetectorSource::IrForeground => {
                if let Some(d) = &ctx.ir_diff {
                    self.last_diff = Some(d.clone());
                }
                let signal_img = match ctx.ir_diff.as_ref().or(self.last_diff.as_ref()) {
                    Some(d) => d,
                    None => {
                        if self.frame % 30 == 0 {
                            eprintln!(
                                "palm: skipped — no ir_diff (flashlight={})",
                                ctx.ir_flashlight_on
                            );
                        }
                        return (None, None);
                    }
                };
                match ModelInput::from_ir(signal_img) {
                    Ok(input) => input,
                    Err(e) => {
                        eprintln!("palm ir input error: {e}");
                        return (None, None);
                    }
                }
            }
        };

        let model_result = self.run_model(&input);
        self.log_timing();

        match model_result {
            Ok(Some(rect)) => {
                let mapped = match self.source {
                    DetectorSource::Rgb => rect.padded(self.pad),
                    DetectorSource::IrForeground => {
                        calib::current().map_rect(rect).padded(self.pad)
                    }
                };
                (Some(mapped), None)
            }
            Ok(None) => {
                if let DetectorSource::IrForeground = self.source {
                    if let Ok(mat) = input.grayscale_mat() {
                        match find_blob_fallback(&mat) {
                            Ok(Some(rect)) => {
                                let mapped = calib::current().map_rect(rect).padded(self.pad);
                                if self.frame % 30 == 0 {
                                    eprintln!(
                                        "palm: model rejected, blob fallback rect=[{:.2},{:.2} {:.2}x{:.2}]",
                                        mapped.x, mapped.y, mapped.w, mapped.h
                                    );
                                }
                                return (Some(mapped), None);
                            }
                            _ => {}
                        }
                    }
                }
                if self.frame % 30 == 0 {
                    eprintln!("palm: no detection");
                }
                (None, None)
            }
            Err(e) => {
                eprintln!("palm detector error: {e}");
                (None, None)
            }
        }
    }
}

enum ModelInput {
    Image(Image),
}

impl ModelInput {
    fn from_ir(img: &Image) -> Result<Self> {
        Ok(Self::Image(img.clone()))
    }

    fn from_rgb(img: &Image) -> Result<Self> {
        if img.format != PixelFormat::Rgba8 {
            return Err(anyhow!("expected RGBA8 rgb image"));
        }
        Ok(Self::Image(img.clone()))
    }

    fn image(&self) -> &Image {
        let Self::Image(img) = self;
        img
    }

    fn rows(&self) -> i32 {
        self.image().height as i32
    }

    fn cols(&self) -> i32 {
        self.image().width as i32
    }

    fn channels(&self) -> i32 {
        self.image().format.bytes_per_pixel() as i32
    }

    fn grayscale_mat(&self) -> Result<Mat> {
        let img = self.image();
        let mut grey: Vec<u8> = img.grey_iter().collect();
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(
                img.height as i32,
                img.width as i32,
                CV_8UC1,
                grey.as_mut_ptr() as *mut _,
            )?
        }
        .clone();
        Ok(mat)
    }

    fn sample_rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let img = self.image();
        match img.format {
            PixelFormat::Rgba8 => {
                let i = (y * img.width as usize + x) * 4;
                [img.data[i], img.data[i + 1], img.data[i + 2]]
            }
            PixelFormat::R8 => {
                let g = img.data[y * img.width as usize + x];
                [g, g, g]
            }
        }
    }
}

fn find_blob_fallback(mat: &Mat) -> Result<Option<RectNorm>> {
    let mut binary = Mat::default();
    imgproc::threshold(
        mat,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    let k3 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &binary,
        &mut opened,
        imgproc::MORPH_OPEN,
        &k3,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let mut labels = Mat::default();
    let n = imgproc::connected_components_with_stats(
        &opened,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        opencv::core::CV_32S,
    )?;

    let mut best: Option<(i32, i32, i32, i32, i32)> = None;
    let min_area = (mat.cols() * mat.rows()) as f32 * 0.005;
    for i in 1..n {
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        if (area as f32) < min_area {
            continue;
        }
        if best.map_or(true, |(_, _, _, _, a)| area > a) {
            let bx = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
            let by = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
            let bw = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
            let bh = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
            best = Some((bx, by, bw, bh, area));
        }
    }

    Ok(best.map(|(bx, by, bw, bh, _)| RectNorm {
        x: bx as f32 / mat.cols() as f32,
        y: by as f32 / mat.rows() as f32,
        w: bw as f32 / mat.cols() as f32,
        h: bh as f32 / mat.rows() as f32,
    }))
}

fn generate_anchors_mp(input_size: u32) -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(2944);
    let strides = [8, 16, 32, 32];
    let nums = [2, 2, 6, 6];
    for (i, &stride) in strides.iter().enumerate() {
        let grid_size = input_size / stride;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let cx = (x as f32 + 0.5) / grid_size as f32;
                let cy = (y as f32 + 0.5) / grid_size as f32;
                for _ in 0..nums[i] {
                    anchors.push(Anchor {
                        x_center: cx,
                        y_center: cy,
                    });
                }
            }
        }
    }
    anchors
}
