use anyhow::{Context, Result, anyhow};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel, SessionInputValue};
use ort::value::Tensor;
use std::path::Path;
use opencv::core::{Mat, CV_32F, Point, Scalar, CV_8UC1};
use opencv::imgproc;
use opencv::prelude::*;
use crate::types::{RectNorm, Image, FrameContext};
use crate::calib;
use super::RoiHinter;
use std::borrow::Cow;

pub struct PalmDetector {
    session: Session,
    input_name: String,
    input_size: u32,
    anchors: Vec<Anchor>,
    pad: f32,
    frame: u64,
    /// Most recent IR-diff seen — reused when the current frame has none
    /// (e.g. flashlight just toggled off). Without this, the detector falls
    /// through to the next hinter on every flashlight-off frame and the ROI
    /// geometry flips between detector- and tracker-sized rects.
    last_diff: Option<Image>,
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    x_center: f32,
    y_center: f32,
}

impl PalmDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("ort: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("ort: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("ort: {e}"))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| anyhow!("ort: {e}"))
            .with_context(|| format!("load palm detector {}", model_path.as_ref().display()))?;

        let input = session.inputs().first().ok_or_else(|| anyhow!("no inputs"))?;
        let input_name = input.name().to_string();
        let input_size = input.dtype().tensor_shape().and_then(|s| {
            let dims: &[i64] = s.as_ref();
            if dims.len() >= 4 { Some(dims[2] as u32) } else { None }
        }).ok_or_else(|| anyhow!("invalid input shape"))?;

        let anchors = generate_anchors_mp(input_size);

        Ok(Self {
            session,
            input_name,
            input_size,
            anchors,
            pad: 0.20,
            frame: 0,
            last_diff: None,
        })
    }

    fn run_model(&mut self, signal: &Mat) -> Result<Option<RectNorm>> {
        let w = signal.cols() as f32;
        let h = signal.rows() as f32;
        let scale = (self.input_size as f32 / w).min(self.input_size as f32 / h);
        let nw = (w * scale) as i32;
        let nh = (h * scale) as i32;
        
        let mut resized = Mat::default();
        imgproc::resize(signal, &mut resized, opencv::core::Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        
        let mut letterboxed = Mat::new_rows_cols_with_default(self.input_size as i32, self.input_size as i32, CV_8UC1, Scalar::all(0.0))?;
        let dx = (self.input_size as i32 - nw) / 2;
        let dy = (self.input_size as i32 - nh) / 2;
        {
            let mut roi = Mat::roi_mut(&mut letterboxed, opencv::core::Rect::new(dx, dy, nw, nh))?;
            resized.copy_to(&mut roi)?;
        }

        let mut float_mat = Mat::default();
        letterboxed.convert_to(&mut float_mat, CV_32F, 1.0 / 127.5, -1.0)?;

        let mut input = Array4::<f32>::zeros((1, 3, self.input_size as usize, self.input_size as usize));
        let data_slice = float_mat.data_bytes()?;
        for y in 0..self.input_size as usize {
            for x in 0..self.input_size as usize {
                let offset = (y * self.input_size as usize + x) * 4;
                let bytes = &data_slice[offset .. offset + 4];
                let val = f32::from_ne_bytes(bytes.try_into()?);
                input[[0, 0, y, x]] = val;
                input[[0, 1, y, x]] = val;
                input[[0, 2, y, x]] = val;
            }
        }

        let tensor = Tensor::from_array(input)?;
        let input_map: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![(self.input_name.as_str().into(), tensor.into())];
        let outputs = self.session.run(input_map).map_err(|e| anyhow!("ort run: {e}"))?;

        let scores_out = outputs.iter().find(|o| {
            let shape = o.1.dtype().tensor_shape();
            shape.map_or(false, |s| s.as_ref().last() == Some(&1))
        }).context("scores not found")?;
        
        let boxes_out = outputs.iter().find(|o| {
            let shape = o.1.dtype().tensor_shape();
            shape.map_or(false, |s| s.as_ref().last() == Some(&18))
        }).context("boxes not found")?;

        let (_, raw_scores) = scores_out.1.try_extract_tensor::<f32>().map_err(|e| anyhow!("ort extract: {e}"))?;
        let (_, raw_boxes) = boxes_out.1.try_extract_tensor::<f32>().map_err(|e| anyhow!("ort extract: {e}"))?;

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

        let best_conf = best_idx.map(|_| 1.0 / (1.0 + (-best_score).exp())).unwrap_or(0.0);
        if self.frame % 30 == 0 {
            eprintln!("palm: best_conf={best_conf:.3} (thresh 0.5)");
        }
        if let Some(i) = best_idx {
            let conf = best_conf;
            if conf > 0.5 {
                let b = &boxes[i * 18 .. (i + 1) * 18];
                let anchor = &self.anchors[i];
                
                let s = self.input_size as f32;
                let cx_norm = b[0] / s + anchor.x_center;
                let cy_norm = b[1] / s + anchor.y_center;
                let w_norm  = b[2] / s;
                let h_norm  = b[3] / s;
                
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
}

impl RoiHinter for PalmDetector {
    fn hint(&mut self, ctx: &FrameContext) -> (Option<RectNorm>, Option<Image>) {
        self.frame = self.frame.wrapping_add(1);
        if let Some(d) = &ctx.ir_diff {
            self.last_diff = Some(d.clone());
        }
        let signal_img = match ctx.ir_diff.as_ref().or(self.last_diff.as_ref()) {
            Some(d) => d,
            None => {
                if self.frame % 30 == 0 {
                    eprintln!("palm: skipped — no ir_diff (flashlight={})", ctx.ir_flashlight_on);
                }
                return (None, None);
            }
        };

        let w = signal_img.width as i32;
        let h = signal_img.height as i32;
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(h, w, CV_8UC1, signal_img.data.as_ptr() as *mut _).unwrap()
        };

        match self.run_model(&mat) {
            Ok(Some(rect)) => {
                let mapped = calib::current().map_rect(rect).padded(self.pad);
                (Some(mapped), None)
            }
            Ok(None) => {
                match find_blob_fallback(&mat) {
                    Ok(Some(rect)) => {
                        let mapped = calib::current().map_rect(rect).padded(self.pad);
                        if self.frame % 30 == 0 {
                            eprintln!("palm: model rejected, blob fallback rect=[{:.2},{:.2} {:.2}x{:.2}]", mapped.x, mapped.y, mapped.w, mapped.h);
                        }
                        (Some(mapped), None)
                    }
                    _ => {
                        if self.frame % 30 == 0 {
                            eprintln!("palm: no detection (model rejected, blob empty)");
                        }
                        (None, None)
                    }
                }
            }
            Err(e) => {
                eprintln!("palm detector error: {e}");
                (None, None)
            }
        }
    }
}

fn find_blob_fallback(mat: &Mat) -> Result<Option<RectNorm>> {
    let mut binary = Mat::default();
    imgproc::threshold(mat, &mut binary, 0.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;

    let k3 = imgproc::get_structuring_element(imgproc::MORPH_RECT, opencv::core::Size::new(3, 3), Point::new(-1, -1))?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(&binary, &mut opened, imgproc::MORPH_OPEN, &k3, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::default())?;

    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let mut labels = Mat::default();
    let n = imgproc::connected_components_with_stats(&opened, &mut labels, &mut stats, &mut centroids, 8, opencv::core::CV_32S)?;

    let mut best: Option<(i32, i32, i32, i32, i32)> = None;
    let min_area = (mat.cols() * mat.rows()) as f32 * 0.005;
    for i in 1..n {
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        if (area as f32) < min_area { continue; }
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
    let nums    = [2, 2, 6, 6];
    for (i, &stride) in strides.iter().enumerate() {
        let grid_size = input_size / stride;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let cx = (x as f32 + 0.5) / grid_size as f32;
                let cy = (y as f32 + 0.5) / grid_size as f32;
                for _ in 0..nums[i] {
                    anchors.push(Anchor { x_center: cx, y_center: cy });
                }
            }
        }
    }
    anchors
}
