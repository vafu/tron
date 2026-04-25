use super::HandLandmarker;
use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, Handedness, Image, PixelFormat, RectNorm, Vec3};
use anyhow::{Context, Result, anyhow};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel, SessionInputValue};
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;
use std::borrow::Cow;

pub struct MediaPipeHandLandmarker {
    session: Session,
    /// Cached I/O names from the loaded model.
    input_name: String,
    /// Square input side, read from the model.
    input_size: u32,
    landmarks_output: usize,
    presence_output: Option<usize>,
    handedness_output: Option<usize>,
    /// Throttled diagnostic counter — log every N frames.
    log_counter: u32,
}

impl MediaPipeHandLandmarker {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(num_cpus::get().min(4))
            .map_err(ort_err)?
            .commit_from_file(model_path.as_ref())
            .map_err(ort_err)
            .with_context(|| format!("load {}", model_path.as_ref().display()))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?;
        let input_name = input.name().to_string();
        // Pull the square input size from the model: NCHW expects shape [1, 3, S, S].
        let input_size = input
            .dtype()
            .tensor_shape()
            .and_then(|s| {
                let dims: &[i64] = s.as_ref();
                if dims.len() >= 4 && dims[2] > 0 && dims[3] > 0 {
                    Some(dims[2] as u32)
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("could not read input size from model"))?;

        // Heuristically locate outputs.
        let mut landmarks_output = 0usize;
        let mut landmarks_size = 0usize;
        let mut presence_output = None;
        let mut handedness_output = None;
        for (i, o) in session.outputs().iter().enumerate() {
            let elems = o
                .dtype()
                .tensor_shape()
                .map(|s| {
                    let dims: &[i64] = s.as_ref();
                    dims.iter().product::<i64>() as usize
                })
                .unwrap_or(0);
            if elems > landmarks_size {
                landmarks_size = elems;
                landmarks_output = i;
            }
            let lname = o.name().to_lowercase();
            if lname.contains("presence") || lname.contains("score") {
                presence_output = Some(i);
            }
            if lname.contains("handed") {
                handedness_output = Some(i);
            }
        }

        eprintln!(
            "mediapipe: loaded {}, in={} ({}×{}), lm-out=#{} ({} elems), pres={:?}, handed={:?}",
            model_path.as_ref().display(),
            input_name,
            input_size,
            input_size,
            landmarks_output,
            landmarks_size,
            presence_output,
            handedness_output,
        );

        Ok(Self {
            session,
            input_name,
            input_size,
            landmarks_output,
            presence_output,
            handedness_output,
            log_counter: 0,
        })
    }
}

impl HandLandmarker for MediaPipeHandLandmarker {
    fn run(&mut self, ctx: &FrameContext, roi: Option<RectNorm>) -> Option<HandLandmarks> {
        if ctx.rgb.format != PixelFormat::Rgba8 {
            return None;
        }
        let roi = roi.unwrap_or(RectNorm::FULL);
        // Expand ROI to a square in source-image pixel space (covering the whole hand).
        let crop = square_crop(&roi, ctx.rgb.width, ctx.rgb.height);
        let input = preprocess(&ctx.rgb, &crop, self.input_size);

        let tensor = match Tensor::from_array(input) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("mediapipe tensor: {e}");
                return None;
            }
        };
        
        let input_map: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![(self.input_name.as_str().into(), tensor.into())];
        let outputs = match self.session.run(input_map) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("mediapipe run: {e}");
                return None;
            }
        };

        // Landmarks tensor: indexable by usize.
        let lm_out = &outputs[self.landmarks_output];
        let (_shape, raw_slice) = match lm_out.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("mediapipe extract lm: {e}");
                return None;
            }
        };
        if raw_slice.len() < 63 {
            return None;
        }
        let raw: Vec<f32> = raw_slice.iter().copied().take(63).collect();
        let mut points = [Vec3::default(); 21];
        for i in 0..21 {
            let x = raw[i * 3];
            let y = raw[i * 3 + 1];
            let z = raw[i * 3 + 2];
            let s = self.input_size as f32;
            let (nx, ny, nz) = if max_abs(&raw) < 2.0 {
                (x, y, z)
            } else {
                (x / s, y / s, z / s)
            };
            // Map crop-local 0..1 → source-image normalized 0..1.
            points[i] = Vec3 {
                x: crop.x + nx * crop.w,
                y: crop.y + ny * crop.h,
                z: nz,
            };
        }

        let presence = self
            .presence_output
            .map(|i| &outputs[i])
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, s)| s.iter().copied().next())
            .unwrap_or(1.0);

        let handedness = self
            .handedness_output
            .map(|i| &outputs[i])
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .and_then(|(_, s)| s.iter().copied().next())
            .map(|s| {
                if s > 0.5 {
                    Handedness::Right
                } else {
                    Handedness::Left
                }
            })
            .unwrap_or(Handedness::Unknown);

        // Presence gate: drop low-confidence detections
        if presence < 0.5 {
            self.log_counter = self.log_counter.wrapping_add(1);
            if self.log_counter % 30 == 0 {
                eprintln!("mediapipe: presence={presence:.2} (gated)");
            }
            return None;
        }

        // Throttled diagnostics
        self.log_counter = self.log_counter.wrapping_add(1);
        if self.log_counter % 30 == 0 {
            let raw_max = max_abs(&raw);
            eprintln!(
                "mediapipe: presence={presence:.2} handed={handedness:?} raw_max={raw_max:.3} crop=[{:.2},{:.2} {:.2}x{:.2}] p0=({:.2},{:.2})",
                crop.x, crop.y, crop.w, crop.h, points[0].x, points[0].y
            );
        }

        Some(HandLandmarks {
            points,
            presence,
            handedness,
            timestamp: Instant::now(),
        })
    }
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |acc, x| acc.max(x.abs()))
}

/// Expand `roi` to a square (in source-image normalized coords) so the hand
/// isn't squashed by the resize. Stays inside [0,1].
fn square_crop(roi: &RectNorm, w: u32, h: u32) -> RectNorm {
    let cx = roi.x + roi.w * 0.5;
    let cy = roi.y + roi.h * 0.5;
    let half_px = (roi.w * w as f32).max(roi.h * h as f32) * 0.5;
    let half_x = (half_px / w as f32).clamp(0.0, 0.5);
    let half_y = (half_px / h as f32).clamp(0.0, 0.5);
    RectNorm {
        x: (cx - half_x).clamp(0.0, 1.0),
        y: (cy - half_y).clamp(0.0, 1.0),
        w: (2.0 * half_x).min(1.0),
        h: (2.0 * half_y).min(1.0),
    }
}

/// Crop, resize to size×size, normalize to [0,1], CHW.
fn preprocess(img: &Image, crop: &RectNorm, size: u32) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    let w = img.width as f32;
    let h = img.height as f32;
    let cx0 = crop.x * w;
    let cy0 = crop.y * h;
    let cw = crop.w * w;
    let ch = crop.h * h;
    let stride = (img.width * 4) as usize;
    let s = size as f32;

    for oy in 0..size {
        let sy = (cy0 + (oy as f32 + 0.5) * ch / s) as i32;
        let sy = sy.clamp(0, img.height as i32 - 1) as usize;
        for ox in 0..size {
            let sx = (cx0 + (ox as f32 + 0.5) * cw / s) as i32;
            let sx = sx.clamp(0, img.width as i32 - 1) as usize;
            let i = sy * stride + sx * 4;
            out[[0, 0, oy as usize, ox as usize]] = img.data[i] as f32 / 255.0;
            out[[0, 1, oy as usize, ox as usize]] = img.data[i + 1] as f32 / 255.0;
            out[[0, 2, oy as usize, ox as usize]] = img.data[i + 2] as f32 / 255.0;
        }
    }
    out
}
