use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use tron_api::{Frame, NoContext, PixelFormat, Processor, Rect, RoiResult, Size};

const DEFAULT_INPUT_SIZE: usize = 224;

#[derive(Clone, Debug)]
pub struct MediaPipeHandLandmarkConfig {
    pub min_presence: f32,
    pub roi_scale: f32,
}

impl Default for MediaPipeHandLandmarkConfig {
    fn default() -> Self {
        Self {
            min_presence: 0.5,
            roi_scale: 1.2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MediaPipeHandLandmarkInput<'a> {
    pub frame: Frame<'a>,
    pub roi: Option<Rect>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HandLandmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug)]
pub struct HandLandmarks {
    pub points: [HandLandmark; 21],
    pub presence: f32,
    pub handedness: Option<Handedness>,
    pub timestamp: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

impl HandLandmarks {
    pub fn bounding_roi(&self, frame_size: Size, scale: f32) -> Option<RoiResult> {
        landmark_bounds(&self.points).and_then(|bounds| bounds.to_rect(frame_size, scale))
    }
}

#[derive(Clone, Copy, Debug)]
struct LandmarkBounds {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl LandmarkBounds {
    fn to_rect(self, frame_size: Size, scale: f32) -> Option<RoiResult> {
        if self.x1 <= self.x0 || self.y1 <= self.y0 {
            return None;
        }

        let scale = scale.max(1.0);
        let cx = (self.x0 + self.x1) * 0.5;
        let cy = (self.y0 + self.y1) * 0.5;
        let width = (self.x1 - self.x0) * scale;
        let height = (self.y1 - self.y0) * scale;
        let half_w = width.max(1.0) * 0.5;
        let half_h = height.max(1.0) * 0.5;
        rect_from_f32(
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h,
            frame_size,
        )
        .map(|rect| RoiResult { rect })
    }
}

fn landmark_bounds(points: &[HandLandmark; 21]) -> Option<LandmarkBounds> {
    let mut x0 = f32::INFINITY;
    let mut y0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for point in *points {
        if !point.x.is_finite() || !point.y.is_finite() {
            continue;
        }
        x0 = x0.min(point.x);
        y0 = y0.min(point.y);
        x1 = x1.max(point.x);
        y1 = y1.max(point.y);
    }
    if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return None;
    }
    Some(LandmarkBounds { x0, y0, x1, y1 })
}

pub struct MediaPipeHandLandmarkProcessor {
    session: Session,
    config: MediaPipeHandLandmarkConfig,
    input_name: String,
    input_size: usize,
    landmarks_output: usize,
    presence_output: Option<usize>,
    handedness_output: Option<usize>,
    input: Vec<f32>,
}

impl MediaPipeHandLandmarkProcessor {
    pub fn new(model_path: impl AsRef<Path>, config: MediaPipeHandLandmarkConfig) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path.as_ref())
            .with_context(|| {
                format!(
                    "load MediaPipe hand landmark model {:?}",
                    model_path.as_ref()
                )
            })?;
        let input = session
            .inputs()
            .first()
            .context("MediaPipe hand landmark model has no inputs")?;
        let input_name = input.name().to_owned();
        let input_size = square_input_size(input.dtype()).unwrap_or(DEFAULT_INPUT_SIZE);
        let (landmarks_output, presence_output, handedness_output) = classify_outputs(&session);
        Ok(Self {
            session,
            config,
            input_name,
            input_size,
            landmarks_output,
            presence_output,
            handedness_output,
            input: vec![0.0; 3 * input_size * input_size],
        })
    }

    pub fn config(&self) -> &MediaPipeHandLandmarkConfig {
        &self.config
    }
}

impl Processor<MediaPipeHandLandmarkInput<'_>> for MediaPipeHandLandmarkProcessor {
    type Output = Option<HandLandmarks>;

    fn process(
        &mut self,
        input: MediaPipeHandLandmarkInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        anyhow::ensure!(
            input.frame.format == PixelFormat::Bgra8,
            "MediaPipe hand landmark processor expects BGRA8 RGB frames, got {:?}",
            input.frame.format
        );
        let crop = square_crop(input.roi, input.frame.meta.size);
        preprocess_bgra(input.frame, crop, self.input_size, &mut self.input)?;
        let tensor =
            TensorRef::from_array_view(([1, 3, self.input_size, self.input_size], &*self.input))?;
        let outputs = self.session.run(vec![(self.input_name.as_str(), tensor)])?;
        let (_, landmarks) = outputs[self.landmarks_output]
            .try_extract_tensor::<f32>()
            .context("extract MediaPipe hand landmarks")?;
        if landmarks.len() < 63 {
            return Ok(None);
        }

        let presence = self
            .presence_output
            .and_then(|index| outputs[index].try_extract_tensor::<f32>().ok())
            .and_then(|(_, values)| values.first().copied())
            .unwrap_or(1.0);
        if presence < self.config.min_presence {
            return Ok(None);
        }

        let raw_max = landmarks
            .iter()
            .take(63)
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let mut points = [HandLandmark::default(); 21];
        for i in 0..21 {
            let x = landmarks[i * 3];
            let y = landmarks[i * 3 + 1];
            let z = landmarks[i * 3 + 2];
            let (nx, ny, nz) = if raw_max < 2.0 {
                (x, y, z)
            } else {
                (
                    x / self.input_size as f32,
                    y / self.input_size as f32,
                    z / self.input_size as f32,
                )
            };
            points[i] = HandLandmark {
                x: (crop.x + nx * crop.w) * input.frame.meta.size.width as f32,
                y: (crop.y + ny * crop.h) * input.frame.meta.size.height as f32,
                z: nz,
            };
        }

        let handedness = self
            .handedness_output
            .and_then(|index| outputs[index].try_extract_tensor::<f32>().ok())
            .and_then(|(_, values)| values.first().copied())
            .map(|score| {
                if score > 0.5 {
                    Handedness::Right
                } else {
                    Handedness::Left
                }
            });

        Ok(Some(HandLandmarks {
            points,
            presence,
            handedness,
            timestamp: Instant::now(),
        }))
    }
}

fn square_input_size(value_type: &ValueType) -> Option<usize> {
    let ValueType::Tensor { shape, .. } = value_type else {
        return None;
    };
    let height = *shape.get(2)?;
    let width = *shape.get(3)?;
    if height > 0 && height == width {
        Some(height as usize)
    } else {
        None
    }
}

fn classify_outputs(session: &Session) -> (usize, Option<usize>, Option<usize>) {
    let mut landmarks_output = 0;
    let mut landmarks_size = 0;
    let mut presence_output = None;
    let mut handedness_output = None;
    for (index, output) in session.outputs().iter().enumerate() {
        let name = output.name().to_ascii_lowercase();
        if name.contains("presence") || name.contains("score") {
            presence_output = Some(index);
        }
        if name.contains("handed") {
            handedness_output = Some(index);
        }
        let elements = match output.dtype() {
            ValueType::Tensor { shape, .. } => shape.num_elements(),
            _ => 0,
        };
        if elements > landmarks_size {
            landmarks_size = elements;
            landmarks_output = index;
        }
    }
    (landmarks_output, presence_output, handedness_output)
}

#[derive(Clone, Copy, Debug)]
struct Crop {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn square_crop(roi: Option<Rect>, frame_size: Size) -> Crop {
    let Some(roi) = roi else {
        return Crop {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
    };

    let frame_w = frame_size.width.max(1) as f32;
    let frame_h = frame_size.height.max(1) as f32;
    let cx = (roi.x as f32 + roi.size.width as f32 * 0.5) / frame_w;
    let cy = (roi.y as f32 + roi.size.height as f32 * 0.5) / frame_h;
    let half_px = roi.size.width.max(roi.size.height) as f32 * 0.5;
    let half_x = (half_px / frame_w).clamp(0.0, 0.5);
    let half_y = (half_px / frame_h).clamp(0.0, 0.5);
    let w = (2.0 * half_x).min(1.0);
    let h = (2.0 * half_y).min(1.0);
    Crop {
        x: (cx - half_x).clamp(0.0, 1.0 - w),
        y: (cy - half_y).clamp(0.0, 1.0 - h),
        w,
        h,
    }
}

fn preprocess_bgra(
    frame: Frame<'_>,
    crop: Crop,
    input_size: usize,
    output: &mut [f32],
) -> Result<()> {
    let source_w = frame.meta.size.width as usize;
    let source_h = frame.meta.size.height as usize;
    anyhow::ensure!(source_w > 0 && source_h > 0, "empty RGB frame");
    anyhow::ensure!(
        frame.buffer.stride >= source_w * 4,
        "BGRA frame stride {} is smaller than width {}",
        frame.buffer.stride,
        source_w * 4
    );
    output.fill(0.0);
    let source_wf = source_w as f32;
    let source_hf = source_h as f32;
    let input_sizef = input_size as f32;
    for y in 0..input_size {
        let src_y = ((crop.y * source_hf) + (y as f32 + 0.5) * crop.h * source_hf / input_sizef)
            .floor()
            .clamp(0.0, (source_h - 1) as f32) as usize;
        for x in 0..input_size {
            let src_x = ((crop.x * source_wf) + (x as f32 + 0.5) * crop.w * source_wf / input_sizef)
                .floor()
                .clamp(0.0, (source_w - 1) as f32) as usize;
            let src = src_y * frame.buffer.stride + src_x * 4;
            let dst = y * input_size + x;
            output[dst] = frame.buffer.data[src + 2] as f32 / 255.0;
            output[input_size * input_size + dst] = frame.buffer.data[src + 1] as f32 / 255.0;
            output[2 * input_size * input_size + dst] = frame.buffer.data[src] as f32 / 255.0;
        }
    }
    Ok(())
}

fn rect_from_f32(x0: f32, y0: f32, x1: f32, y1: f32, bounds: Size) -> Option<Rect> {
    let x0 = x0.floor().max(0.0).min(bounds.width as f32) as u32;
    let y0 = y0.floor().max(0.0).min(bounds.height as f32) as u32;
    let x1 = x1.ceil().max(0.0).min(bounds.width as f32) as u32;
    let y1 = y1.ceil().max(0.0).min(bounds.height as f32) as u32;
    let width = x1.saturating_sub(x0);
    let height = y1.saturating_sub(y0);
    if width == 0 || height == 0 {
        return None;
    }
    Some(Rect {
        x: x0,
        y: y0,
        size: Size { width, height },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landmarks_bounding_roi_uses_points() {
        let mut points = [HandLandmark {
            x: 10.2,
            y: 20.8,
            z: 0.0,
        }; 21];
        points[1] = HandLandmark {
            x: 30.1,
            y: 40.2,
            z: 0.0,
        };
        let landmarks = HandLandmarks {
            points,
            presence: 1.0,
            handedness: None,
            timestamp: Instant::now(),
        };
        let roi = landmarks
            .bounding_roi(
                Size {
                    width: 100,
                    height: 100,
                },
                1.0,
            )
            .expect("roi");
        assert_eq!(roi.rect.x, 10);
        assert_eq!(roi.rect.y, 20);
        assert_eq!(roi.rect.size.width, 21);
        assert_eq!(roi.rect.size.height, 21);
    }

    #[test]
    fn degenerate_landmark_roi_is_absent() {
        let landmarks = HandLandmarks {
            points: [HandLandmark {
                x: 10.0,
                y: 20.0,
                z: 0.0,
            }; 21],
            presence: 1.0,
            handedness: None,
            timestamp: Instant::now(),
        };
        assert!(
            landmarks
                .bounding_roi(
                    Size {
                        width: 100,
                        height: 100,
                    },
                    1.0,
                )
                .is_none()
        );
    }
}
