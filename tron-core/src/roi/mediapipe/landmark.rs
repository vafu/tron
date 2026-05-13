use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use tron_api::{Frame, NoContext, OrientedBoundingBox, PixelFormat, Processor, RoiResult, Size};

const DEFAULT_INPUT_SIZE: usize = 224;
const WRIST_LANDMARK: usize = 0;
const MIDDLE_MCP_LANDMARK: usize = 9;
const MEDIAPIPE_LANDMARK_SCALE: f32 = 2.0;
const MEDIAPIPE_LANDMARK_SHIFT_Y: f32 = -0.1;

#[derive(Clone, Debug)]
pub struct MediaPipeHandLandmarkConfig {
    pub min_presence: f32,
    pub roi_scale: f32,
}

impl Default for MediaPipeHandLandmarkConfig {
    fn default() -> Self {
        Self {
            min_presence: 0.5,
            roi_scale: MEDIAPIPE_LANDMARK_SCALE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MediaPipeHandLandmarkInput<'a> {
    pub frame: Frame<'a>,
    pub roi: Option<RoiResult>,
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
        landmark_bounds(&self.points).and_then(|bounds| {
            bounds.to_mediapipe_roi(
                self.points[WRIST_LANDMARK],
                self.points[MIDDLE_MCP_LANDMARK],
                frame_size,
                scale,
            )
        })
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
    fn to_mediapipe_roi(
        self,
        wrist: HandLandmark,
        middle_mcp: HandLandmark,
        frame_size: Size,
        scale: f32,
    ) -> Option<RoiResult> {
        if self.x1 <= self.x0 || self.y1 <= self.y0 {
            return None;
        }

        // MediaPipe Hand Landmark model expects fingertips at the top (y=0).
        // The orientation axis should point from middle-MCP to wrist.
        let axis_y = [wrist.x - middle_mcp.x, wrist.y - middle_mcp.y];
        let axis_len = hypot(axis_y[0], axis_y[1]);
        if axis_len < 1.0 || !axis_len.is_finite() {
            return self.to_axis_aligned_roi(frame_size, scale);
        }
        let axis_y = [axis_y[0] / axis_len, axis_y[1] / axis_len];
        // axis_x should be axis_y rotated 90 deg clockwise to maintain right-handedness.
        // If axis_y is backward (down), axis_x should be right.
        let axis_x = [axis_y[1], -axis_y[0]];
        let origin = [middle_mcp.x, middle_mcp.y];
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for corner in [
            [self.x0, self.y0],
            [self.x1, self.y0],
            [self.x1, self.y1],
            [self.x0, self.y1],
        ] {
            let delta = [corner[0] - origin[0], corner[1] - origin[1]];
            let x = dot(delta, axis_x);
            let y = dot(delta, axis_y);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
            return None;
        }
        let raw_width = (max_x - min_x).max(1.0);
        let raw_height = (max_y - min_y).max(1.0);
        let center_x = (min_x + max_x) * 0.5;
        // Shift center along axis_y (middle-MCP to wrist).
        // Negative shift moves towards fingertips.
        let center_y = (min_y + max_y) * 0.5 + MEDIAPIPE_LANDMARK_SHIFT_Y * raw_height;
        let side = raw_width.max(raw_height) * scale.max(1.0);
        let half_side = side * 0.5;
        let center = add(add(origin, mul(axis_x, center_x)), mul(axis_y, center_y));
        let oriented_box = OrientedBoundingBox {
            corners: [
                add(
                    add(center, mul(axis_x, -half_side)),
                    mul(axis_y, -half_side),
                ),
                add(add(center, mul(axis_x, half_side)), mul(axis_y, -half_side)),
                add(add(center, mul(axis_x, half_side)), mul(axis_y, half_side)),
                add(add(center, mul(axis_x, -half_side)), mul(axis_y, half_side)),
            ],
        };
        oriented_box
            .enclosing_rect(frame_size)
            .map(|rect| RoiResult {
                rect,
                oriented_box: Some(oriented_box),
            })
    }

    fn to_axis_aligned_roi(self, frame_size: Size, scale: f32) -> Option<RoiResult> {
        let scale = scale.max(1.0);
        let cx = (self.x0 + self.x1) * 0.5;
        let cy = (self.y0 + self.y1) * 0.5;
        let side = (self.x1 - self.x0).max(self.y1 - self.y0).max(1.0) * scale;
        let half = side * 0.5;
        let oriented_box = OrientedBoundingBox {
            corners: [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
        };
        oriented_box
            .enclosing_rect(frame_size)
            .map(|rect| RoiResult {
                rect,
                oriented_box: Some(oriented_box),
            })
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
        let crop = crop_from_roi(input.roi, input.frame.meta.size);
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
                x: crop.frame_x(nx, ny),
                y: crop.frame_y(nx, ny),
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
    let mut landmarks_priority = -1;
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
        if elements == 63 {
            let priority = if name.contains("world") {
                0
            } else if name.contains("image") || name.contains("landmark") {
                2
            } else {
                1
            };
            if priority > landmarks_priority {
                landmarks_priority = priority;
                landmarks_output = index;
                landmarks_size = elements;
            }
        } else if elements > landmarks_size && landmarks_priority < 0 {
            landmarks_size = elements;
            landmarks_output = index;
        }
    }
    (landmarks_output, presence_output, handedness_output)
}

#[derive(Clone, Copy, Debug)]
struct Crop {
    origin: [f32; 2],
    x_axis: [f32; 2],
    y_axis: [f32; 2],
}

impl Crop {
    fn frame_x(self, x: f32, y: f32) -> f32 {
        self.origin[0] + x * self.x_axis[0] + y * self.y_axis[0]
    }

    fn frame_y(self, x: f32, y: f32) -> f32 {
        self.origin[1] + x * self.x_axis[1] + y * self.y_axis[1]
    }
}

fn crop_from_roi(roi: Option<RoiResult>, frame_size: Size) -> Crop {
    let Some(roi) = roi else {
        return Crop {
            origin: [0.0, 0.0],
            x_axis: [frame_size.width as f32, 0.0],
            y_axis: [0.0, frame_size.height as f32],
        };
    };
    if let Some(oriented_box) = roi.oriented_box {
        let [c0, c1, _, c3] = oriented_box.corners;
        return Crop {
            origin: c0,
            x_axis: [c1[0] - c0[0], c1[1] - c0[1]],
            y_axis: [c3[0] - c0[0], c3[1] - c0[1]],
        };
    }

    let frame_w = frame_size.width.max(1) as f32;
    let frame_h = frame_size.height.max(1) as f32;
    let cx = roi.rect.x as f32 + roi.rect.size.width as f32 * 0.5;
    let cy = roi.rect.y as f32 + roi.rect.size.height as f32 * 0.5;
    let half = roi.rect.size.width.max(roi.rect.size.height) as f32 * 0.5;
    let x0 = (cx - half).clamp(0.0, frame_w);
    let y0 = (cy - half).clamp(0.0, frame_h);
    let x1 = (cx + half).clamp(0.0, frame_w);
    let y1 = (cy + half).clamp(0.0, frame_h);
    Crop {
        origin: [x0, y0],
        x_axis: [x1 - x0, 0.0],
        y_axis: [0.0, y1 - y0],
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
    let pixels = frame.view()?;
    output.fill(0.0);
    let input_sizef = input_size as f32;
    for y in 0..input_size {
        let ny = (y as f32 + 0.5) / input_sizef;
        for x in 0..input_size {
            let nx = (x as f32 + 0.5) / input_sizef;
            let src_xf = crop.frame_x(nx, ny) - 0.5;
            let src_yf = crop.frame_y(nx, ny) - 0.5;

            let x0 = src_xf.floor() as isize;
            let y0 = src_yf.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let dx = src_xf - x0 as f32;
            let dy = src_yf - y0 as f32;

            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;

            for (iy, weight_y) in [(y0, 1.0 - dy), (y1, dy)] {
                let iy = iy.clamp(0, source_h as isize - 1) as usize;
                for (ix, weight_x) in [(x0, 1.0 - dx), (x1, dx)] {
                    let ix = ix.clamp(0, source_w as isize - 1) as usize;
                    let weight = weight_x * weight_y;
                    r += pixels[[iy, ix, 2]] as f32 * weight;
                    g += pixels[[iy, ix, 1]] as f32 * weight;
                    b += pixels[[iy, ix, 0]] as f32 * weight;
                }
            }

            let dst = y * input_size + x;
            output[dst] = r / 255.0;
            output[input_size * input_size + dst] = g / 255.0;
            output[2 * input_size * input_size + dst] = b / 255.0;
        }
    }
    Ok(())
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn mul(v: [f32; 2], scale: f32) -> [f32; 2] {
    [v[0] * scale, v[1] * scale]
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
