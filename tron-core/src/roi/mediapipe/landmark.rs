use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{Affine2, Vec2};
use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use tron_api::{Frame, NoContext, PixelFormat, Processor, Rect, RoiResult, Size};

const DEFAULT_INPUT_SIZE: usize = 224;
const HAND_LANDMARKS: usize = 21;
const LANDMARK_COORDS: usize = 3;
const WRIST_LANDMARK: usize = 0;
const INDEX_MCP_LANDMARK: usize = 5;
const MIDDLE_MCP_LANDMARK: usize = 9;
const PINKY_MCP_LANDMARK: usize = 17;
const MEDIAPIPE_LANDMARK_SCALE: f32 = 1.0;
const LANDMARK_SILHOUETTE_MARGIN_OF_PALM_WIDTH: f32 = 0.20;
const LANDMARK_SILHOUETTE_MARGIN_OF_PALM_LENGTH: f32 = 0.10;
const PINKY_MCP_EDGE_EPSILON_PX: f32 = 4.0;
const PINKY_MCP_EDGE_EXTRA_MARGIN_OF_PALM_WIDTH: f32 = 0.12;

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

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct HandLandmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl HandLandmark {
    fn xy(self) -> Option<Vec2> {
        let point = Vec2::new(self.x, self.y);
        point.is_finite().then_some(point)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HandLandmarks {
    pub points: [HandLandmark; 21],
    pub presence: f32,
    pub handedness: Option<Handedness>,
    #[serde(skip)]
    pub timestamp: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Handedness {
    Left,
    Right,
}

impl HandLandmarks {
    pub fn bounding_roi(&self, frame_size: Size, scale: f32) -> Option<RoiResult> {
        landmark_bounding_roi(&self.points, frame_size, scale)
    }
}

fn landmark_bounding_roi(
    points: &[HandLandmark; HAND_LANDMARKS],
    frame_size: Size,
    scale: f32,
) -> Option<RoiResult> {
    let (min, max) = landmark_bounds(points)?;
    let [x0, y0] = min.to_array();
    let [x1, y1] = max.to_array();
    let scale = scale.max(1.0);
    let margin = landmark_silhouette_margin(points).unwrap_or(0.0).max(0.0);
    let (left_edge_margin, right_edge_margin) =
        pinky_edge_margins(points, min.x, max.x).unwrap_or((0.0, 0.0));
    let left_margin = margin + left_edge_margin.max(0.0);
    let right_margin = margin + right_edge_margin.max(0.0);
    let w = ((x1 - x0) + left_margin + right_margin) * scale;
    let h = ((y1 - y0) + margin * 2.0) * scale;
    let cx = (x0 - left_margin + x1 + right_margin) * 0.5;
    let cy = (y0 + y1) * 0.5;

    let rect_x0 = (cx - w * 0.5).floor().max(0.0).min(frame_size.width as f32) as u32;
    let rect_y0 = (cy - h * 0.5)
        .floor()
        .max(0.0)
        .min(frame_size.height as f32) as u32;
    let rect_x1 = (cx + w * 0.5).ceil().max(0.0).min(frame_size.width as f32) as u32;
    let rect_y1 = (cy + h * 0.5).ceil().max(0.0).min(frame_size.height as f32) as u32;

    let rect = Rect {
        x: rect_x0,
        y: rect_y0,
        size: Size {
            width: rect_x1.saturating_sub(rect_x0),
            height: rect_y1.saturating_sub(rect_y0),
        },
    };

    tracing::debug!(
        "ROI: bounds=({:.1}, {:.1}, {:.1}, {:.1}), scale={:.1}, margin={:.1}, pinky_edge=({:.1}, {:.1}), rect={:?}",
        x0,
        y0,
        x1,
        y1,
        scale,
        margin,
        left_edge_margin,
        right_edge_margin,
        rect
    );

    (rect.size.width > 0 && rect.size.height > 0).then_some(RoiResult {
        rect,
        oriented_box: None,
    })
}

fn pinky_edge_margins(
    points: &[HandLandmark; HAND_LANDMARKS],
    x0: f32,
    x1: f32,
) -> Option<(f32, f32)> {
    let pinky = points[PINKY_MCP_LANDMARK].xy()?;
    let palm_width = distance(points[INDEX_MCP_LANDMARK], points[PINKY_MCP_LANDMARK])?;
    let extra = palm_width * PINKY_MCP_EDGE_EXTRA_MARGIN_OF_PALM_WIDTH;
    if extra <= 0.0 || !extra.is_finite() {
        return None;
    }

    let left = if (pinky.x - x0).abs() <= PINKY_MCP_EDGE_EPSILON_PX {
        extra
    } else {
        0.0
    };
    let right = if (x1 - pinky.x).abs() <= PINKY_MCP_EDGE_EPSILON_PX {
        extra
    } else {
        0.0
    };
    Some((left, right))
}

fn landmark_silhouette_margin(points: &[HandLandmark; HAND_LANDMARKS]) -> Option<f32> {
    let palm_width = distance(points[INDEX_MCP_LANDMARK], points[PINKY_MCP_LANDMARK]);
    let palm_length = distance(points[WRIST_LANDMARK], points[MIDDLE_MCP_LANDMARK]);
    let margin = palm_width
        .map(|width| width * LANDMARK_SILHOUETTE_MARGIN_OF_PALM_WIDTH)
        .into_iter()
        .chain(palm_length.map(|length| length * LANDMARK_SILHOUETTE_MARGIN_OF_PALM_LENGTH))
        .fold(0.0_f32, f32::max);
    (margin > 0.0 && margin.is_finite()).then_some(margin)
}

fn distance(a: HandLandmark, b: HandLandmark) -> Option<f32> {
    Some((a.xy()? - b.xy()?).length())
}

fn landmark_bounds(points: &[HandLandmark; HAND_LANDMARKS]) -> Option<(Vec2, Vec2)> {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    let mut found = false;
    for point in *points {
        if let Some(point) = point.xy() {
            min = min.min(point);
            max = max.max(point);
            found = true;
        }
    }
    found.then_some((min, max))
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
        let (landmarks_output, presence_output, handedness_output) = classify_outputs(&session)?;
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
        if landmarks.len() < HAND_LANDMARKS * LANDMARK_COORDS {
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
            .take(HAND_LANDMARKS * LANDMARK_COORDS)
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let mut points = [HandLandmark::default(); HAND_LANDMARKS];
        for i in 0..HAND_LANDMARKS {
            let base = i * LANDMARK_COORDS;
            let x = landmarks[base];
            let y = landmarks[base + 1];
            let z = landmarks[base + 2];
            let (nx, ny, nz) = if raw_max < 2.0 {
                (x, y, z)
            } else {
                (
                    x / self.input_size as f32,
                    y / self.input_size as f32,
                    z / self.input_size as f32,
                )
            };

            // Filter out points that are exactly at the origin (0, 0) in crop space.
            // These are usually invalid/uninitialized points from the model.
            if nx.abs() < f32::EPSILON && ny.abs() < f32::EPSILON {
                points[i] = HandLandmark {
                    x: f32::NAN,
                    y: f32::NAN,
                    z: f32::NAN,
                };
                continue;
            }

            let [frame_x, frame_y] = crop.transform_point2(Vec2::new(nx, ny)).to_array();
            points[i] = HandLandmark {
                x: frame_x,
                y: frame_y,
                z: nz,
            };
            tracing::trace!(
                "landmark {}: crop=({:.4}, {:.4}), frame=({:.1}, {:.1})",
                i,
                nx,
                ny,
                points[i].x,
                points[i].y
            );
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

fn classify_outputs(session: &Session) -> Result<(usize, Option<usize>, Option<usize>)> {
    let mut landmarks_output = None;
    let mut presence_output = None;
    let mut handedness_output = None;

    let outputs = session.outputs();

    // 1. Try to find by name (MediaPipe/TFLite-to-ONNX conventions)
    for (i, output) in outputs.iter().enumerate() {
        let name = output.name().to_ascii_lowercase();
        let elements = match output.dtype() {
            ValueType::Tensor { shape, .. } => shape.num_elements(),
            _ => 0,
        };

        if landmarks_output.is_none()
            && (name.contains("landmark") || name.contains("identity"))
            && !name.contains("world")
            && elements == 63
        {
            landmarks_output = Some(i);
        }

        if presence_output.is_none()
            && (name.contains("presence") || name.contains("score"))
            && elements == 1
        {
            presence_output = Some(i);
        }

        if handedness_output.is_none() && name.contains("handed") && elements == 1 {
            handedness_output = Some(i);
        }
    }

    // 2. Fallback to shape-based signatures if names were generic/missing
    if landmarks_output.is_none() {
        landmarks_output = outputs.iter().position(
            |o| matches!(o.dtype(), ValueType::Tensor { shape, .. } if shape.num_elements() == 63),
        );
    }
    if presence_output.is_none() {
        presence_output = outputs.iter().position(
            |o| matches!(o.dtype(), ValueType::Tensor { shape, .. } if shape.num_elements() == 1),
        );
    }

    let Some(landmarks_output) = landmarks_output else {
        let outputs = outputs
            .iter()
            .enumerate()
            .map(|(i, output)| {
                let shape = match output.dtype() {
                    ValueType::Tensor { shape, .. } => format!("{shape:?}"),
                    _ => "not a tensor".to_string(),
                };
                format!("{i}:{}:{shape}", output.name())
            })
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "MediaPipe hand landmark model has no 21x3 landmark output tensor; outputs: {outputs}"
        );
    };

    Ok((landmarks_output, presence_output, handedness_output))
}

fn crop_from_roi(roi: Option<RoiResult>, frame_size: Size) -> Affine2 {
    let Some(roi) = roi else {
        return Affine2::from_cols(
            Vec2::new(frame_size.width as f32, 0.0),
            Vec2::new(0.0, frame_size.height as f32),
            Vec2::ZERO,
        );
    };
    if let Some(oriented_box) = roi.oriented_box {
        let [c0, c1, _, c3] = oriented_box.corners.map(Vec2::from_array);
        return Affine2::from_cols(c1 - c0, c3 - c0, c0);
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
    Affine2::from_cols(
        Vec2::new(x1 - x0, 0.0),
        Vec2::new(0.0, y1 - y0),
        Vec2::new(x0, y0),
    )
}

fn preprocess_bgra(
    frame: Frame<'_>,
    crop: Affine2,
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
            let [src_xf, src_yf] = crop.transform_point2(Vec2::new(nx, ny)).to_array();
            let src_xf = src_xf - 0.5;
            let src_yf = src_yf - 0.5;

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
    fn degenerate_landmark_roi_is_present() {
        let landmarks = HandLandmarks {
            points: [HandLandmark {
                x: 10.2,
                y: 20.8,
                z: 0.0,
            }; 21],
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
        // For a single point at (10.2, 20.8):
        // x0 = floor(10.2) = 10, y0 = floor(20.8) = 20
        // x1 = ceil(10.2) = 11, y1 = ceil(20.8) = 21
        assert_eq!(roi.rect.x, 10);
        assert_eq!(roi.rect.y, 20);
        assert_eq!(roi.rect.size.width, 1);
        assert_eq!(roi.rect.size.height, 1);
    }
}
