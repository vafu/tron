use std::path::Path;

use anyhow::{Context, Result};
use glam::Vec2;
use ort::session::Session;
use ort::value::TensorRef;
use tron_api::{
    Frame, NoContext, OrientedBoundingBox, PixelFormat, Processor, Rect, RoiResult, Size,
};

const INPUT_SIZE: usize = 256;
const SCORE_CLIP: f32 = 100.0;
const PALM_KEYPOINTS: usize = 7;
const WRIST_KEYPOINT: usize = 0;
const MIDDLE_MCP_KEYPOINT: usize = 2;
const MEDIAPIPE_PALM_SHIFT_Y: f32 = -0.5;
const MEDIAPIPE_PALM_SCALE: f32 = 2.6;

#[derive(Clone, Debug)]
pub struct MediaPipeRoiConfig {
    pub min_score: f32,
    pub box_scale: f32,
}

impl Default for MediaPipeRoiConfig {
    fn default() -> Self {
        Self {
            min_score: 0.75,
            box_scale: MEDIAPIPE_PALM_SCALE,
        }
    }
}

pub struct MediaPipeRoiProcessor {
    session: Session,
    config: MediaPipeRoiConfig,
    input: Vec<f32>,
    anchors: Vec<Anchor>,
}

impl MediaPipeRoiProcessor {
    pub fn new(model_path: impl AsRef<Path>, config: MediaPipeRoiConfig) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("load MediaPipe hand detector {:?}", model_path.as_ref()))?;
        Ok(Self {
            session,
            config,
            input: vec![0.0; 3 * INPUT_SIZE * INPUT_SIZE],
            anchors: generate_palm_anchors(),
        })
    }
}

impl Processor<Frame<'_>> for MediaPipeRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(&mut self, input: Frame<'_>, _context: NoContext) -> Result<Self::Output> {
        anyhow::ensure!(
            input.format == PixelFormat::Bgra8,
            "MediaPipe ROI processor expects BGRA8 RGB frames, got {:?}",
            input.format
        );
        let resize = preprocess_bgra(input, &mut self.input)?;
        let tensor = TensorRef::from_array_view(([1, 3, INPUT_SIZE, INPUT_SIZE], &*self.input))?;
        let outputs = self.session.run(ort::inputs!["image" => tensor])?;
        let (_, coords) = outputs["box_coords"]
            .try_extract_tensor::<f32>()
            .context("extract MediaPipe box_coords")?;
        let (_, scores) = outputs["box_scores"]
            .try_extract_tensor::<f32>()
            .context("extract MediaPipe box_scores")?;
        let detection = best_detection(coords, scores, &self.anchors, self.config.min_score);
        Ok(detection.map(|detection| {
            let oriented_box =
                detection.to_oriented_box(resize, input.meta.size, self.config.box_scale);
            let rect = oriented_box
                .and_then(|bbox| bbox.enclosing_rect(input.meta.size))
                .or_else(|| detection.to_frame_rect(resize, input.meta.size, self.config.box_scale))
                .unwrap_or(Rect {
                    x: 0,
                    y: 0,
                    size: input.meta.size,
                });
            RoiResult { rect, oriented_box }
        }))
    }
}

#[derive(Clone, Copy, Debug)]
struct ResizeMapping {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    score: f32,
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
    keypoints: [[f32; 2]; PALM_KEYPOINTS],
}

impl Detection {
    fn to_frame_rect(
        self,
        resize: ResizeMapping,
        frame_size: Size,
        box_scale: f32,
    ) -> Option<Rect> {
        let width = (self.width * box_scale).max(1.0 / INPUT_SIZE as f32);
        let height = (self.height * box_scale).max(1.0 / INPUT_SIZE as f32);
        let x0 = (self.x_center - width * 0.5) * resize.scale * INPUT_SIZE as f32 - resize.pad_x;
        let y0 = (self.y_center - height * 0.5) * resize.scale * INPUT_SIZE as f32 - resize.pad_y;
        let x1 = (self.x_center + width * 0.5) * resize.scale * INPUT_SIZE as f32 - resize.pad_x;
        let y1 = (self.y_center + height * 0.5) * resize.scale * INPUT_SIZE as f32 - resize.pad_y;
        rect_from_f32(x0, y0, x1, y1, frame_size)
    }

    fn to_oriented_box(
        self,
        resize: ResizeMapping,
        frame_size: Size,
        fingertip_scale: f32,
    ) -> Option<OrientedBoundingBox> {
        let center = self.center_to_frame(resize);
        let wrist = self.keypoint_to_frame(WRIST_KEYPOINT, resize);
        let middle_mcp = self.keypoint_to_frame(MIDDLE_MCP_KEYPOINT, resize);

        // MediaPipe Hand Landmark model expects fingertips at the top (y=0).
        // The orientation axis should point from middle-MCP to wrist.
        let axis_y = wrist - middle_mcp;
        let palm_len = axis_y.length();
        if palm_len < 1.0 || !palm_len.is_finite() {
            return self
                .to_frame_rect(resize, frame_size, fingertip_scale)
                .map(rect_to_oriented_box);
        }

        let axis_y = axis_y / palm_len;
        // axis_x should be axis_y rotated 90 deg clockwise to maintain right-handedness.
        // If axis_y is backward (down), axis_x should be right.
        let axis_x = Vec2::new(axis_y.y, -axis_y.x);
        let raw_width = (self.width.abs() * resize.scale * INPUT_SIZE as f32).max(palm_len);
        let raw_height = (self.height.abs() * resize.scale * INPUT_SIZE as f32).max(palm_len);

        // Shift center along axis_y (middle-MCP to wrist).
        // Standard shift is -0.5, but since axis_y is flipped, we use +0.5.
        // wait, MEDIAPIPE_PALM_SHIFT_Y is -0.5.
        // If axis_y is middle-mcp to wrist, then +axis_y is towards wrist.
        // We want to shift towards fingertips (opposite of axis_y).
        // So we shift by -0.5 * height * axis_y.
        // The original code was center + (-shift * height * axis_y) where axis_y was wrist to mcp.
        // center + (-(-0.5) * height * forward) = center + 0.5 * height * forward.
        // Now center + (-(-0.5) * height * backward) = center + 0.5 * height * backward.
        // That's WRONG. We want center + 0.5 * height * forward.
        // If axis_y is backward, then forward = -axis_y.
        // center + 0.5 * height * (-axis_y) = center - 0.5 * height * axis_y.
        // So we want + shift * height * axis_y if shift is -0.5.
        let center = center + axis_y * (MEDIAPIPE_PALM_SHIFT_Y * raw_height);

        let side = raw_width.max(raw_height).max(1.0) * fingertip_scale.max(1.0);
        let half_side = side * 0.5;
        let corners = [
            center - axis_x * half_side - axis_y * half_side,
            center + axis_x * half_side - axis_y * half_side,
            center + axis_x * half_side + axis_y * half_side,
            center - axis_x * half_side + axis_y * half_side,
        ]
        .map(|corner| corner.to_array());
        Some(OrientedBoundingBox { corners })
    }

    fn center_to_frame(self, resize: ResizeMapping) -> Vec2 {
        Vec2::new(
            self.x_center * resize.scale * INPUT_SIZE as f32 - resize.pad_x,
            self.y_center * resize.scale * INPUT_SIZE as f32 - resize.pad_y,
        )
    }

    fn keypoint_to_frame(self, index: usize, resize: ResizeMapping) -> Vec2 {
        self.keypoint_to_frame_point(self.keypoints[index], resize)
    }

    fn keypoint_to_frame_point(self, point: [f32; 2], resize: ResizeMapping) -> Vec2 {
        let [x, y] = point;
        Vec2::new(
            x * resize.scale * INPUT_SIZE as f32 - resize.pad_x,
            y * resize.scale * INPUT_SIZE as f32 - resize.pad_y,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

fn preprocess_bgra(frame: Frame<'_>, output: &mut [f32]) -> Result<ResizeMapping> {
    let source_w = frame.meta.size.width as usize;
    let source_h = frame.meta.size.height as usize;
    anyhow::ensure!(source_w > 0 && source_h > 0, "empty RGB frame");
    let pixels = frame.view()?;
    let (resized_w, resized_h) = if source_h >= source_w {
        (INPUT_SIZE * source_w / source_h, INPUT_SIZE)
    } else {
        (INPUT_SIZE, INPUT_SIZE * source_h / source_w)
    };
    let pad_x = (INPUT_SIZE - resized_w) / 2;
    let pad_y = (INPUT_SIZE - resized_h) / 2;
    output.fill(0.0);

    let scale_x = source_w as f32 / resized_w as f32;
    let scale_y = source_h as f32 / resized_h as f32;

    for y in 0..resized_h {
        let src_yf = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = src_yf.floor() as isize;
        let y1 = y0 + 1;
        let dy = src_yf - y0 as f32;

        for x in 0..resized_w {
            let src_xf = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = src_xf.floor() as isize;
            let x1 = x0 + 1;
            let dx = src_xf - x0 as f32;

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

            let dst_x = pad_x + x;
            let dst_y = pad_y + y;
            let dst = dst_y * INPUT_SIZE + dst_x;
            output[dst] = r / 255.0;
            output[INPUT_SIZE * INPUT_SIZE + dst] = g / 255.0;
            output[2 * INPUT_SIZE * INPUT_SIZE + dst] = b / 255.0;
        }
    }
    Ok(ResizeMapping {
        scale: scale_x,
        pad_x: pad_x as f32 * scale_x,
        pad_y: pad_y as f32 * scale_y,
    })
}

fn best_detection(
    coords: &[f32],
    scores: &[f32],
    anchors: &[Anchor],
    min_score: f32,
) -> Option<Detection> {
    let n = anchors.len().min(scores.len()).min(coords.len() / 18);
    let mut best = None;
    for i in 0..n {
        let score = sigmoid(scores[i].clamp(-SCORE_CLIP, SCORE_CLIP));
        if score < min_score {
            continue;
        }
        let base = i * 18;
        let anchor = anchors[i];
        // MediaPipe palm detection uses reverse_output_order, so raw boxes and
        // keypoints are decoded as x/y pairs.
        let x_center = coords[base] / INPUT_SIZE as f32 * anchor.width + anchor.x_center;
        let y_center = coords[base + 1] / INPUT_SIZE as f32 * anchor.height + anchor.y_center;
        let width = coords[base + 2] / INPUT_SIZE as f32 * anchor.width;
        let height = coords[base + 3] / INPUT_SIZE as f32 * anchor.height;
        let mut keypoints = [[0.0; 2]; PALM_KEYPOINTS];
        for (keypoint, point) in keypoints.iter_mut().enumerate() {
            let offset = base + 4 + keypoint * 2;
            let x = coords[offset] / INPUT_SIZE as f32 * anchor.width + anchor.x_center;
            let y = coords[offset + 1] / INPUT_SIZE as f32 * anchor.height + anchor.y_center;
            *point = [x, y];
        }
        let detection = Detection {
            score,
            x_center,
            y_center,
            width,
            height,
            keypoints,
        };
        if best
            .map(|best: Detection| best.score < detection.score)
            .unwrap_or(true)
        {
            best = Some(detection);
        }
    }
    best
}

fn rect_to_oriented_box(rect: Rect) -> OrientedBoundingBox {
    let x0 = rect.x as f32;
    let y0 = rect.y as f32;
    let x1 = (rect.x + rect.size.width) as f32;
    let y1 = (rect.y + rect.size.height) as f32;
    OrientedBoundingBox {
        corners: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }
}

fn generate_palm_anchors() -> Vec<Anchor> {
    let strides = [8usize, 16, 32, 32, 32];
    let mut anchors = Vec::with_capacity(2944);
    for stride in strides {
        let feature_map = INPUT_SIZE.div_ceil(stride);
        for y in 0..feature_map {
            for x in 0..feature_map {
                let x_center = (x as f32 + 0.5) / feature_map as f32;
                let y_center = (y as f32 + 0.5) / feature_map as f32;
                for _ in 0..2 {
                    anchors.push(Anchor {
                        x_center,
                        y_center,
                        width: 1.0,
                        height: 1.0,
                    });
                }
            }
        }
    }
    anchors
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palm_anchor_count_matches_model() {
        assert_eq!(generate_palm_anchors().len(), 2944);
    }

    #[test]
    fn oriented_palm_box_uses_mediapipe_shifted_square_transform() {
        let detection = Detection {
            score: 1.0,
            x_center: 0.5,
            y_center: 0.5,
            width: 0.2,
            height: 0.2,
            keypoints: [
                [0.50, 0.50],
                [0.42, 0.60],
                [0.50, 0.60],
                [0.58, 0.60],
                [0.64, 0.58],
                [0.38, 0.56],
                [0.34, 0.54],
            ],
        };
        let box_ = detection
            .to_oriented_box(
                ResizeMapping {
                    scale: 1.0,
                    pad_x: 0.0,
                    pad_y: 0.0,
                },
                Size {
                    width: INPUT_SIZE as u32,
                    height: INPUT_SIZE as u32,
                },
                MEDIAPIPE_PALM_SCALE,
            )
            .unwrap();
        let wrist_y = 0.5 * INPUT_SIZE as f32;
        // c0, c1 are now the "forward" (fingertips) side.
        let forward = (box_.corners[0][1] + box_.corners[1][1]) * 0.5 - wrist_y;
        let back = wrist_y - (box_.corners[2][1] + box_.corners[3][1]) * 0.5;
        let top_width = (box_.corners[1][0] - box_.corners[0][0]).abs();
        let side_height = (box_.corners[3][1] - box_.corners[0][1]).abs();

        assert!(forward > back * 2.0);
        assert!((top_width - side_height).abs() < 1.0);
    }
}
