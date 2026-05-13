use std::path::Path;

use anyhow::{Context, Result};
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
const WRIST_BACK_PADDING: f32 = 0.06;
const WRIST_BACK_MAX_EXTENSION: f32 = 0.12;
const SIDE_PADDING: f32 = 0.16;
const MIN_HALF_WIDTH: f32 = 0.45;

#[derive(Clone, Debug)]
pub struct MediaPipeRoiConfig {
    pub min_score: f32,
    pub box_scale: f32,
}

impl Default for MediaPipeRoiConfig {
    fn default() -> Self {
        Self {
            min_score: 0.75,
            box_scale: 1.0,
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
        let wrist = self.keypoint_to_frame(WRIST_KEYPOINT, resize);
        let middle_mcp = self.keypoint_to_frame(MIDDLE_MCP_KEYPOINT, resize);
        let axis_y = [middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]];
        let palm_len = hypot(axis_y[0], axis_y[1]);
        if palm_len < 1.0 || !palm_len.is_finite() {
            return self
                .to_frame_rect(resize, frame_size, fingertip_scale)
                .map(rect_to_oriented_box);
        }

        let axis_y = [axis_y[0] / palm_len, axis_y[1] / palm_len];
        let axis_x = [-axis_y[1], axis_y[0]];
        let mut min_x = 0.0f32;
        let mut max_x = 0.0f32;
        let mut min_y = 0.0f32;
        let mut max_y = palm_len;
        for point in self.keypoints {
            let point = self.keypoint_to_frame_point(point, resize);
            let delta = [point[0] - wrist[0], point[1] - wrist[1]];
            let x = dot(delta, axis_x);
            let y = dot(delta, axis_y);
            if x.is_finite() && y.is_finite() {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        let half_width =
            min_x.abs().max(max_x.abs()).max(palm_len * MIN_HALF_WIDTH) + palm_len * SIDE_PADDING;
        let back = (-min_y).max(0.0).min(palm_len * WRIST_BACK_MAX_EXTENSION)
            + palm_len * WRIST_BACK_PADDING;
        let forward = max_y.max(palm_len) * fingertip_scale.max(1.0);
        let corners = [
            add(add(wrist, mul(axis_x, -half_width)), mul(axis_y, -back)),
            add(add(wrist, mul(axis_x, half_width)), mul(axis_y, -back)),
            add(add(wrist, mul(axis_x, half_width)), mul(axis_y, forward)),
            add(add(wrist, mul(axis_x, -half_width)), mul(axis_y, forward)),
        ];
        Some(OrientedBoundingBox { corners })
    }

    fn keypoint_to_frame(self, index: usize, resize: ResizeMapping) -> [f32; 2] {
        self.keypoint_to_frame_point(self.keypoints[index], resize)
    }

    fn keypoint_to_frame_point(self, point: [f32; 2], resize: ResizeMapping) -> [f32; 2] {
        let [x, y] = point;
        [
            x * resize.scale * INPUT_SIZE as f32 - resize.pad_x,
            y * resize.scale * INPUT_SIZE as f32 - resize.pad_y,
        ]
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
    for y in 0..resized_h {
        let src_y = y * source_h / resized_h;
        for x in 0..resized_w {
            let src_x = x * source_w / resized_w;
            let dst_x = pad_x + x;
            let dst_y = pad_y + y;
            let dst = dst_y * INPUT_SIZE + dst_x;
            output[dst] = pixels[[src_y, src_x, 2]] as f32 / 255.0;
            output[INPUT_SIZE * INPUT_SIZE + dst] = pixels[[src_y, src_x, 1]] as f32 / 255.0;
            output[2 * INPUT_SIZE * INPUT_SIZE + dst] = pixels[[src_y, src_x, 0]] as f32 / 255.0;
        }
    }
    Ok(ResizeMapping {
        scale: source_w as f32 / resized_w as f32,
        pad_x: pad_x as f32 * source_w as f32 / resized_w as f32,
        pad_y: pad_y as f32 * source_h as f32 / resized_h as f32,
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

fn hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn mul(v: [f32; 2], scale: f32) -> [f32; 2] {
    [v[0] * scale, v[1] * scale]
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
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
    fn oriented_palm_box_extends_toward_fingertips() {
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
                2.5,
            )
            .unwrap();
        let wrist_y = 0.5 * INPUT_SIZE as f32;
        let back = wrist_y - (box_.corners[0][1] + box_.corners[1][1]) * 0.5;
        let forward = (box_.corners[2][1] + box_.corners[3][1]) * 0.5 - wrist_y;

        assert!(forward > back * 10.0);
    }
}
