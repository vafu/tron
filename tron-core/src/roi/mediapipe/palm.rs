use std::path::Path;

use anyhow::{Context, Result};
use glam::Vec2;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions, Session};
use ort::value::TensorRef;
use tron_api::{
    Frame, NoContext, OrientedBoundingBox, PixelFormat, Processor, Rect, RoiResult, Size,
};

use super::{
    ModelInputLayout, ModelInputSpec, fallback_input_spec, letterbox_inverse_affine,
    model_input_spec, output_summary, preprocess_bgra, tensor_last_dim, tensor_num_elements,
};

const DEFAULT_INPUT_SIZE: usize = 256;
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
    input_spec: ModelInputSpec,
    outputs: PalmOutputSpec,
    run_options: RunOptions<HasSelectedOutputs>,
    input: Vec<f32>,
    anchors: Vec<Anchor>,
}

impl MediaPipeRoiProcessor {
    pub fn new(model_path: impl AsRef<Path>, config: MediaPipeRoiConfig) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("load MediaPipe hand detector {:?}", model_path.as_ref()))?;
        let input = session
            .inputs()
            .first()
            .context("MediaPipe hand detector model has no inputs")?;
        let input_spec = model_input_spec(input)
            .unwrap_or_else(|| fallback_input_spec(input, DEFAULT_INPUT_SIZE));
        let outputs = classify_outputs(&session)?;
        let run_options = outputs.run_options()?;
        let input_size = input_spec.size;
        let anchor_count = outputs.anchor_count;
        Ok(Self {
            session,
            config,
            input_spec,
            outputs,
            run_options,
            input: vec![0.0; 3 * input_size * input_size],
            anchors: generate_palm_anchors(input_size, anchor_count),
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
        let input_size = self.input_spec.size;
        let resize =
            preprocess_palm_bgra(input, input_size, self.input_spec.layout, &mut self.input)?;
        let tensor = TensorRef::from_array_view((self.input_spec.shape(), &*self.input))?;
        let outputs = self.session.run_with_options(
            vec![(self.input_spec.name.as_str(), tensor)],
            &self.run_options,
        )?;
        let (_, coords) = outputs[self.outputs.coords_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extract MediaPipe box_coords")?;
        let (_, scores) = outputs[self.outputs.scores_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extract MediaPipe box_scores")?;

        let detection = best_detection(
            coords,
            scores,
            &self.anchors,
            self.config.min_score,
            input_size,
        );

        Ok(detection.map(|detection| {
            let oriented_box = detection.to_oriented_box(
                resize,
                input.meta.size,
                input_size,
                self.config.box_scale,
            );

            let rect = oriented_box
                .and_then(|bbox| bbox.enclosing_rect(input.meta.size))
                .or_else(|| {
                    detection.to_frame_rect(
                        resize,
                        input.meta.size,
                        input_size,
                        self.config.box_scale,
                    )
                })
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
        input_size: usize,
        box_scale: f32,
    ) -> Option<Rect> {
        let input_size = input_size as f32;
        let width = (self.width * box_scale).max(1.0 / input_size);
        let height = (self.height * box_scale).max(1.0 / input_size);
        let x0 = (self.x_center - width * 0.5) * resize.scale * input_size - resize.pad_x;
        let y0 = (self.y_center - height * 0.5) * resize.scale * input_size - resize.pad_y;
        let x1 = (self.x_center + width * 0.5) * resize.scale * input_size - resize.pad_x;
        let y1 = (self.y_center + height * 0.5) * resize.scale * input_size - resize.pad_y;
        rect_from_f32(x0, y0, x1, y1, frame_size)
    }

    fn to_oriented_box(
        self,
        resize: ResizeMapping,
        frame_size: Size,
        input_size: usize,
        fingertip_scale: f32,
    ) -> Option<OrientedBoundingBox> {
        let center = self.center_to_frame(resize, input_size);
        let wrist = self.keypoint_to_frame(WRIST_KEYPOINT, resize, input_size);
        let middle_mcp = self.keypoint_to_frame(MIDDLE_MCP_KEYPOINT, resize, input_size);

        // MediaPipe Hand Landmark model expects fingertips at the top (y=0).
        // The orientation axis points from middle-MCP knuckle to the wrist.
        let axis_y = wrist - middle_mcp;
        let palm_len = axis_y.length();
        if palm_len < 1.0 || !palm_len.is_finite() {
            return self
                .to_frame_rect(resize, frame_size, input_size, fingertip_scale)
                .map(rect_to_oriented_box);
        }

        let axis_y = axis_y / palm_len;
        // axis_x is axis_y rotated 90 deg clockwise.
        let axis_x = Vec2::new(axis_y.y, -axis_y.x);

        // ROI size is derived from either the predicted box or the actual palm length.
        let input_size = input_size as f32;
        let raw_width = (self.width.abs() * resize.scale * input_size).max(palm_len);
        let raw_height = (self.height.abs() * resize.scale * input_size).max(palm_len);

        // Shift center along axis_y (middle-MCP to wrist).
        // Since axis_y points towards the wrist, we move by negative axis_y (towards fingertips).
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

    fn center_to_frame(self, resize: ResizeMapping, input_size: usize) -> Vec2 {
        let input_size = input_size as f32;
        Vec2::new(
            self.x_center * resize.scale * input_size - resize.pad_x,
            self.y_center * resize.scale * input_size - resize.pad_y,
        )
    }

    fn keypoint_to_frame(self, index: usize, resize: ResizeMapping, input_size: usize) -> Vec2 {
        self.keypoint_to_frame_point(self.keypoints[index], resize, input_size)
    }

    fn keypoint_to_frame_point(
        self,
        point: [f32; 2],
        resize: ResizeMapping,
        input_size: usize,
    ) -> Vec2 {
        let [x, y] = point;
        let input_size = input_size as f32;
        Vec2::new(
            x * resize.scale * input_size - resize.pad_x,
            y * resize.scale * input_size - resize.pad_y,
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

#[derive(Clone, Debug)]
struct PalmOutputSpec {
    coords_name: String,
    scores_name: String,
    anchor_count: usize,
}

impl PalmOutputSpec {
    fn run_options(&self) -> Result<RunOptions<HasSelectedOutputs>> {
        Ok(RunOptions::new()?.with_outputs(
            OutputSelector::no_default()
                .with(self.coords_name.clone())
                .with(self.scores_name.clone()),
        ))
    }
}

fn preprocess_palm_bgra(
    frame: Frame<'_>,
    input_size: usize,
    input_layout: ModelInputLayout,
    output: &mut [f32],
) -> Result<ResizeMapping> {
    let source_w = frame.meta.size.width as usize;
    let source_h = frame.meta.size.height as usize;
    anyhow::ensure!(source_w > 0 && source_h > 0, "empty RGB frame");
    let stride = frame.buffer.stride();
    anyhow::ensure!(stride == source_w * 4, "frame must be tightly packed BGRA8");

    let (resized_w, resized_h) = if source_h >= source_w {
        (input_size * source_w / source_h, input_size)
    } else {
        (input_size, input_size * source_h / source_w)
    };
    let pad_x = (input_size - resized_w) / 2;
    let pad_y = (input_size - resized_h) / 2;

    let scale_x = source_w as f32 / resized_w as f32;
    let scale_y = source_h as f32 / resized_h as f32;

    let transform = letterbox_inverse_affine(
        frame.meta.size,
        resized_w,
        resized_h,
        pad_x,
        pad_y,
        frame.buffer.is_horizontally_mirrored(),
        frame.buffer.is_vertically_mirrored(),
    )?;
    preprocess_bgra(frame, input_size, input_layout, &transform, "palm", output)?;

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
    input_size: usize,
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

        let input_size = input_size as f32;
        let x_center = coords[base] / input_size * anchor.width + anchor.x_center;
        let y_center = coords[base + 1] / input_size * anchor.height + anchor.y_center;
        let width = coords[base + 2] / input_size * anchor.width;
        let height = coords[base + 3] / input_size * anchor.height;

        let mut keypoints = [[0.0; 2]; PALM_KEYPOINTS];
        for (keypoint, point) in keypoints.iter_mut().enumerate() {
            let offset = base + 4 + keypoint * 2;
            let x = coords[offset] / input_size * anchor.width + anchor.x_center;
            let y = coords[offset + 1] / input_size * anchor.height + anchor.y_center;
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

fn classify_outputs(session: &Session) -> Result<PalmOutputSpec> {
    let outputs = session.outputs();
    let mut coords_output = None;
    let mut scores_output = None;

    for (i, output) in outputs.iter().enumerate() {
        let name = output.name().to_ascii_lowercase();
        let elements = tensor_num_elements(output.dtype());
        let last_dim = tensor_last_dim(output.dtype());
        if coords_output.is_none()
            && (name.contains("box_coords")
                || name.contains("regressor")
                || last_dim == Some(18)
                || (elements > 0 && last_dim != Some(1) && elements % 18 == 0))
        {
            coords_output = Some((i, elements / 18));
        }
        if scores_output.is_none()
            && (name.contains("box_scores")
                || name.contains("classificator")
                || name.contains("score")
                || last_dim == Some(1))
            && elements > 0
        {
            scores_output = Some(i);
        }
    }

    let Some((coords_output, anchor_count)) = coords_output else {
        let outputs = output_summary(outputs);
        anyhow::bail!(
            "MediaPipe hand detector model has no box coordinate output; outputs: {outputs}"
        );
    };
    let scores_output = scores_output.unwrap_or_else(|| {
        outputs
            .iter()
            .enumerate()
            .find_map(|(i, output)| {
                (tensor_num_elements(output.dtype()) == anchor_count).then_some(i)
            })
            .unwrap_or(
                coords_output
                    .saturating_add(1)
                    .min(outputs.len().saturating_sub(1)),
            )
    });
    Ok(PalmOutputSpec {
        coords_name: outputs[coords_output].name().to_owned(),
        scores_name: outputs[scores_output].name().to_owned(),
        anchor_count,
    })
}

fn generate_palm_anchors(input_size: usize, anchor_count: usize) -> Vec<Anchor> {
    let candidates: &[&[usize]] = &[&[8, 16, 32, 32, 32], &[8, 16, 16, 16]];
    for strides in candidates {
        let anchors = generate_palm_anchors_for_strides(input_size, strides);
        if anchors.len() == anchor_count {
            return anchors;
        }
    }
    let anchors = generate_palm_anchors_for_strides(input_size, candidates[0]);
    tracing::warn!(
        "generated {} palm anchors for {}x{} detector, but model has {} anchors",
        anchors.len(),
        input_size,
        input_size,
        anchor_count
    );
    anchors
}

fn generate_palm_anchors_for_strides(input_size: usize, strides: &[usize]) -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(2944);
    for &stride in strides {
        let feature_map = input_size.div_ceil(stride);
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
        assert_eq!(generate_palm_anchors(256, 2944).len(), 2944);
        assert_eq!(generate_palm_anchors(192, 2016).len(), 2016);
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
                    width: DEFAULT_INPUT_SIZE as u32,
                    height: DEFAULT_INPUT_SIZE as u32,
                },
                DEFAULT_INPUT_SIZE,
                MEDIAPIPE_PALM_SCALE,
            )
            .unwrap();
        let wrist_y = 0.5 * DEFAULT_INPUT_SIZE as f32;
        // c0, c1 are now the "forward" (fingertips) side.
        let forward = (box_.corners[0][1] + box_.corners[1][1]) * 0.5 - wrist_y;
        let back = wrist_y - (box_.corners[2][1] + box_.corners[3][1]) * 0.5;
        let top_width = (box_.corners[1][0] - box_.corners[0][0]).abs();
        let side_height = (box_.corners[3][1] - box_.corners[0][1]).abs();

        assert!(forward > back * 2.0);
        assert!((top_width - side_height).abs() < 1.0);
    }
}
