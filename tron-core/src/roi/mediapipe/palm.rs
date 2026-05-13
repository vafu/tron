use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use tron_api::{Frame, NoContext, PixelFormat, Processor, Rect, RoiResult, Size};

const INPUT_SIZE: usize = 256;
const SCORE_CLIP: f32 = 100.0;

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
        Ok(detection.map(|detection| RoiResult {
            rect: detection
                .to_frame_rect(resize, input.meta.size, self.config.box_scale)
                .unwrap_or(Rect {
                    x: 0,
                    y: 0,
                    size: input.meta.size,
                }),
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
    anyhow::ensure!(
        frame.buffer.stride >= source_w * 4,
        "BGRA frame stride {} is smaller than width {}",
        frame.buffer.stride,
        source_w * 4
    );
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
            let src = src_y * frame.buffer.stride + src_x * 4;
            let dst_x = pad_x + x;
            let dst_y = pad_y + y;
            let dst = dst_y * INPUT_SIZE + dst_x;
            output[dst] = frame.buffer.data[src + 2] as f32 / 255.0;
            output[INPUT_SIZE * INPUT_SIZE + dst] = frame.buffer.data[src + 1] as f32 / 255.0;
            output[2 * INPUT_SIZE * INPUT_SIZE + dst] = frame.buffer.data[src] as f32 / 255.0;
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
        let y_center = coords[base] / INPUT_SIZE as f32 * anchor.height + anchor.y_center;
        let x_center = coords[base + 1] / INPUT_SIZE as f32 * anchor.width + anchor.x_center;
        let height = coords[base + 2] / INPUT_SIZE as f32 * anchor.height;
        let width = coords[base + 3] / INPUT_SIZE as f32 * anchor.width;
        let detection = Detection {
            score,
            x_center,
            y_center,
            width,
            height,
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
}
