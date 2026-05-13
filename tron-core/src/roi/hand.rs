use anyhow::Result;
use tron_api::{Frame, NoContext, PixelFormat, Processor, Rect, RoiCandidate, RoiResult, Size};

pub struct HandRoiInput<'a> {
    pub candidates: &'a [RoiCandidate],
    pub motion: Option<Frame<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct HandRoiTrackerConfig {
    pub min_motion_pixels: u32,
    pub max_lost_frames: u32,
}

impl Default for HandRoiTrackerConfig {
    fn default() -> Self {
        Self {
            min_motion_pixels: 8,
            max_lost_frames: 10,
        }
    }
}

pub struct HandRoiTracker {
    config: HandRoiTrackerConfig,
    previous: Option<Rect>,
    lost_frames: u32,
}

impl HandRoiTracker {
    pub fn new(config: HandRoiTrackerConfig) -> Self {
        Self {
            config,
            previous: None,
            lost_frames: 0,
        }
    }

    fn score(&self, candidate: RoiCandidate, motion: Option<Frame<'_>>) -> Result<f32> {
        let motion_pixels = motion_overlap(candidate.rect, motion)?;
        let area_score = (candidate.area as f32).sqrt() * 0.05;
        let previous_score = self
            .previous
            .map(|previous| {
                let iou = rect_iou(previous, candidate.rect);
                let distance = center_distance(previous, candidate.rect);
                iou * 250.0 + (100.0 / (1.0 + distance / 24.0))
            })
            .unwrap_or(0.0);
        Ok(motion_pixels as f32 * 20.0 + previous_score + area_score)
    }

    fn mark_lost(&mut self) {
        self.lost_frames = self.lost_frames.saturating_add(1);
        if self.lost_frames > self.config.max_lost_frames {
            self.previous = None;
        }
    }
}

impl Processor<HandRoiInput<'_>> for HandRoiTracker {
    type Output = Option<RoiResult>;

    fn process(&mut self, input: HandRoiInput<'_>, _context: NoContext) -> Result<Self::Output> {
        if input.candidates.is_empty() {
            self.mark_lost();
            return Ok(self.previous.map(|rect| RoiResult { rect }));
        }

        let mut best = None;
        for candidate in input.candidates {
            let score = self.score(*candidate, input.motion)?;
            if best
                .map(|(_, best_score): (RoiCandidate, f32)| best_score < score)
                .unwrap_or(true)
            {
                best = Some((*candidate, score));
            }
        }

        let Some((candidate, score)) = best else {
            self.mark_lost();
            return Ok(self.previous.map(|rect| RoiResult { rect }));
        };
        if self.previous.is_none() && score < self.config.min_motion_pixels as f32 {
            return Ok(None);
        }

        self.previous = Some(candidate.rect);
        self.lost_frames = 0;
        Ok(Some(RoiResult {
            rect: candidate.rect,
        }))
    }
}

fn motion_overlap(rect: Rect, motion: Option<Frame<'_>>) -> Result<u32> {
    let Some(motion) = motion else {
        return Ok(0);
    };
    if motion.format != PixelFormat::Gray8 {
        return Ok(0);
    }
    let rect = clamp_rect(rect, motion.meta.size);
    let mut count = 0;
    let y_end = rect.y.saturating_add(rect.size.height);
    let x_end = rect.x.saturating_add(rect.size.width);
    for y in rect.y..y_end {
        let row = motion.row(y)?;
        for x in rect.x..x_end {
            if row[x as usize] > 0 {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn rect_iou(a: Rect, b: Rect) -> f32 {
    let ax1 = a.x.saturating_add(a.size.width);
    let ay1 = a.y.saturating_add(a.size.height);
    let bx1 = b.x.saturating_add(b.size.width);
    let by1 = b.y.saturating_add(b.size.height);
    let ix0 = a.x.max(b.x);
    let iy0 = a.y.max(b.y);
    let ix1 = ax1.min(bx1);
    let iy1 = ay1.min(by1);
    let intersection = ix1
        .saturating_sub(ix0)
        .saturating_mul(iy1.saturating_sub(iy0));
    let a_area = a.size.width.saturating_mul(a.size.height);
    let b_area = b.size.width.saturating_mul(b.size.height);
    let union = a_area.saturating_add(b_area).saturating_sub(intersection);
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn center_distance(a: Rect, b: Rect) -> f32 {
    let ax = a.x as f32 + a.size.width as f32 * 0.5;
    let ay = a.y as f32 + a.size.height as f32 * 0.5;
    let bx = b.x as f32 + b.size.width as f32 * 0.5;
    let by = b.y as f32 + b.size.height as f32 * 0.5;
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

fn clamp_rect(rect: Rect, bounds: Size) -> Rect {
    let x = rect.x.min(bounds.width);
    let y = rect.y.min(bounds.height);
    let width = rect.size.width.min(bounds.width.saturating_sub(x));
    let height = rect.size.height.min(bounds.height.saturating_sub(y));
    Rect {
        x,
        y,
        size: Size { width, height },
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use tron_api::{FrameMeta, FrameTimestamp, SensorKind, TimestampSource};

    #[test]
    fn initial_lock_prefers_candidate_with_motion() {
        let candidates = [candidate(0, 0, 20, 20, 400), candidate(60, 0, 20, 20, 400)];
        let data = motion_frame_data(
            100,
            40,
            Rect {
                x: 60,
                y: 0,
                size: Size {
                    width: 20,
                    height: 20,
                },
            },
        );
        let mut tracker = HandRoiTracker::new(HandRoiTrackerConfig::default());

        let roi = tracker
            .process(
                HandRoiInput {
                    candidates: &candidates,
                    motion: Some(frame(&data, 100, 40)),
                },
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(roi.rect, candidates[1].rect);
    }

    #[test]
    fn initial_lock_rejects_static_bright_candidates() {
        let candidates = [candidate(0, 0, 20, 20, 400)];
        let data = vec![0; 100 * 40];
        let mut tracker = HandRoiTracker::new(HandRoiTrackerConfig::default());

        let roi = tracker
            .process(
                HandRoiInput {
                    candidates: &candidates,
                    motion: Some(frame(&data, 100, 40)),
                },
                NoContext,
            )
            .unwrap();

        assert_eq!(roi, None);
    }

    fn candidate(x: u32, y: u32, width: u32, height: u32, area: u32) -> RoiCandidate {
        RoiCandidate {
            rect: Rect {
                x,
                y,
                size: Size { width, height },
            },
            area,
        }
    }

    fn motion_frame_data(width: usize, height: usize, rect: Rect) -> Vec<u8> {
        let mut data = vec![0; width * height];
        for y in rect.y..rect.y + rect.size.height {
            for x in rect.x..rect.x + rect.size.width {
                data[y as usize * width + x as usize] = 255;
            }
        }
        data
    }

    fn frame(data: &[u8], width: u32, height: u32) -> Frame<'_> {
        Frame::new(
            FrameMeta {
                id: 1,
                sensor: SensorKind::Ir,
                size: Size { width, height },
                timestamp: FrameTimestamp {
                    camera_monotonic_us: None,
                    source: TimestampSource::Unknown,
                    received_at: Instant::now(),
                },
                sequence: None,
            },
            PixelFormat::Gray8,
            width as usize,
            data,
        )
        .unwrap()
    }
}
