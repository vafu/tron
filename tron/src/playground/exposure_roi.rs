use anyhow::Result;
use tron_api::{Frame, PixelFormat, Rect, RoiResult, Size};

#[derive(Clone, Copy, Debug)]
pub struct ClippedExposureRoiConfig {
    pub threshold: u8,
    pub min_pixels: u32,
    pub padding: u32,
}

impl Default for ClippedExposureRoiConfig {
    fn default() -> Self {
        Self {
            threshold: 250,
            min_pixels: 16,
            padding: 0,
        }
    }
}

pub struct ClippedExposureRoiDetector {
    config: ClippedExposureRoiConfig,
    heights: Vec<u32>,
    stack: Vec<usize>,
}

impl ClippedExposureRoiDetector {
    pub fn new(config: ClippedExposureRoiConfig) -> Self {
        Self {
            config,
            heights: Vec::new(),
            stack: Vec::new(),
        }
    }

    pub fn detect(
        &mut self,
        frame: Frame<'_>,
        candidate: Option<RoiResult>,
    ) -> Result<Option<RoiResult>> {
        anyhow::ensure!(
            frame.format == PixelFormat::Gray8,
            "clipped exposure ROI requires Gray8 input, got {:?}",
            frame.format
        );
        let bounds = frame.meta.size;
        let search = candidate
            .map(|roi| clamp_rect(roi.rect, bounds))
            .unwrap_or(Rect {
                x: 0,
                y: 0,
                size: bounds,
            });

        let mut clipped = 0;
        let mut best: Option<InteriorRect> = None;
        let width = search.size.width as usize;
        self.heights.resize(width, 0);
        self.heights.fill(0);
        self.stack.clear();
        let y_end = search.y.saturating_add(search.size.height);
        let x_end = search.x.saturating_add(search.size.width);
        let pixels = frame.view()?;
        for y in search.y..y_end {
            for (xi, x) in (search.x..x_end).enumerate() {
                if pixels[[y as usize, x as usize, 0]] >= self.config.threshold {
                    clipped += 1;
                    self.heights[xi] = self.heights[xi].saturating_add(1);
                } else {
                    self.heights[xi] = 0;
                }
            }
            if let Some(row_best) = largest_histogram_rect(&self.heights, y, &mut self.stack) {
                if best
                    .map(|current| current.area() < row_best.area())
                    .unwrap_or(true)
                {
                    best = Some(row_best);
                }
            }
        }

        if clipped < self.config.min_pixels {
            return Ok(None);
        }

        let Some(best) = best else {
            return Ok(None);
        };

        Ok(Some(RoiResult {
            rect: padded_rect(
                Rect {
                    x: search.x.saturating_add(best.x),
                    y: best.y,
                    size: Size {
                        width: best.width,
                        height: best.height,
                    },
                },
                self.config.padding,
                bounds,
            ),
            oriented_box: None,
        }))
    }
}

#[derive(Clone, Copy, Debug)]
struct InteriorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl InteriorRect {
    fn area(self) -> u32 {
        self.width.saturating_mul(self.height)
    }
}

fn largest_histogram_rect(
    heights: &[u32],
    bottom_y: u32,
    stack: &mut Vec<usize>,
) -> Option<InteriorRect> {
    let mut best = None;
    stack.clear();
    for i in 0..=heights.len() {
        let current_height = heights.get(i).copied().unwrap_or(0);
        while let Some(&top) = stack.last()
            && current_height < heights[top]
        {
            stack.pop();
            let height = heights[top];
            let left = stack.last().map(|left| left + 1).unwrap_or(0);
            let width = i.saturating_sub(left) as u32;
            let candidate = InteriorRect {
                x: left as u32,
                y: bottom_y.saturating_add(1).saturating_sub(height),
                width,
                height,
            };
            if best
                .map(|current: InteriorRect| current.area() < candidate.area())
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
        stack.push(i);
    }
    best
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

fn padded_rect(rect: Rect, padding: u32, bounds: Size) -> Rect {
    let x0 = rect.x.saturating_sub(padding);
    let y0 = rect.y.saturating_sub(padding);
    let x1 = rect
        .x
        .saturating_add(rect.size.width)
        .saturating_add(padding)
        .min(bounds.width);
    let y1 = rect
        .y
        .saturating_add(rect.size.height)
        .saturating_add(padding)
        .min(bounds.height);
    Rect {
        x: x0,
        y: y0,
        size: Size {
            width: x1.saturating_sub(x0),
            height: y1.saturating_sub(y0),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use tron_api::{FrameMeta, FrameTimestamp, SensorKind, TimestampSource};

    #[test]
    fn detects_clipped_pixels_inside_candidate_roi() {
        let data = [
            0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, //
            0, 0, 251, 252, 0, 0, //
            0, 0, 253, 254, 0, 0, //
            0, 0, 0, 0, 0, 0, //
        ];
        let frame = frame(&data, 6, 5);
        let mut detector = ClippedExposureRoiDetector::new(ClippedExposureRoiConfig {
            threshold: 250,
            min_pixels: 4,
            padding: 0,
        });

        let roi = detector
            .detect(
                frame,
                Some(RoiResult {
                    rect: Rect {
                        x: 1,
                        y: 1,
                        size: Size {
                            width: 4,
                            height: 4,
                        },
                    },
                    oriented_box: None,
                }),
            )
            .unwrap()
            .unwrap();

        assert_eq!(
            roi.rect,
            Rect {
                x: 2,
                y: 2,
                size: Size {
                    width: 2,
                    height: 2
                }
            }
        );
    }

    #[test]
    fn chooses_interior_rect_instead_of_clipped_bounding_box() {
        let data = [
            0, 0, 0, 0, 0, 0, //
            0, 251, 251, 251, 0, 0, //
            0, 251, 251, 251, 0, 0, //
            0, 251, 251, 251, 0, 253, //
            0, 0, 0, 0, 0, 253, //
        ];
        let frame = frame(&data, 6, 5);
        let mut detector = ClippedExposureRoiDetector::new(ClippedExposureRoiConfig {
            threshold: 250,
            min_pixels: 4,
            padding: 0,
        });

        let roi = detector.detect(frame, None).unwrap().unwrap();

        assert_eq!(
            roi.rect,
            Rect {
                x: 1,
                y: 1,
                size: Size {
                    width: 3,
                    height: 3
                }
            }
        );
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
