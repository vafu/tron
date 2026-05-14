use anyhow::Result;
use tron_api::{NoContext, Processor, Rect, RoiResult, Size};

#[derive(Clone, Copy, Debug)]
pub struct CameraRoiFollowConfig {
    pub min_edge: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraRoiFollowInput {
    pub roi: Option<RoiResult>,
    pub allowed_bounds: Option<Rect>,
    pub source_size: Size,
    pub target_size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraRoiFollowProcessor {
    config: CameraRoiFollowConfig,
}

impl CameraRoiFollowProcessor {
    pub fn new(config: CameraRoiFollowConfig) -> Self {
        Self { config }
    }
}

impl Processor<CameraRoiFollowInput, NoContext> for CameraRoiFollowProcessor {
    type Output = Option<Rect>;

    fn process(
        &mut self,
        input: CameraRoiFollowInput,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(roi) = input.roi else {
            return Ok(None);
        };
        let bounds = input
            .allowed_bounds
            .map(|bounds| clamp_rect(bounds, input.target_size))
            .unwrap_or(Rect {
                x: 0,
                y: 0,
                size: input.target_size,
            });
        let rect = map_rect(roi.rect, input.source_size, input.target_size);
        Ok(Some(expand_to_min_edge(rect, self.config.min_edge, bounds)))
    }
}

fn map_rect(rect: Rect, source_size: Size, target_size: Size) -> Rect {
    let sx = target_size.width as f32 / source_size.width.max(1) as f32;
    let sy = target_size.height as f32 / source_size.height.max(1) as f32;
    let cx = (rect.x as f32 + rect.size.width as f32 * 0.5) * sx;
    let cy = (rect.y as f32 + rect.size.height as f32 * 0.5) * sy;
    let width = (rect.size.width as f32 * sx).round().max(1.0) as u32;
    let height = (rect.size.height as f32 * sy).round().max(1.0) as u32;
    let x = (cx - width as f32 * 0.5).round().max(0.0) as u32;
    let y = (cy - height as f32 * 0.5).round().max(0.0) as u32;
    Rect {
        x,
        y,
        size: Size { width, height },
    }
}

fn expand_to_min_edge(rect: Rect, min_edge: u32, bounds: Rect) -> Rect {
    let width = rect.size.width.max(min_edge).min(bounds.size.width).max(1);
    let height = rect
        .size
        .height
        .max(min_edge)
        .min(bounds.size.height)
        .max(1);
    let bx1 = bounds.x.saturating_add(bounds.size.width);
    let by1 = bounds.y.saturating_add(bounds.size.height);
    let cx = rect
        .x
        .saturating_add(rect.size.width / 2)
        .clamp(bounds.x, bx1);
    let cy = rect
        .y
        .saturating_add(rect.size.height / 2)
        .clamp(bounds.y, by1);
    let x = cx
        .saturating_sub(width / 2)
        .clamp(bounds.x, bx1.saturating_sub(width));
    let y = cy
        .saturating_sub(height / 2)
        .clamp(bounds.y, by1.saturating_sub(height));
    Rect {
        x,
        y,
        size: Size { width, height },
    }
}

fn clamp_rect(rect: Rect, frame_size: Size) -> Rect {
    let x = rect.x.min(frame_size.width);
    let y = rect.y.min(frame_size.height);
    let width = rect.size.width.min(frame_size.width.saturating_sub(x));
    let height = rect.size.height.min(frame_size.height.saturating_sub(y));
    Rect {
        x,
        y,
        size: Size { width, height },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roi(rect: Rect) -> RoiResult {
        RoiResult {
            rect,
            oriented_box: None,
        }
    }

    #[test]
    fn expands_roi_inside_allowed_bounds() {
        let rect = expand_to_min_edge(
            Rect {
                x: 95,
                y: 95,
                size: Size {
                    width: 5,
                    height: 5,
                },
            },
            40,
            Rect {
                x: 80,
                y: 80,
                size: Size {
                    width: 50,
                    height: 50,
                },
            },
        );

        assert_eq!(
            rect,
            Rect {
                x: 80,
                y: 80,
                size: Size {
                    width: 40,
                    height: 40
                }
            }
        );
    }

    #[test]
    fn maps_roi_center_between_frame_sizes() {
        let mut processor = CameraRoiFollowProcessor::new(CameraRoiFollowConfig { min_edge: 1 });
        let rect = processor
            .process(
                CameraRoiFollowInput {
                    roi: Some(roi(Rect {
                        x: 100,
                        y: 50,
                        size: Size {
                            width: 20,
                            height: 10,
                        },
                    })),
                    allowed_bounds: None,
                    source_size: Size {
                        width: 200,
                        height: 100,
                    },
                    target_size: Size {
                        width: 400,
                        height: 200,
                    },
                },
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(rect.x + rect.size.width / 2, 220);
        assert_eq!(rect.y + rect.size.height / 2, 110);
    }
}
