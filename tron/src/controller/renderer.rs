use anyhow::Result;
use tron_api::{PixelFormat, Point2d, PointerEvent, Sink, Size};
use tron_core::render::hand_landmarks_overlay::{
    HandLandmarksOverlayRenderer, HandLandmarksOverlayView,
};
use tron_core::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};
use tron_core::render::roi_overlay::{RoiOverlayRenderer, RoiOverlayView};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};

use crate::pipeline::ControllerFrame;

pub struct Renderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    palm_roi_overlay: RoiOverlayRenderer,
    roi_overlay: RoiOverlayRenderer,
    landmarks_overlay: HandLandmarksOverlayRenderer,
    pointer_overlay: LineOverlayRenderer,
    pointer_position: Option<Point2d>,
    pointer_down: bool,
    pointer_vertices: [LineVertex; 4],
}

impl Renderer {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-controller-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            rgb: WgpuFrameRenderer::new(surface.device(), format),
            palm_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            landmarks_overlay: HandLandmarksOverlayRenderer::new(surface.device(), format),
            pointer_overlay: LineOverlayRenderer::new(
                surface.device(),
                format,
                "tron-controller-pointer-overlay",
            ),
            pointer_position: None,
            pointer_down: false,
            pointer_vertices: [LineVertex {
                position: [0.0, 0.0],
                color: [0.0, 0.0, 0.0, 0.0],
            }; 4],
            surface,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a ControllerFrame<'a>> for Renderer {
    async fn consume(&mut self, view: &'a ControllerFrame<'a>) -> Result<()> {
        if view.rgb.format != PixelFormat::Bgra8 {
            anyhow::bail!("controller RGB feed expects BGRA8 frames");
        }

        self.surface.render(
            "tron-controller-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                let rgb = view.rgb;
                pollster::block_on(self.rgb.consume(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame: rgb,
                    rect: NdcRect::FULL,
                    target_size: surface.size,
                }))?;
                if let Some(palm_roi) = view.palm_roi {
                    pollster::block_on(self.palm_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: palm_roi.rect,
                        oriented_roi: palm_roi.oriented_box,
                        color: [1.0, 0.62, 0.08, 1.0],
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(rgb_roi) = view.rgb_roi {
                    pollster::block_on(self.roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: rgb_roi.rect,
                        oriented_roi: rgb_roi.oriented_box,
                        color: [0.2, 1.0, 0.2, 1.0],
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(landmarks) = view.landmarks.as_ref() {
                    pollster::block_on(self.landmarks_overlay.consume(HandLandmarksOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        landmarks,
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(position) = self.pointer_position {
                    self.pointer_vertices = pointer_vertices(position, self.pointer_down);
                    pollster::block_on(self.pointer_overlay.consume(LineOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        vertices: &self.pointer_vertices,
                    }))?;
                }
                Ok(())
            },
        )
    }
}

#[async_trait::async_trait(?Send)]
impl Sink<PointerEvent> for Renderer {
    async fn consume(&mut self, event: PointerEvent) -> Result<()> {
        match event {
            PointerEvent::Move {
                position, delta, ..
            } => {
                self.pointer_position = match (position, self.pointer_position) {
                    (Some(position), _) => Some(position),
                    (None, Some(position)) => {
                        Some((position + delta).clamp(Point2d::ZERO, Point2d::ONE))
                    }
                    (None, None) => None,
                };
            }
            PointerEvent::Down { .. } => self.pointer_down = true,
            PointerEvent::Up { .. } => self.pointer_down = false,
            PointerEvent::Click { .. } => {}
            PointerEvent::Cancel { .. } => {
                self.pointer_position = None;
                self.pointer_down = false;
            }
        }
        Ok(())
    }
}

fn pointer_vertices(position: Point2d, down: bool) -> [LineVertex; 4] {
    let x = (position.x * 2.0 - 1.0) as f32;
    let y = (1.0 - position.y * 2.0) as f32;
    let radius = if down { 0.04 } else { 0.025 };
    let color = if down {
        [1.0, 0.25, 0.12, 1.0]
    } else {
        [0.1, 1.0, 0.9, 1.0]
    };
    [
        LineVertex {
            position: [x - radius, y],
            color,
        },
        LineVertex {
            position: [x + radius, y],
            color,
        },
        LineVertex {
            position: [x, y - radius],
            color,
        },
        LineVertex {
            position: [x, y + radius],
            color,
        },
    ]
}
