use anyhow::Result;
use tron_api::{OrientedBoundingBox, Rect, Renderer, Size};

use crate::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};
use crate::render::wgpu::{NdcRect, project_frame_point};

pub struct RoiOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub roi: Rect,
    pub oriented_roi: Option<OrientedBoundingBox>,
    pub color: [f32; 4],
    pub frame_size: Size,
    pub rect: NdcRect,
    pub target_size: Size,
}

pub struct RoiOverlayRenderer {
    lines: LineOverlayRenderer,
    vertices: [LineVertex; 8],
}

impl RoiOverlayRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            lines: LineOverlayRenderer::new(device, surface_format, "tron-roi-overlay"),
            vertices: [LineVertex {
                position: [0.0, 0.0],
                color: [0.0, 0.0, 0.0, 0.0],
            }; 8],
        }
    }
}

impl<'frame, 'pass> Renderer<RoiOverlayView<'frame, 'pass>> for RoiOverlayRenderer {
    fn render(&mut self, view: RoiOverlayView<'frame, 'pass>) -> Result<()> {
        self.vertices = roi_vertices(
            view.oriented_roi
                .unwrap_or_else(|| rect_to_oriented_box(view.roi)),
            view.color,
            view.frame_size,
            view.rect,
            view.target_size,
        );
        self.lines.render(LineOverlayView {
            device: view.device,
            queue: view.queue,
            pass: view.pass,
            vertices: &self.vertices,
        })
    }
}

fn roi_vertices(
    roi: OrientedBoundingBox,
    color: [f32; 4],
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
) -> [LineVertex; 8] {
    let [c0, c1, c2, c3] = roi
        .corners
        .map(|corner| project_frame_point(corner, frame_size, rect, target_size));
    [
        LineVertex {
            position: c0,
            color,
        },
        LineVertex {
            position: c1,
            color,
        },
        LineVertex {
            position: c1,
            color,
        },
        LineVertex {
            position: c2,
            color,
        },
        LineVertex {
            position: c2,
            color,
        },
        LineVertex {
            position: c3,
            color,
        },
        LineVertex {
            position: c3,
            color,
        },
        LineVertex {
            position: c0,
            color,
        },
    ]
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
