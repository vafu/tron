use anyhow::Result;
use tron_api::{Rect, Renderer, Size};

use crate::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};
use crate::render::wgpu::{NdcRect, project_frame_point};

pub struct RoiOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub roi: Rect,
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
            view.roi,
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
    roi: Rect,
    color: [f32; 4],
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
) -> [LineVertex; 8] {
    let left_top = project_frame_point([roi.x as f32, roi.y as f32], frame_size, rect, target_size);
    let right_top = project_frame_point(
        [(roi.x + roi.size.width) as f32, roi.y as f32],
        frame_size,
        rect,
        target_size,
    );
    let right_bottom = project_frame_point(
        [
            (roi.x + roi.size.width) as f32,
            (roi.y + roi.size.height) as f32,
        ],
        frame_size,
        rect,
        target_size,
    );
    let left_bottom = project_frame_point(
        [roi.x as f32, (roi.y + roi.size.height) as f32],
        frame_size,
        rect,
        target_size,
    );
    [
        LineVertex {
            position: left_top,
            color,
        },
        LineVertex {
            position: right_top,
            color,
        },
        LineVertex {
            position: right_top,
            color,
        },
        LineVertex {
            position: right_bottom,
            color,
        },
        LineVertex {
            position: right_bottom,
            color,
        },
        LineVertex {
            position: left_bottom,
            color,
        },
        LineVertex {
            position: left_bottom,
            color,
        },
        LineVertex {
            position: left_top,
            color,
        },
    ]
}
