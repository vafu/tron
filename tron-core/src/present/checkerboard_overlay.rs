use anyhow::Result;
use tron_api::{CheckerboardDetection, Presenter, Size};

use crate::present::line_overlay::{LineOverlayPresenter, LineOverlayView, LineVertex};
use crate::present::wgpu::{NdcRect, project_frame_point};

pub struct CheckerboardOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub detection: &'frame CheckerboardDetection,
    pub color: [f32; 4],
    pub rect: NdcRect,
    pub target_size: Size,
}

pub struct CheckerboardOverlayPresenter {
    lines: LineOverlayPresenter,
    vertices: Vec<LineVertex>,
}

impl CheckerboardOverlayPresenter {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            lines: LineOverlayPresenter::new(device, surface_format, "tron-checkerboard-overlay"),
            vertices: Vec::new(),
        }
    }
}

impl<'frame, 'pass> Presenter<CheckerboardOverlayView<'frame, 'pass>>
    for CheckerboardOverlayPresenter
{
    fn present(&mut self, view: CheckerboardOverlayView<'frame, 'pass>) -> Result<()> {
        build_vertices(
            view.detection,
            view.color,
            view.rect,
            view.target_size,
            &mut self.vertices,
        );
        if self.vertices.is_empty() {
            return Ok(());
        }
        self.lines.present(LineOverlayView {
            device: view.device,
            queue: view.queue,
            pass: view.pass,
            vertices: &self.vertices,
        })
    }
}

fn build_vertices(
    detection: &CheckerboardDetection,
    color: [f32; 4],
    rect: NdcRect,
    target_size: Size,
    vertices: &mut Vec<LineVertex>,
) {
    vertices.clear();
    let cols = detection.spec.inner_corners.width as usize;
    let rows = detection.spec.inner_corners.height as usize;
    if cols == 0 || rows == 0 || detection.corners.len() < cols * rows {
        return;
    }
    for y in 0..rows {
        for x in 0..cols.saturating_sub(1) {
            push_line(
                detection,
                color,
                rect,
                target_size,
                y * cols + x,
                y * cols + x + 1,
                vertices,
            );
        }
    }
    for y in 0..rows.saturating_sub(1) {
        for x in 0..cols {
            push_line(
                detection,
                color,
                rect,
                target_size,
                y * cols + x,
                (y + 1) * cols + x,
                vertices,
            );
        }
    }
}

fn push_line(
    detection: &CheckerboardDetection,
    color: [f32; 4],
    rect: NdcRect,
    target_size: Size,
    a: usize,
    b: usize,
    vertices: &mut Vec<LineVertex>,
) {
    vertices.push(LineVertex {
        position: project_corner(detection, rect, target_size, a),
        color,
    });
    vertices.push(LineVertex {
        position: project_corner(detection, rect, target_size, b),
        color,
    });
}

fn project_corner(
    detection: &CheckerboardDetection,
    rect: NdcRect,
    target_size: Size,
    index: usize,
) -> [f32; 2] {
    let corner = detection.corners[index];
    project_frame_point(
        [corner.x as f32, corner.y as f32],
        detection.frame_size,
        rect,
        target_size,
    )
}
