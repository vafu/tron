use anyhow::Result;
use tron_api::{Renderer, Size};

use crate::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};
use crate::render::wgpu::{NdcRect, project_frame_point};
use crate::roi::mediapipe::{HandLandmark, HandLandmarks};

const HAND_CONNECTIONS: [(usize, usize); 20] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

pub struct HandLandmarksOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub landmarks: &'frame HandLandmarks,
    pub frame_size: Size,
    pub rect: NdcRect,
    pub target_size: Size,
}

pub struct HandLandmarksOverlayRenderer {
    lines: LineOverlayRenderer,
    vertices: Vec<LineVertex>,
}

impl HandLandmarksOverlayRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            lines: LineOverlayRenderer::new(device, surface_format, "tron-hand-landmarks-overlay"),
            vertices: Vec::new(),
        }
    }
}

impl<'frame, 'pass> Renderer<HandLandmarksOverlayView<'frame, 'pass>>
    for HandLandmarksOverlayRenderer
{
    fn render(&mut self, view: HandLandmarksOverlayView<'frame, 'pass>) -> Result<()> {
        build_vertices(
            view.landmarks,
            view.frame_size,
            view.rect,
            view.target_size,
            &mut self.vertices,
        );
        self.lines.render(LineOverlayView {
            device: view.device,
            queue: view.queue,
            pass: view.pass,
            vertices: &self.vertices,
        })
    }
}

fn build_vertices(
    landmarks: &HandLandmarks,
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
    vertices: &mut Vec<LineVertex>,
) {
    vertices.clear();
    for (a, b) in HAND_CONNECTIONS {
        push_segment(
            landmarks.points[a],
            landmarks.points[b],
            frame_size,
            rect,
            target_size,
            [1.0, 0.92, 0.16, 1.0],
            vertices,
        );
    }
    for point in landmarks.points {
        push_cross(
            point,
            frame_size,
            rect,
            target_size,
            [0.1, 0.9, 1.0, 1.0],
            vertices,
        );
    }
}

fn push_segment(
    a: HandLandmark,
    b: HandLandmark,
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
    color: [f32; 4],
    vertices: &mut Vec<LineVertex>,
) {
    if !a.x.is_finite() || !a.y.is_finite() || !b.x.is_finite() || !b.y.is_finite() {
        return;
    }
    vertices.push(LineVertex {
        position: project_frame_point([a.x, a.y], frame_size, rect, target_size),
        color,
    });
    vertices.push(LineVertex {
        position: project_frame_point([b.x, b.y], frame_size, rect, target_size),
        color,
    });
}

fn push_cross(
    point: HandLandmark,
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
    color: [f32; 4],
    vertices: &mut Vec<LineVertex>,
) {
    if !point.x.is_finite() || !point.y.is_finite() {
        return;
    }
    let radius = 3.0;
    push_segment(
        HandLandmark {
            x: point.x - radius,
            y: point.y,
            z: point.z,
        },
        HandLandmark {
            x: point.x + radius,
            y: point.y,
            z: point.z,
        },
        frame_size,
        rect,
        target_size,
        color,
        vertices,
    );
    push_segment(
        HandLandmark {
            x: point.x,
            y: point.y - radius,
            z: point.z,
        },
        HandLandmark {
            x: point.x,
            y: point.y + radius,
            z: point.z,
        },
        frame_size,
        rect,
        target_size,
        color,
        vertices,
    );
}
