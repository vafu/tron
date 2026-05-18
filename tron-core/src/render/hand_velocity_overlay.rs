use anyhow::Result;
use tron_api::{Sink, Size};

use crate::process::landmark_velocity::{HandLandmarkMotion, HandLandmarkVelocity};
use crate::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};
use crate::render::wgpu::{NdcRect, project_frame_point};
use crate::roi::mediapipe::HandLandmark;

const VELOCITY_LANDMARKS: [usize; 6] = [0, 4, 8, 12, 16, 20];
const VECTOR_SECONDS: f64 = 0.08;
const VECTOR_COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

pub struct HandVelocityOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub motion: &'frame HandLandmarkMotion,
    pub frame_size: Size,
    pub rect: NdcRect,
    pub target_size: Size,
}

pub struct HandVelocityOverlayRenderer {
    lines: LineOverlayRenderer,
    vertices: Vec<LineVertex>,
}

impl HandVelocityOverlayRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            lines: LineOverlayRenderer::new(device, surface_format, "tron-hand-velocity-overlay"),
            vertices: Vec::new(),
        }
    }
}

#[async_trait::async_trait(?Send)]
impl<'frame, 'pass> Sink<HandVelocityOverlayView<'frame, 'pass>> for HandVelocityOverlayRenderer {
    async fn consume(&mut self, view: HandVelocityOverlayView<'frame, 'pass>) -> Result<()> {
        build_vertices(
            view.motion,
            view.frame_size,
            view.rect,
            view.target_size,
            &mut self.vertices,
        );
        self.lines
            .consume(LineOverlayView {
                device: view.device,
                queue: view.queue,
                pass: view.pass,
                vertices: &self.vertices,
            })
            .await
    }
}

fn build_vertices(
    motion: &HandLandmarkMotion,
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
    vertices: &mut Vec<LineVertex>,
) {
    vertices.clear();
    for index in VELOCITY_LANDMARKS {
        push_velocity(
            motion.landmarks.points[index],
            motion.velocities[index],
            frame_size,
            rect,
            target_size,
            vertices,
        );
    }
}

fn push_velocity(
    point: HandLandmark,
    velocity: HandLandmarkVelocity,
    frame_size: Size,
    rect: NdcRect,
    target_size: Size,
    vertices: &mut Vec<LineVertex>,
) {
    if !point.x.is_finite()
        || !point.y.is_finite()
        || !velocity.x.is_finite()
        || !velocity.y.is_finite()
    {
        return;
    }
    let end = [
        point.x + velocity.x * VECTOR_SECONDS,
        point.y + velocity.y * VECTOR_SECONDS,
    ];
    vertices.push(LineVertex {
        position: project_frame_point(
            [point.x as f32, point.y as f32],
            frame_size,
            rect,
            target_size,
        ),
        color: VECTOR_COLOR,
    });
    vertices.push(LineVertex {
        position: project_frame_point(
            [end[0] as f32, end[1] as f32],
            frame_size,
            rect,
            target_size,
        ),
        color: VECTOR_COLOR,
    });
}
