use anyhow::Result;
use tron_api::{Point2d, PointerEvent, Sink};
use tron_core::render::line_overlay::{LineOverlayRenderer, LineOverlayView, LineVertex};

pub struct PointerOverlaySink {
    overlay: LineOverlayRenderer,
    position: Option<Point2d>,
    down: bool,
    vertices: [LineVertex; POINTER_VERTEX_COUNT],
}

const POINTER_VERTEX_COUNT: usize = 20;

impl PointerOverlaySink {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            overlay: LineOverlayRenderer::new(
                device,
                surface_format,
                "tron-controller-pointer-overlay",
            ),
            position: None,
            down: false,
            vertices: [TRANSPARENT_VERTEX; POINTER_VERTEX_COUNT],
        }
    }

    pub fn render<'frame, 'pass>(
        &'frame mut self,
        device: &'frame wgpu::Device,
        queue: &'frame wgpu::Queue,
        pass: &'frame mut wgpu::RenderPass<'pass>,
    ) -> Result<()> {
        let Some(position) = self.position else {
            return Ok(());
        };
        self.vertices = pointer_vertices(position, self.down);
        pollster::block_on(self.overlay.consume(LineOverlayView {
            device,
            queue,
            pass,
            vertices: &self.vertices,
        }))
    }
}

#[async_trait::async_trait(?Send)]
impl Sink<PointerEvent> for PointerOverlaySink {
    async fn consume(&mut self, event: PointerEvent) -> Result<()> {
        match event {
            PointerEvent::Move {
                position, delta, ..
            } => {
                self.position = match (position, self.position) {
                    (Some(position), _) => Some(position),
                    (None, Some(position)) => {
                        Some((position + delta).clamp(Point2d::ZERO, Point2d::ONE))
                    }
                    (None, None) => None,
                };
            }
            PointerEvent::Down { .. } => self.down = true,
            PointerEvent::Up { .. } => self.down = false,
            PointerEvent::Click { .. } => {}
            PointerEvent::Cancel { .. } => {
                self.position = None;
                self.down = false;
            }
        }
        Ok(())
    }
}

const TRANSPARENT_VERTEX: LineVertex = LineVertex {
    position: [0.0, 0.0],
    color: [0.0, 0.0, 0.0, 0.0],
};

fn pointer_vertices(position: Point2d, down: bool) -> [LineVertex; POINTER_VERTEX_COUNT] {
    let x = (position.x * 2.0 - 1.0) as f32;
    let y = (1.0 - position.y * 2.0) as f32;
    let radius = if down { 0.055 } else { 0.038 };
    let color = if down {
        [1.0, 0.25, 0.12, 1.0]
    } else {
        [0.1, 1.0, 0.9, 1.0]
    };
    let offsets = [-0.012, -0.006, 0.0, 0.006, 0.012];
    let mut vertices = [TRANSPARENT_VERTEX; POINTER_VERTEX_COUNT];
    for (index, offset) in offsets.into_iter().enumerate() {
        let base = index * 4;
        vertices[base] = LineVertex {
            position: [x - radius, y + offset],
            color,
        };
        vertices[base + 1] = LineVertex {
            position: [x + radius, y + offset],
            color,
        };
        vertices[base + 2] = LineVertex {
            position: [x + offset, y - radius],
            color,
        };
        vertices[base + 3] = LineVertex {
            position: [x + offset, y + radius],
            color,
        };
    }
    vertices
}
