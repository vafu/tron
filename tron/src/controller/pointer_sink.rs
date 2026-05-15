use anyhow::Result;
use tron_api::{
    Point2d, PointerEvent, PointerJoystickVisualization, PointerOutput, PointerVisualization, Sink,
    Size,
};
use tron_core::render::thick_line_overlay::{
    ThickLine, ThickLineOverlayRenderer, ThickLineOverlayView,
};

pub struct PointerOverlaySink {
    overlay: ThickLineOverlayRenderer,
    position: Point2d,
    down: bool,
    joystick: Option<PointerJoystickVisualization>,
    lines: Vec<ThickLine>,
}

const JOYSTICK_DEADZONE_SEGMENTS: usize = 48;
const JOYSTICK_MAX_LINE_COUNT: usize = JOYSTICK_DEADZONE_SEGMENTS + 5;

impl PointerOverlaySink {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            overlay: ThickLineOverlayRenderer::new(
                device,
                surface_format,
                "tron-controller-pointer-overlay",
            ),
            position: Point2d::splat(0.5),
            down: false,
            joystick: None,
            lines: Vec::with_capacity(2 + JOYSTICK_MAX_LINE_COUNT),
        }
    }

    pub fn render<'frame, 'pass>(
        &'frame mut self,
        device: &'frame wgpu::Device,
        queue: &'frame wgpu::Queue,
        pass: &'frame mut wgpu::RenderPass<'pass>,
        target_size: Size,
    ) -> Result<()> {
        self.lines.clear();
        if let Some(joystick) = self.joystick {
            append_joystick_lines(&mut self.lines, joystick);
        }
        append_pointer_cross(&mut self.lines, self.position, self.down);
        pollster::block_on(self.overlay.consume(ThickLineOverlayView {
            device,
            queue,
            pass,
            lines: &self.lines,
            target_size,
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
                    (Some(position), _) => position,
                    (None, position) => (position + delta).clamp(Point2d::ZERO, Point2d::ONE),
                };
            }
            PointerEvent::Down { .. } => self.down = true,
            PointerEvent::Up { .. } => self.down = false,
            PointerEvent::Click { .. } => {}
            PointerEvent::Cancel { .. } => {
                self.down = false;
                self.joystick = None;
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl Sink<PointerOutput> for PointerOverlaySink {
    async fn consume(&mut self, output: PointerOutput) -> Result<()> {
        match output {
            PointerOutput::Event(event) => Sink::<PointerEvent>::consume(self, event).await,
            PointerOutput::Visualization(PointerVisualization::Joystick(joystick)) => {
                self.joystick = if joystick.anchor.is_some() || joystick.current.is_some() {
                    Some(joystick)
                } else {
                    None
                };
                Ok(())
            }
        }
    }
}

fn append_pointer_cross(lines: &mut Vec<ThickLine>, position: Point2d, down: bool) {
    let radius = if down { 0.055 } else { 0.038 };
    let width_px = if down { 9.0 } else { 7.0 };
    let color = if down {
        [1.0, 0.25, 0.12, 1.0]
    } else {
        [0.1, 1.0, 0.9, 1.0]
    };
    append_line(
        lines,
        position + Point2d::new(-radius, 0.0),
        position + Point2d::new(radius, 0.0),
        width_px,
        color,
    );
    append_line(
        lines,
        position + Point2d::new(0.0, -radius),
        position + Point2d::new(0.0, radius),
        width_px,
        color,
    );
}

fn append_joystick_lines(lines: &mut Vec<ThickLine>, visualization: PointerJoystickVisualization) {
    if let Some(anchor) = visualization.anchor {
        append_deadzone_circle(lines, anchor, visualization.deadzone_radius);
        append_anchor_cross(lines, anchor);
    }
    if let (Some(anchor), Some(current)) = (visualization.anchor, visualization.current) {
        append_line(lines, anchor, current, 5.0, [1.0, 1.0, 1.0, 0.65]);
    }
    if let Some(current) = visualization.current {
        append_current_cross(lines, current, visualization.engaged);
    }
}

fn append_deadzone_circle(lines: &mut Vec<ThickLine>, center: Point2d, radius: f64) {
    let color = [1.0, 1.0, 1.0, 0.34];
    for index in 0..JOYSTICK_DEADZONE_SEGMENTS {
        let start = circle_point(center, radius, index);
        let end = circle_point(center, radius, index + 1);
        append_line(lines, start, end, 5.0, color);
    }
}

fn append_anchor_cross(lines: &mut Vec<ThickLine>, center: Point2d) {
    append_cross(lines, center, 0.018, 6.0, [1.0, 0.85, 0.1, 0.95]);
}

fn append_current_cross(lines: &mut Vec<ThickLine>, center: Point2d, engaged: bool) {
    let color = if engaged {
        [0.2, 0.75, 1.0, 0.95]
    } else {
        [0.65, 0.65, 0.65, 0.65]
    };
    append_cross(lines, center, 0.014, 5.0, color);
}

fn append_cross(
    lines: &mut Vec<ThickLine>,
    center: Point2d,
    radius: f64,
    width_px: f32,
    color: [f32; 4],
) {
    append_line(
        lines,
        center + Point2d::new(-radius, 0.0),
        center + Point2d::new(radius, 0.0),
        width_px,
        color,
    );
    append_line(
        lines,
        center + Point2d::new(0.0, -radius),
        center + Point2d::new(0.0, radius),
        width_px,
        color,
    );
}

fn append_line(
    lines: &mut Vec<ThickLine>,
    start: Point2d,
    end: Point2d,
    width_px: f32,
    color: [f32; 4],
) {
    lines.push(ThickLine {
        start: ndc(start),
        end: ndc(end),
        width_px,
        color,
    })
}

fn circle_point(center: Point2d, radius: f64, index: usize) -> Point2d {
    let angle = (index as f64 / JOYSTICK_DEADZONE_SEGMENTS as f64) * std::f64::consts::TAU;
    center + Point2d::new(angle.cos() * radius, angle.sin() * radius)
}

fn ndc(position: Point2d) -> [f32; 2] {
    [
        (position.x * 2.0 - 1.0) as f32,
        (1.0 - position.y * 2.0) as f32,
    ]
}
