use anyhow::Result;
use tron_api::{Sink, Size};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub struct ThickLine {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub width_px: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ThickLineVertex {
    position: [f32; 2],
    color: [f32; 4],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<ThickLineVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
};

pub struct ThickLineOverlayView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub lines: &'frame [ThickLine],
    pub target_size: Size,
}

pub struct ThickLineOverlayRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertices: Vec<ThickLineVertex>,
}

impl ThickLineOverlayRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat, label: &str) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}-shader")),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}-pipeline-layout")),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{label}-pipeline")),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vertex_capacity = 6;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-vertices")),
            contents: bytemuck::cast_slice(
                &[ThickLineVertex {
                    position: [0.0, 0.0],
                    color: [0.0, 0.0, 0.0, 0.0],
                }; 6],
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            vertex_buffer,
            vertex_capacity,
            vertices: Vec::new(),
        }
    }
}

#[async_trait::async_trait(?Send)]
impl<'frame, 'pass> Sink<ThickLineOverlayView<'frame, 'pass>> for ThickLineOverlayRenderer {
    async fn consume(&mut self, view: ThickLineOverlayView<'frame, 'pass>) -> Result<()> {
        if view.lines.is_empty() {
            return Ok(());
        }

        self.vertices.clear();
        for line in view.lines {
            append_line_quad(&mut self.vertices, *line, view.target_size);
        }
        if self.vertices.is_empty() {
            return Ok(());
        }

        if self.vertices.len() > self.vertex_capacity {
            self.vertex_capacity = self.vertices.len().next_power_of_two();
            self.vertex_buffer = view.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tron-thick-line-overlay-vertices"),
                size: (self.vertex_capacity * std::mem::size_of::<ThickLineVertex>())
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        view.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        view.pass.set_pipeline(&self.pipeline);
        view.pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        view.pass.draw(0..self.vertices.len() as u32, 0..1);
        Ok(())
    }
}

fn append_line_quad(vertices: &mut Vec<ThickLineVertex>, line: ThickLine, target_size: Size) {
    let start = ndc_to_pixel(line.start, target_size);
    let end = ndc_to_pixel(line.end, target_size);
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return;
    }

    let half = line.width_px.max(1.0) * 0.5;
    let nx = -dy / len * half;
    let ny = dx / len * half;
    let p0 = pixel_to_ndc([start[0] + nx, start[1] + ny], target_size);
    let p1 = pixel_to_ndc([end[0] + nx, end[1] + ny], target_size);
    let p2 = pixel_to_ndc([end[0] - nx, end[1] - ny], target_size);
    let p3 = pixel_to_ndc([start[0] - nx, start[1] - ny], target_size);

    vertices.extend_from_slice(&[
        vertex(p0, line.color),
        vertex(p1, line.color),
        vertex(p2, line.color),
        vertex(p0, line.color),
        vertex(p2, line.color),
        vertex(p3, line.color),
    ]);
}

fn vertex(position: [f32; 2], color: [f32; 4]) -> ThickLineVertex {
    ThickLineVertex { position, color }
}

fn ndc_to_pixel(position: [f32; 2], target_size: Size) -> [f32; 2] {
    [
        (position[0] + 1.0) * 0.5 * target_size.width as f32,
        (1.0 - position[1]) * 0.5 * target_size.height as f32,
    ]
}

fn pixel_to_ndc(position: [f32; 2], target_size: Size) -> [f32; 2] {
    [
        position[0] / target_size.width as f32 * 2.0 - 1.0,
        1.0 - position[1] / target_size.height as f32 * 2.0,
    ]
}

const SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;
