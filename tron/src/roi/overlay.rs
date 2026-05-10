use crate::roi::RoiRect;
use anyhow::Result;
use tron_api::{Presenter, Size};
use tron_core::present::wgpu::NdcRect;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
};

pub struct RoiOverlayView<'pass> {
    pub queue: &'pass wgpu::Queue,
    pub pass: &'pass mut wgpu::RenderPass<'pass>,
    pub roi: RoiRect,
    pub frame_size: Size,
    pub rect: NdcRect,
    pub target_size: Size,
}

pub struct RoiOverlayPresenter {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
}

impl RoiOverlayPresenter {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tron-roi-overlay-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tron-roi-overlay-pipeline-layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tron-roi-overlay-pipeline"),
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
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tron-roi-overlay-vertices"),
            contents: bytemuck::cast_slice(
                &[Vertex {
                    position: [0.0, 0.0],
                    color: [0.0, 0.0, 0.0, 0.0],
                }; 8],
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            vertex_buffer,
        }
    }
}

impl<'pass> Presenter<RoiOverlayView<'pass>> for RoiOverlayPresenter {
    fn present(&mut self, view: RoiOverlayView<'pass>) -> Result<()> {
        let vertices = roi_vertices(view.roi, view.frame_size, view.rect, view.target_size);
        view.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        view.pass.set_pipeline(&self.pipeline);
        view.pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        view.pass.draw(0..8, 0..1);
        Ok(())
    }
}

fn roi_vertices(roi: RoiRect, frame_size: Size, rect: NdcRect, target_size: Size) -> [Vertex; 8] {
    let (x0, y0, x1, y1) = letterbox(frame_size, rect, target_size);
    let fx0 = roi.x as f32 / frame_size.width.max(1) as f32;
    let fy0 = roi.y as f32 / frame_size.height.max(1) as f32;
    let fx1 = (roi.x + roi.width) as f32 / frame_size.width.max(1) as f32;
    let fy1 = (roi.y + roi.height) as f32 / frame_size.height.max(1) as f32;
    let left = lerp(x0, x1, fx0);
    let right = lerp(x0, x1, fx1);
    let top = lerp(y1, y0, fy0);
    let bottom = lerp(y1, y0, fy1);
    let color = [0.1, 0.9, 1.0, 1.0];
    [
        Vertex {
            position: [left, top],
            color,
        },
        Vertex {
            position: [right, top],
            color,
        },
        Vertex {
            position: [right, top],
            color,
        },
        Vertex {
            position: [right, bottom],
            color,
        },
        Vertex {
            position: [right, bottom],
            color,
        },
        Vertex {
            position: [left, bottom],
            color,
        },
        Vertex {
            position: [left, bottom],
            color,
        },
        Vertex {
            position: [left, top],
            color,
        },
    ]
}

fn letterbox(frame: Size, rect: NdcRect, target: Size) -> (f32, f32, f32, f32) {
    let rect_width_ndc = (rect.x1 - rect.x0).abs();
    let rect_height_ndc = (rect.y1 - rect.y0).abs();
    let rect_pixel_width = target.width as f32 * rect_width_ndc / 2.0;
    let rect_pixel_height = target.height as f32 * rect_height_ndc / 2.0;
    let frame_aspect = frame.width as f32 / frame.height.max(1) as f32;
    let rect_aspect = rect_pixel_width / rect_pixel_height.max(1.0);

    if rect_aspect > frame_aspect {
        let width_ndc = rect_width_ndc * frame_aspect / rect_aspect;
        let cx = (rect.x0 + rect.x1) * 0.5;
        (cx - width_ndc * 0.5, rect.y0, cx + width_ndc * 0.5, rect.y1)
    } else {
        let height_ndc = rect_height_ndc * rect_aspect / frame_aspect;
        let cy = (rect.y0 + rect.y1) * 0.5;
        (
            rect.x0,
            cy - height_ndc * 0.5,
            rect.x1,
            cy + height_ndc * 0.5,
        )
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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
