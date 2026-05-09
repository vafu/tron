mod mesh;

use super::depth::DepthTexture;
use mesh::{
    CUBE_INDICES, CUBE_VERTICES, build_cube_overlay, mat4_mul, perspective, rotation_x, rotation_y,
    translation,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

const CUBE_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<CubeVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeUniform {
    mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeEdgeVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    kind: f32,
    color: [f32; 3],
    intensity: f32,
}

const CUBE_EDGE_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<CubeEdgeVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32x3, 4 => Float32
    ],
};

pub(super) struct CubeRenderer {
    pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    edge_vbuf: wgpu::Buffer,
    index_count: u32,
    edge_count: u32,
    edge_capacity: usize,
    rot_x: f32,
    rot_y: f32,
}

impl CubeRenderer {
    pub(super) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cube-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cube-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cube-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[CUBE_VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cube-edge-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_edge",
                buffers: &[CUBE_EDGE_VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_edge",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-ubuf"),
            size: std::mem::size_of::<CubeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cube-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-vbuf"),
            size: (CUBE_VERTICES.len() * std::mem::size_of::<CubeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-ibuf"),
            size: (CUBE_INDICES.len() * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_capacity = (12 + 8) * 6;
        let edge_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-edge-vbuf"),
            size: (edge_capacity * std::mem::size_of::<CubeEdgeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            edge_pipeline,
            bind_group,
            ubuf,
            vbuf,
            ibuf,
            edge_vbuf,
            index_count: CUBE_INDICES.len() as u32,
            edge_count: 0,
            edge_capacity,
            rot_x: 0.35,
            rot_y: -0.45,
        }
    }

    pub(super) fn rotate(&mut self, dx: f32, dy: f32) {
        self.rot_y += dx;
        self.rot_x = (self.rot_x + dy).clamp(-1.4, 1.4);
    }

    pub(super) fn update(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(CUBE_VERTICES));
        queue.write_buffer(&self.ibuf, 0, bytemuck::cast_slice(CUBE_INDICES));

        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let proj = perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view = translation(0.0, 0.0, -4.0);
        let model = mat4_mul(rotation_y(self.rot_y), rotation_x(self.rot_x));
        let mvp = mat4_mul(mat4_mul(proj, view), model);
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&CubeUniform { mvp }));

        let mut verts = Vec::with_capacity(self.edge_capacity);
        build_cube_overlay(&mut verts, mvp, width.max(1), height.max(1));
        self.edge_count = verts.len() as u32;
        if !verts.is_empty() {
            queue.write_buffer(&self.edge_vbuf, 0, bytemuck::cast_slice(&verts));
        }
    }

    pub(super) fn draw<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.vbuf.slice(..));
        rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..self.index_count, 0, 0..1);
    }

    pub(super) fn draw_overlay<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        if self.edge_count == 0 {
            return;
        }
        rp.set_pipeline(&self.edge_pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.edge_vbuf.slice(..));
        rp.draw(0..self.edge_count, 0..1);
    }
}
