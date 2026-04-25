use crate::camera::SharedImage;
use crate::pipeline::SharedHand;
use crate::proximity::SharedProx;
use crate::skeleton_render::{letterbox_rect, SkeletonRenderer};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
};

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Vertex; 6] {
    // image origin at top-left, NDC y up. First row of texture should appear at y1.
    let tl = Vertex { pos: [x0, y1], uv: [0.0, 0.0] };
    let tr = Vertex { pos: [x1, y1], uv: [1.0, 0.0] };
    let bl = Vertex { pos: [x0, y0], uv: [0.0, 1.0] };
    let br = Vertex { pos: [x1, y0], uv: [1.0, 1.0] };
    [tl, bl, br, tl, br, tr]
}

struct TexQuad {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    w: u32,
    h: u32,
    last_seq: u64,
    /// NDC pane bounds (x0, y0, x1, y1) — area allocated to this quad.
    pane: (f32, f32, f32, f32),
}

impl TexQuad {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        w: u32,
        h: u32,
        rect: (f32, f32, f32, f32),
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cam-tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cam-vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self { texture, bind_group, vbuf, w, h, last_seq: u64::MAX, pane: rect }
    }

    /// Recompute the vertex buffer to fit the image inside `pane` while
    /// preserving its aspect ratio in physical pixels (letterbox/pillarbox).
    fn fit(&self, queue: &wgpu::Queue, win: PhysicalSize<u32>) {
        let win_w = win.width.max(1) as f32;
        let win_h = win.height.max(1) as f32;
        let (px0, py0, px1, py1) = self.pane;
        // Pane size in physical pixels.
        let pane_px_w = (px1 - px0) * 0.5 * win_w;
        let pane_px_h = (py1 - py0) * 0.5 * win_h;
        let pane_ar = pane_px_w / pane_px_h;
        let img_ar = self.w as f32 / self.h as f32;
        let (mut x0, mut y0, mut x1, mut y1) = self.pane;
        if img_ar > pane_ar {
            // image wider than pane → fit width, shrink height
            let new_px_h = pane_px_w / img_ar;
            let half_ndc = new_px_h / win_h; // half-extent in NDC
            let cy = (py0 + py1) * 0.5;
            y0 = cy - half_ndc;
            y1 = cy + half_ndc;
        } else {
            // image taller than pane → fit height, shrink width
            let new_px_w = pane_px_h * img_ar;
            let half_ndc = new_px_w / win_w;
            let cx = (px0 + px1) * 0.5;
            x0 = cx - half_ndc;
            x1 = cx + half_ndc;
        }
        let verts = quad(x0, y0, x1, y1);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    fn upload(&mut self, queue: &wgpu::Queue, rgba: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.w * 4),
                rows_per_image: Some(self.h),
            },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
    }
}

struct SolidQuad {
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    ubuf: wgpu::Buffer, // kept alive for the bind group
    vbuf: wgpu::Buffer,
}

impl SolidQuad {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        rect: (f32, f32, f32, f32),
        color: [f32; 4],
    ) -> Self {
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solid-ubuf"),
            contents: bytemuck::cast_slice(&color),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid-bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solid-vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self { bind_group, ubuf, vbuf }
    }
}

pub struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    window: Arc<Window>,

    tex_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    tex_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    solid_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    sampler: wgpu::Sampler,

    rgb: TexQuad,
    ir: TexQuad,
    bar_bg: SolidQuad,
    bar_fill: SolidQuad,
    skeleton: SkeletonRenderer,
    rgb_pane: (f32, f32, f32, f32),

    rgb_src: SharedImage,
    ir_src: SharedImage,
    prox_src: SharedProx,
    hand_src: SharedHand,
    prox_max: i64,
}

impl Gfx {
    pub fn new(
        window: Arc<Window>,
        rgb_src: SharedImage,
        ir_src: SharedImage,
        prox_src: SharedProx,
        hand_src: SharedHand,
        rgb_size: (u32, u32),
        ir_size: (u32, u32),
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).context("create surface")?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("no adapter")?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let solid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let tex_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex-pl"),
            bind_group_layouts: &[&tex_bgl],
            push_constant_ranges: &[],
        });
        let solid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid-pl"),
            bind_group_layouts: &[&solid_bgl],
            push_constant_ranges: &[],
        });

        let tex_pipeline = make_pipeline(&device, &tex_pl, &shader, "fs_tex", format);
        let solid_pipeline = make_pipeline(&device, &solid_pl, &shader, "fs_solid", format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Layout: top half split into RGB (left) and IR (right). Bottom strip = bar.
        let rgb_rect = (-1.0, -0.5, 0.0, 1.0);
        let ir_rect = (0.0, -0.5, 1.0, 1.0);
        let bar_bg_rect = (-1.0, -1.0, 1.0, -0.5);
        let bar_fill_rect = (-0.95, -0.9, -0.95, -0.6); // updated each frame

        let rgb = TexQuad::new(&device, &tex_bgl, &sampler, rgb_size.0, rgb_size.1, rgb_rect);
        let ir = TexQuad::new(&device, &tex_bgl, &sampler, ir_size.0, ir_size.1, ir_rect);
        rgb.fit(&queue, size);
        ir.fit(&queue, size);
        let bar_bg = SolidQuad::new(&device, &solid_bgl, bar_bg_rect, [0.08, 0.08, 0.10, 1.0]);
        let bar_fill = SolidQuad::new(&device, &solid_bgl, bar_fill_rect, [0.20, 0.85, 0.45, 1.0]);

        let skeleton = SkeletonRenderer::new(&device, format);

        Ok(Self {
            surface, device, queue, config, size, window,
            tex_pipeline, solid_pipeline, tex_bgl, solid_bgl, sampler,
            rgb, ir, bar_bg, bar_fill,
            skeleton, rgb_pane: rgb_rect,
            rgb_src, ir_src, prox_src, hand_src,
            prox_max: 1024,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 { return; }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.rgb.fit(&self.queue, size);
        self.ir.fit(&self.queue, size);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Pull latest camera frames
        if let Some(f) = self.rgb_src.lock().unwrap().as_ref() {
            if f.seq != self.rgb.last_seq {
                self.rgb.upload(&self.queue, &f.data);
                self.rgb.last_seq = f.seq;
            }
        }
        if let Some(f) = self.ir_src.lock().unwrap().as_ref() {
            if f.seq != self.ir.last_seq {
                self.ir.upload(&self.queue, &f.data);
                self.ir.last_seq = f.seq;
            }
        }

        // Update prox bar fill
        let prox = *self.prox_src.lock().unwrap();
        if let Some(p) = prox {
            if p > self.prox_max { self.prox_max = p; }
            let norm = (p as f32 / self.prox_max as f32).clamp(0.0, 1.0);
            let x0 = -0.95;
            let x1 = x0 + 1.9 * norm;
            let verts = quad(x0, -0.9, x1, -0.6);
            self.queue.write_buffer(&self.bar_fill.vbuf, 0, bytemuck::cast_slice(&verts));
        }

        // Pull hand state, update skeleton mesh.
        let hand = self.hand_src.lock().unwrap().clone();
        let mut have_hand = false;
        let mut gesture_label = "";
        if let Some(state) = &hand {
            have_hand = true;
            gesture_label = state.gesture.map(|g| g.name()).unwrap_or("");
            let clip = letterbox_rect(self.rgb_pane, self.rgb.w, self.rgb.h, self.size.width, self.size.height);
            self.skeleton.update(&self.queue, &state.landmarks, clip);
        }

        // Window title combines proximity + gesture.
        let title = match (prox, gesture_label) {
            (Some(p), g) if !g.is_empty() => format!("tron — prox: {p} — {g}"),
            (Some(p), _) => format!("tron — prox: {p}"),
            (None, g) if !g.is_empty() => format!("tron — {g}"),
            _ => "tron".into(),
        };
        self.window.set_title(&title);

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.tex_pipeline);
            rp.set_bind_group(0, &self.rgb.bind_group, &[]);
            rp.set_vertex_buffer(0, self.rgb.vbuf.slice(..));
            rp.draw(0..6, 0..1);

            rp.set_bind_group(0, &self.ir.bind_group, &[]);
            rp.set_vertex_buffer(0, self.ir.vbuf.slice(..));
            rp.draw(0..6, 0..1);

            rp.set_pipeline(&self.solid_pipeline);
            rp.set_bind_group(0, &self.bar_bg.bind_group, &[]);
            rp.set_vertex_buffer(0, self.bar_bg.vbuf.slice(..));
            rp.draw(0..6, 0..1);
            rp.set_bind_group(0, &self.bar_fill.bind_group, &[]);
            rp.set_vertex_buffer(0, self.bar_fill.vbuf.slice(..));
            rp.draw(0..6, 0..1);

            if have_hand {
                self.skeleton.draw(&mut rp);
            }
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
        Ok(())
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fs_entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs_entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs",
            buffers: &[VERTEX_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: fs_entry,
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    })
}
