use crate::camera::SharedImage;
use crate::pipeline::{SharedHand, SharedMask, SharedPointer};
use crate::proximity::SharedProx;
use crate::skeleton_render::{SkeletonRenderer, letterbox_rect};
use crate::types::{Gesture, Image};
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::window::Window;

#[derive(Clone, Copy)]
pub struct RenderOptions {
    pub cube: bool,
    pub skeleton: bool,
    pub classifier_debug: bool,
}

pub struct Gfx {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    tex_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,

    /// Center pane: raw RGB camera, with skeleton overlay.
    main_view: TexQuad,
    /// Side pane: RGB darkened by IR mask (the "dimmed" debug image).
    masked_view: TexQuad,
    /// Side pane: grayscale IR foreground signal (the mask itself).
    mask_view: TexQuad,

    bar_bg: SolidQuad,
    bar_fill: SolidQuad,

    skeleton: SkeletonRenderer,
    cube: CubeRenderer,
    depth: DepthTexture,

    main_pane: (f32, f32, f32, f32),

    rgb_src: SharedImage,
    #[allow(dead_code)]
    ir_src: SharedImage,
    prox_src: SharedProx,
    hand_src: SharedHand,
    mask_src: SharedMask,
    pointer_src: SharedPointer,
    options: RenderOptions,

    /// Scratch buffer for R8 → RGBA8 expansion when uploading the mask.
    mask_rgba: Vec<u8>,

    prox_max: i64,
    last_grab_pos: Option<[f32; 2]>,
    render_timing: RenderTiming,
}

#[derive(Default)]
struct RenderTiming {
    last_log: Option<Instant>,
    frames: u32,
    lock_us: u64,
    upload_us: u64,
    overlay_us: u64,
    encode_us: u64,
    submit_us: u64,
}

impl Gfx {
    pub async fn new(
        window: Arc<Window>,
        rgb_src: SharedImage,
        ir_src: SharedImage,
        prox_src: SharedProx,
        hand_src: SharedHand,
        mask_src: SharedMask,
        pointer_src: SharedPointer,
        options: RenderOptions,
        rgb_size: (u32, u32),
        ir_size: (u32, u32),
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("request adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_bgl"),
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
            label: Some("solid_bgl"),
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

        let tex_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex_layout"),
            bind_group_layouts: &[&tex_bgl],
            push_constant_ranges: &[],
        });
        let solid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid_layout"),
            bind_group_layouts: &[&solid_bgl],
            push_constant_ranges: &[],
        });

        let tex_pipeline = make_pipeline(&device, &tex_layout, &shader, "fs_tex", format);
        let solid_pipeline = make_pipeline(&device, &solid_layout, &shader, "fs_solid", format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // UI Layout (normalized NDC -1..1):
        //   Top row: helpers — IR-diff mask (left), masked RGB / pre-landmark (right).
        //   Below:   main RGB + landmarks, full width.
        let mask_rect = (-0.95, 0.30, -0.05, 0.95);
        let masked_rect = (0.05, 0.30, 0.95, 0.95);
        let main_rect = (-0.95, -0.90, 0.95, 0.20);

        let main_view = TexQuad::new(
            &device, &tex_bgl, &sampler, rgb_size.0, rgb_size.1, main_rect,
        );
        let masked_view = TexQuad::new(
            &device,
            &tex_bgl,
            &sampler,
            rgb_size.0,
            rgb_size.1,
            masked_rect,
        );
        let mask_view = TexQuad::new(&device, &tex_bgl, &sampler, ir_size.0, ir_size.1, mask_rect);
        main_view.fit(&queue, size);
        masked_view.fit(&queue, size);
        mask_view.fit(&queue, size);

        let bar_bg = SolidQuad::new(
            &device,
            &queue,
            &solid_bgl,
            [0.1, 0.1, 0.15, 1.0],
            (-1.0, -1.0, 1.0, -0.97),
        );
        let bar_fill = SolidQuad::new(
            &device,
            &queue,
            &solid_bgl,
            [0.2, 0.8, 0.4, 1.0],
            (-1.0, -1.0, -1.0, -0.97),
        );

        let skeleton = SkeletonRenderer::new(&device, format);
        let cube = CubeRenderer::new(&device, format);
        let depth = DepthTexture::new(&device, size);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            tex_pipeline,
            solid_pipeline,
            main_view,
            masked_view,
            mask_view,
            bar_bg,
            bar_fill,
            cube,
            depth,
            main_pane: main_rect,
            rgb_src,
            ir_src,
            prox_src,
            hand_src,
            mask_src,
            pointer_src,
            options,
            mask_rgba: Vec::new(),
            skeleton,
            prox_max: 1,
            last_grab_pos: None,
            render_timing: RenderTiming::default(),
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.main_view.fit(&self.queue, size);
        self.masked_view.fit(&self.queue, size);
        self.mask_view.fit(&self.queue, size);
        self.depth = DepthTexture::new(&self.device, size);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let log_start = self
            .render_timing
            .last_log
            .get_or_insert_with(Instant::now)
            .to_owned();
        let t_lock = Instant::now();
        // Pull latest data
        let (rgb_opt, hand, mask_opt, pointer) = {
            let _span = tracing::debug_span!("gfx.lock_inputs").entered();
            (
                self.rgb_src.lock().unwrap().clone(),
                self.hand_src.lock().unwrap().clone(),
                self.mask_src.lock().unwrap().clone(),
                *self.pointer_src.lock().unwrap(),
            )
        };
        self.render_timing.lock_us += t_lock.elapsed().as_micros() as u64;

        let t_upload = Instant::now();
        {
            let _span = tracing::debug_span!("gfx.upload").entered();
            // Main pane: raw RGB camera (skeleton drawn on top later).
            if let Some(f) = &rgb_opt {
                if f.seq != self.main_view.last_seq {
                    self.main_view.upload(&self.queue, &f.data);
                    self.main_view.last_seq = f.seq;
                }
            }

            // Masked-RGB pane: pipeline's debug image (RGB darkened by IR mask).
            // Falls back to raw RGB until the pipeline produces its first frame.
            if let Some(state) = &hand {
                if let Some(dbg) = &state.debug_image {
                    if dbg.seq != self.masked_view.last_seq {
                        self.masked_view.upload(&self.queue, &dbg.data);
                        self.masked_view.last_seq = dbg.seq;
                    }
                }
            } else if let Some(f) = &rgb_opt {
                if f.seq != self.masked_view.last_seq {
                    self.masked_view.upload(&self.queue, &f.data);
                    self.masked_view.last_seq = f.seq;
                }
            }

            // Mask pane: grayscale IR-diff. Texture is RGBA8 so expand R8 → RGBA8.
            if let Some(m) = &mask_opt {
                if m.seq != self.mask_view.last_seq {
                    let needed = (m.width * m.height) as usize * 4;
                    if self.mask_rgba.len() != needed {
                        self.mask_rgba.resize(needed, 255);
                    }
                    expand_to_rgba(m, &mut self.mask_rgba);
                    self.mask_view.upload(&self.queue, &self.mask_rgba);
                    self.mask_view.last_seq = m.seq;
                }
            }
        }
        self.render_timing.upload_us += t_upload.elapsed().as_micros() as u64;

        let t_overlay = Instant::now();
        let mut have_overlay = false;
        let mut gesture_label = "";
        let mut gesture_debug = String::new();
        let prox;
        {
            let _span = tracing::debug_span!("gfx.overlay_update").entered();
            // Update prox bar fill
            prox = *self.prox_src.lock().unwrap();
            if let Some(p) = prox {
                if p > self.prox_max {
                    self.prox_max = p;
                }
                let norm = (p as f32 / self.prox_max as f32).clamp(0.0, 1.0);
                self.bar_fill
                    .set_rect(&self.queue, (-1.0, -1.0, -1.0 + 2.0 * norm, -0.97));
            }

            // Pull hand state, update skeleton + ROI mesh.
            if let Some(state) = &hand {
                have_overlay = true;
                gesture_label = state.gesture.map(|g| g.name()).unwrap_or("");
                gesture_debug = state.gesture_features.summary();
                if self.options.skeleton {
                    let clip = letterbox_rect(
                        self.main_pane,
                        self.main_view.w,
                        self.main_view.h,
                        self.size.width,
                        self.size.height,
                    );
                    self.skeleton.update(
                        &self.queue,
                        Some(&state.landmarks),
                        Some(&state.roi),
                        clip,
                        (self.size.width, self.size.height),
                        state.gesture == Some(Gesture::Fist),
                    );
                }
            }
        }

        // Window title combines proximity + gesture.
        let title = match (prox, gesture_label) {
            (Some(p), g) if !g.is_empty() && self.options.classifier_debug => {
                format!("tron — prox: {p} — {g} — {gesture_debug}")
            }
            (Some(p), g) if !g.is_empty() => format!("tron — prox: {p} — {g}"),
            (Some(p), _) => format!("tron — prox: {p}"),
            (None, g) if !g.is_empty() && self.options.classifier_debug => {
                format!("tron — {g} — {gesture_debug}")
            }
            (None, g) if !g.is_empty() => format!("tron — {g}"),
            _ => "tron".into(),
        };
        self.window.set_title(&title);

        if self.options.cube {
            if let Some(pointer) = pointer {
                if pointer.grabbed {
                    let pos = [pointer.position.x, pointer.position.y];
                    if let Some(last) = self.last_grab_pos {
                        let dx = pos[0] - last[0];
                        let dy = pos[1] - last[1];
                        self.cube.rotate(-dx * 8.0, -dy * 8.0);
                    }
                    self.last_grab_pos = Some(pos);
                } else {
                    self.last_grab_pos = None;
                }
            } else {
                self.last_grab_pos = None;
            }
            self.cube
                .update(&self.queue, self.size.width, self.size.height);
        } else {
            self.last_grab_pos = None;
        }
        self.render_timing.overlay_us += t_overlay.elapsed().as_micros() as u64;

        let t_encode = Instant::now();
        let _encode_span = tracing::debug_span!("gfx.encode").entered();
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-2d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.tex_pipeline);
            for q in [&self.main_view, &self.masked_view, &self.mask_view] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }

            rp.set_pipeline(&self.solid_pipeline);
            for q in [&self.bar_bg, &self.bar_fill] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }
        }

        if self.options.cube {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-cube"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.cube.draw(&mut rp);
        }

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if self.options.cube {
                self.cube.draw_overlay(&mut rp);
            }
            if have_overlay && self.options.skeleton {
                self.skeleton.draw(&mut rp);
            }
        }
        self.render_timing.encode_us += t_encode.elapsed().as_micros() as u64;
        drop(_encode_span);
        let t_submit = Instant::now();
        {
            let _span = tracing::debug_span!("gfx.submit").entered();
            self.queue.submit(Some(enc.finish()));
            frame.present();
        }
        self.render_timing.submit_us += t_submit.elapsed().as_micros() as u64;
        self.render_timing.frames += 1;
        self.log_render_timing(log_start);
        Ok(())
    }

    fn log_render_timing(&mut self, log_start: Instant) {
        let elapsed = log_start.elapsed();
        if elapsed < Duration::from_secs(2) {
            return;
        }
        let n = self.render_timing.frames.max(1) as f32;
        tracing::debug!(
            target: "tron::gfx",
            fps = self.render_timing.frames as f32 / elapsed.as_secs_f32(),
            lock_ms = self.render_timing.lock_us as f32 / n / 1000.0,
            upload_ms = self.render_timing.upload_us as f32 / n / 1000.0,
            overlay_ms = self.render_timing.overlay_us as f32 / n / 1000.0,
            encode_ms = self.render_timing.encode_us as f32 / n / 1000.0,
            submit_ms = self.render_timing.submit_us as f32 / n / 1000.0,
            "render timing"
        );
        self.render_timing = RenderTiming {
            last_log: Some(Instant::now()),
            ..Default::default()
        };
    }
}

struct DepthTexture {
    view: wgpu::TextureView,
}

impl DepthTexture {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    fn new(device: &wgpu::Device, size: PhysicalSize<u32>) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

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

struct CubeRenderer {
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
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube-shader"),
            source: wgpu::ShaderSource::Wgsl(CUBE_SHADER.into()),
        });
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

    fn rotate(&mut self, dx: f32, dy: f32) {
        self.rot_y += dx;
        self.rot_x = (self.rot_x + dy).clamp(-1.4, 1.4);
    }

    fn update(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
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

    fn draw<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.vbuf.slice(..));
        rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..self.index_count, 0, 0..1);
    }

    fn draw_overlay<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        if self.edge_count == 0 {
            return;
        }
        rp.set_pipeline(&self.edge_pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.edge_vbuf.slice(..));
        rp.draw(0..self.edge_count, 0..1);
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

/// Expand a single-channel mask into a pre-sized RGBA8 buffer (gray, opaque).
fn expand_to_rgba(src: &Image, dst: &mut [u8]) {
    for (i, g) in src.grey_iter().enumerate() {
        let o = i * 4;
        dst[o] = g;
        dst[o + 1] = g;
        dst[o + 2] = g;
        dst[o + 3] = 255;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
};

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Vertex; 6] {
    [
        Vertex {
            pos: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [x0, y0],
            uv: [0.0, 1.0],
        },
        Vertex {
            pos: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            pos: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            pos: [x1, y1],
            uv: [1.0, 0.0],
        },
    ]
}

struct TexQuad {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    w: u32,
    h: u32,
    rect: (f32, f32, f32, f32),
    last_seq: u64,
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
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            texture,
            bind_group,
            vbuf,
            w,
            h,
            rect,
            last_seq: 0,
        }
    }

    fn fit(&self, queue: &wgpu::Queue, win_size: PhysicalSize<u32>) {
        let (x0, y0, x1, y1) =
            letterbox_rect(self.rect, self.w, self.h, win_size.width, win_size.height);
        let verts = quad(x0, y0, x1, y1);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    fn upload(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.w * 4),
                rows_per_image: Some(self.h),
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// A coloured rectangle in NDC. Renders with `fs_solid`, which reads its color
/// from a 16-byte uniform.
struct SolidQuad {
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    rect: (f32, f32, f32, f32),
}

impl SolidQuad {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        color: [f32; 4],
        rect: (f32, f32, f32, f32),
    ) -> Self {
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid-color"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&color));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid-bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid-vbuf"),
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        Self {
            bind_group,
            vbuf,
            rect,
        }
    }

    fn set_rect(&mut self, queue: &wgpu::Queue, rect: (f32, f32, f32, f32)) {
        self.rect = rect;
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }
}

const CUBE_VERTICES: &[CubeVertex] = &[
    CubeVertex {
        pos: [-0.7, -0.7, 0.7],
        color: [0.1, 0.9, 1.0],
    },
    CubeVertex {
        pos: [0.7, -0.7, 0.7],
        color: [0.9, 1.0, 1.0],
    },
    CubeVertex {
        pos: [0.7, 0.7, 0.7],
        color: [0.2, 0.6, 1.0],
    },
    CubeVertex {
        pos: [-0.7, 0.7, 0.7],
        color: [0.0, 0.4, 0.9],
    },
    CubeVertex {
        pos: [-0.7, -0.7, -0.7],
        color: [0.0, 0.3, 0.8],
    },
    CubeVertex {
        pos: [0.7, -0.7, -0.7],
        color: [0.0, 0.8, 0.9],
    },
    CubeVertex {
        pos: [0.7, 0.7, -0.7],
        color: [0.7, 0.9, 1.0],
    },
    CubeVertex {
        pos: [-0.7, 0.7, -0.7],
        color: [0.0, 0.6, 1.0],
    },
];

const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 4, 5,
    1, 4, 1, 0,
];

const CUBE_EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn build_cube_overlay(out: &mut Vec<CubeEdgeVertex>, mvp: [[f32; 4]; 4], win_w: u32, win_h: u32) {
    let mut projected = [ProjectedPoint::default(); 8];
    for (i, v) in CUBE_VERTICES.iter().enumerate() {
        projected[i] = project_point(mvp, v.pos);
    }

    let ndcx = 2.0 / win_w as f32;
    let ndcy = 2.0 / win_h as f32;
    let edge_color = [0.35, 1.0, 1.2];
    for &(a, b) in CUBE_EDGES {
        push_cube_edge(
            out,
            projected[a].pos,
            projected[b].pos,
            8.5,
            ndcx,
            ndcy,
            edge_color,
            1.15,
        );
    }

    for p in projected {
        let radius = 13.0 + (1.0 - p.depth).clamp(0.0, 1.0) * 25.0;
        push_cube_corner(out, p.pos, radius, ndcx, ndcy, edge_color, 1.2);
    }
}

#[derive(Clone, Copy, Default)]
struct ProjectedPoint {
    pos: [f32; 2],
    depth: f32,
}

fn project_point(m: [[f32; 4]; 4], p: [f32; 3]) -> ProjectedPoint {
    let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
    let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
    let z = m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2];
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    let inv_w = if w.abs() > 1e-5 { 1.0 / w } else { 1.0 };
    ProjectedPoint {
        pos: [x * inv_w, y * inv_w],
        depth: (z * inv_w).clamp(0.0, 1.0),
    }
}

fn push_cube_edge(
    out: &mut Vec<CubeEdgeVertex>,
    a: [f32; 2],
    b: [f32; 2],
    half_w_px: f32,
    ndcx: f32,
    ndcy: f32,
    color: [f32; 3],
    intensity: f32,
) {
    let dx_px = (b[0] - a[0]) / ndcx;
    let dy_px = (b[1] - a[1]) / ndcy;
    let len = (dx_px * dx_px + dy_px * dy_px).sqrt().max(1e-6);
    let perp_x_ndc = (-dy_px / len * half_w_px) * ndcx;
    let perp_y_ndc = (dx_px / len * half_w_px) * ndcy;
    let am = [a[0] - perp_x_ndc, a[1] - perp_y_ndc];
    let ap = [a[0] + perp_x_ndc, a[1] + perp_y_ndc];
    let bp = [b[0] + perp_x_ndc, b[1] + perp_y_ndc];
    let bm = [b[0] - perp_x_ndc, b[1] - perp_y_ndc];
    for (pos, uv) in [
        (am, [0.0, -1.0]),
        (bm, [1.0, -1.0]),
        (bp, [1.0, 1.0]),
        (am, [0.0, -1.0]),
        (bp, [1.0, 1.0]),
        (ap, [0.0, 1.0]),
    ] {
        out.push(CubeEdgeVertex {
            pos,
            uv,
            kind: 1.0,
            color,
            intensity,
        });
    }
}

fn push_cube_corner(
    out: &mut Vec<CubeEdgeVertex>,
    c: [f32; 2],
    r_px: f32,
    ndcx: f32,
    ndcy: f32,
    color: [f32; 3],
    intensity: f32,
) {
    let rx = r_px * ndcx;
    let ry = r_px * ndcy;
    let tl = [c[0] - rx, c[1] + ry];
    let tr = [c[0] + rx, c[1] + ry];
    let bl = [c[0] - rx, c[1] - ry];
    let br = [c[0] + rx, c[1] - ry];
    for (pos, uv) in [
        (tl, [-1.0, 1.0]),
        (bl, [-1.0, -1.0]),
        (br, [1.0, -1.0]),
        (tl, [-1.0, 1.0]),
        (br, [1.0, -1.0]),
        (tr, [1.0, 1.0]),
    ] {
        out.push(CubeEdgeVertex {
            pos,
            uv,
            kind: 0.0,
            color,
            intensity,
        });
    }
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (far * near) / (near - far), 0.0],
    ]
}

fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn rotation_x(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_y(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = a.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

const CUBE_SHADER: &str = r#"
struct U { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;

struct VsIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec3<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) local: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  out.pos = u.mvp * vec4<f32>(in.pos, 1.0);
  out.color = in.color;
  out.local = in.pos;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let a = abs(in.local);
  let near_x = smoothstep(0.46, 0.66, a.x);
  let near_y = smoothstep(0.46, 0.66, a.y);
  let near_z = smoothstep(0.46, 0.66, a.z);
  let edge = clamp(near_x * near_y + near_x * near_z + near_y * near_z, 0.0, 1.0);
  let pulse = 0.82 + 0.18 * sin((in.local.x + in.local.y + in.local.z) * 9.0);
  let face = vec3<f32>(0.02, 0.10, 0.14);
  let tron = vec3<f32>(0.25, 1.05, 1.25) * (1.25 + edge * 1.8 * pulse);
  let hot = vec3<f32>(1.0, 1.0, 1.0) * edge * 0.9;
  let color = mix(face, tron + hot, edge);
  return vec4<f32>(color, 0.94);
}

struct EdgeIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) kind: f32,
  @location(3) color: vec3<f32>,
  @location(4) intensity: f32,
};

struct EdgeOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) kind: f32,
  @location(2) color: vec3<f32>,
  @location(3) intensity: f32,
};

@vertex
fn vs_edge(in: EdgeIn) -> EdgeOut {
  var out: EdgeOut;
  out.pos = vec4<f32>(in.pos, 0.0, 1.0);
  out.uv = in.uv;
  out.kind = in.kind;
  out.color = in.color;
  out.intensity = in.intensity;
  return out;
}

@fragment
fn fs_edge(in: EdgeOut) -> @location(0) vec4<f32> {
  let kind = i32(in.kind + 0.5);
  if (kind == 0) {
    let r = length(in.uv);
    let ring = exp(-pow((r - 0.58) * 4.2, 2.0));
    let pip = exp(-pow(r * 4.2, 2.0));
    let halo = exp(-r * 2.0) * 0.65 * smoothstep(1.08, 0.45, r);
    let core = clamp(ring + pip, 0.0, 1.0);
    let lum = (core + halo) * in.intensity;
    let col = mix(in.color, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.45, 1.0, core));
    return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
  }

  let d = abs(in.uv.y);
  let core = smoothstep(0.22, 0.0, d);
  let halo = exp(-pow(d * 2.0, 2.0)) * 0.55;
  let pulse = exp(-pow(abs(in.uv.x - 0.5) * 4.2, 2.0)) * 0.35;
  let lum = (core + halo + pulse) * in.intensity;
  let col = mix(in.color, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.35, 1.0, core + pulse));
  return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
}
"#;
