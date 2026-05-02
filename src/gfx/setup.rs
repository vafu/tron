use super::*;

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
}
