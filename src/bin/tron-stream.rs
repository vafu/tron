#[allow(dead_code)]
#[path = "../camera/mod.rs"]
mod camera;
#[allow(dead_code)]
#[path = "../types/mod.rs"]
mod types;

use anyhow::{Context, Result};
use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use v4l::FourCC;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::video::capture::Parameters;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Clone)]
struct Config {
    camera: Option<String>,
    sensor: Sensor,
    format: Option<StreamFormat>,
    decoder: DecoderKind,
    size: Option<(u32, u32)>,
    fps: Option<u32>,
    buffers: u32,
    drain_latest: bool,
    list_cameras: bool,
    list_camera_modes: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sensor {
    Rgb,
    Ir,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamFormat {
    Yuyv,
    Mjpg,
    Grey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoderKind {
    OpenCv,
    TurboJpeg,
}

impl StreamFormat {
    fn fourcc(self) -> FourCC {
        match self {
            StreamFormat::Yuyv => FourCC::new(b"YUYV"),
            StreamFormat::Mjpg => FourCC::new(b"MJPG"),
            StreamFormat::Grey => FourCC::new(b"GREY"),
        }
    }

    fn published_format(self) -> PixelFormat {
        match self {
            StreamFormat::Yuyv => PixelFormat::Yuyv,
            StreamFormat::Mjpg => PixelFormat::Bgra8,
            StreamFormat::Grey => PixelFormat::Grey,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PixelFormat {
    Yuyv,
    Bgra8,
    Grey,
}

impl PixelFormat {
    fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            PixelFormat::Yuyv => wgpu::TextureFormat::Rg8Unorm,
            PixelFormat::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
            PixelFormat::Grey => wgpu::TextureFormat::R8Unorm,
        }
    }

    fn bytes_per_row(self, width: u32) -> u32 {
        match self {
            PixelFormat::Yuyv => width * 2,
            PixelFormat::Bgra8 => width * 4,
            PixelFormat::Grey => width,
        }
    }
}

struct RawFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: PixelFormat,
    seq: u64,
    timestamp: Instant,
}

type SharedRawFrame = Arc<Mutex<RawFrame>>;

struct App {
    frame: SharedRawFrame,
    window: Option<Arc<Window>>,
    renderer: Option<StreamRenderer>,
    window_size: (u32, u32),
    loop_timing: LoopTiming,
}

#[derive(Default)]
struct LoopTiming {
    last_log: Option<Instant>,
    redraws: u32,
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    if cfg.list_cameras {
        if cfg.list_camera_modes {
            println!("{}", camera::select::available_summary_detailed());
        } else {
            println!("{}", camera::select::available_summary());
        }
        return Ok(());
    }

    let camera_set = match cfg.camera.as_deref() {
        Some(name) => camera::select::by_name(name)?,
        None => camera::select::default_set(),
    };
    let mut stream = match cfg.sensor {
        Sensor::Rgb => with_overrides(camera_set.rgb, cfg.size, cfg.fps),
        Sensor::Ir => with_overrides(camera_set.ir, cfg.size, cfg.fps),
    };
    let stream_format = cfg.format.unwrap_or(match cfg.sensor {
        Sensor::Rgb => StreamFormat::Mjpg,
        Sensor::Ir => StreamFormat::Grey,
    });
    if cfg.sensor == Sensor::Ir && stream_format != StreamFormat::Grey {
        anyhow::bail!("IR stream only supports --format grey in tron-stream");
    }
    let pixel_format = stream_format.published_format();
    if cfg.sensor == Sensor::Rgb && stream_format == StreamFormat::Grey {
        anyhow::bail!("RGB stream supports --format mjpg or yuyv");
    }
    if cfg.sensor == Sensor::Rgb && stream_format == StreamFormat::Mjpg && cfg.size.is_none() {
        // Use the browser-like compressed path at a useful default resolution.
        stream.width = 1280;
        stream.height = 720;
        stream.fps = Some(stream.fps.unwrap_or(30));
    }
    if cfg.sensor == Sensor::Rgb && stream_format == StreamFormat::Yuyv {
        stream.fps = Some(stream.fps.unwrap_or(30));
    }
    if cfg.sensor == Sensor::Ir {
        stream.fps = Some(stream.fps.unwrap_or(30));
    }
    eprintln!(
        "stream: {:?} {} {}x{}{} {:?}",
        cfg.sensor,
        stream.path,
        stream.width,
        stream.height,
        stream
            .fps
            .map(|fps| format!("@{fps}fps"))
            .unwrap_or_default(),
        stream_format
    );

    let frame = Arc::new(Mutex::new(RawFrame {
        data: Vec::new(),
        width: stream.width,
        height: stream.height,
        format: pixel_format,
        seq: 0,
        timestamp: Instant::now(),
    }));
    spawn_raw_capture(
        stream.path.clone(),
        stream.width,
        stream.height,
        stream.fps,
        stream_format,
        cfg.decoder,
        cfg.buffers,
        cfg.drain_latest,
        frame.clone(),
    )?;

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        frame,
        window: None,
        renderer: None,
        window_size: (1280, 720),
        loop_timing: LoopTiming::default(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn spawn_raw_capture(
    path: String,
    width: u32,
    height: u32,
    fps: Option<u32>,
    format: StreamFormat,
    decoder: DecoderKind,
    buffers: u32,
    drain_latest: bool,
    shared: SharedRawFrame,
) -> Result<()> {
    thread::Builder::new()
        .name(format!("raw-cap:{path}"))
        .spawn(move || {
            if let Err(e) = run_raw_capture(
                &path,
                width,
                height,
                fps,
                format,
                decoder,
                buffers,
                drain_latest,
                shared,
            ) {
                eprintln!("camera {path} raw thread exited: {e:#}");
            }
        })?;
    Ok(())
}

fn run_raw_capture(
    path: &str,
    width: u32,
    height: u32,
    fps: Option<u32>,
    format: StreamFormat,
    decoder: DecoderKind,
    buffers: u32,
    drain_latest: bool,
    shared: SharedRawFrame,
) -> Result<()> {
    let dev = Device::with_path(path).with_context(|| format!("open {path}"))?;
    let mut fmt = dev.format()?;
    fmt.width = width;
    fmt.height = height;
    fmt.fourcc = format.fourcc();
    let fmt = dev.set_format(&fmt)?;
    if let Some(fps) = fps {
        match dev.set_params(&Parameters::with_fps(fps)) {
            Ok(params) => eprintln!("{path}: requested {fps}fps, interval {}", params.interval),
            Err(err) => eprintln!("{path}: failed to request {fps}fps: {err}"),
        }
    }
    eprintln!(
        "{path}: negotiated {}x{} {}",
        fmt.width, fmt.height, fmt.fourcc
    );

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, buffers)
        .with_context(|| format!("create mmap stream with {buffers} buffers"))?;
    let mut seq = 0u64;
    let mut last_log = Instant::now();
    let mut frames = 0u32;
    let mut wait_us = 0u64;
    let mut decode_timing = DecodeTiming::default();
    let mut publish_us = 0u64;
    let mut drained_frames = 0u32;
    let mut v4l_timestamp_delta_us = 0u64;
    let mut v4l_timestamp_deltas = 0u32;
    let mut v4l_sequence_gaps = 0u32;
    let mut last_v4l_sequence: Option<u32> = None;
    let mut last_v4l_timestamp_us: Option<i64> = None;
    let mut decoded_owned = Vec::new();
    let mut turbojpeg = match decoder {
        DecoderKind::TurboJpeg => Some(turbojpeg::Decompressor::new()?),
        DecoderKind::OpenCv => None,
    };
    let mut drain_candidate = Vec::new();

    loop {
        if drain_latest {
            loop {
                let t_wait = Instant::now();
                let (buf, meta) = stream.next()?;
                wait_us += t_wait.elapsed().as_micros() as u64;
                update_v4l_stats(
                    meta,
                    &mut last_v4l_sequence,
                    &mut v4l_sequence_gaps,
                    &mut last_v4l_timestamp_us,
                    &mut v4l_timestamp_delta_us,
                    &mut v4l_timestamp_deltas,
                );

                drain_candidate.clear();
                let used_len = (meta.bytesused as usize).min(buf.len());
                let frame_bytes = if used_len > 0 { &buf[..used_len] } else { buf };
                drain_candidate.extend_from_slice(frame_bytes);
                frames += 1;

                if buffers <= 1 || stream.handle().poll(LINUX_POLLIN, 0)? == 0 {
                    let decoded = decode_frame(
                        format,
                        decoder,
                        &drain_candidate,
                        &mut decoded_owned,
                        turbojpeg.as_mut(),
                    )?;
                    decode_timing.add(decoded.timing);
                    let t_publish = Instant::now();
                    seq = seq.wrapping_add(1);
                    publish_frame(&shared, &fmt, decoded, seq);
                    publish_us += t_publish.elapsed().as_micros() as u64;
                    break;
                }
                drained_frames = drained_frames.saturating_add(1);
            }
        } else {
            let t_wait = Instant::now();
            let (buf, meta) = stream.next()?;
            wait_us += t_wait.elapsed().as_micros() as u64;
            update_v4l_stats(
                meta,
                &mut last_v4l_sequence,
                &mut v4l_sequence_gaps,
                &mut last_v4l_timestamp_us,
                &mut v4l_timestamp_delta_us,
                &mut v4l_timestamp_deltas,
            );
            let used_len = (meta.bytesused as usize).min(buf.len());
            let frame_bytes = if used_len > 0 { &buf[..used_len] } else { buf };
            let decoded = decode_frame(
                format,
                decoder,
                frame_bytes,
                &mut decoded_owned,
                turbojpeg.as_mut(),
            )?;
            decode_timing.add(decoded.timing);
            let t_publish = Instant::now();
            seq = seq.wrapping_add(1);
            publish_frame(&shared, &fmt, decoded, seq);
            publish_us += t_publish.elapsed().as_micros() as u64;
            frames += 1;
        }

        if last_log.elapsed() >= Duration::from_secs(2) {
            let elapsed = last_log.elapsed().as_secs_f32();
            let n = frames.max(1) as f32;
            let v4l_ts_delta_ms = if v4l_timestamp_deltas > 0 {
                v4l_timestamp_delta_us as f32 / v4l_timestamp_deltas as f32 / 1000.0
            } else {
                0.0
            };
            eprintln!(
                "raw camera: fps={:.1} wait={:.3}ms v4l_dt={:.3}ms v4l_seq={} gaps={} drained={} mjpg_decode={:.3}ms color={:.3}ms passthrough={:.3}ms publish={:.3}ms",
                frames as f32 / elapsed,
                wait_us as f32 / n / 1000.0,
                v4l_ts_delta_ms,
                last_v4l_sequence.unwrap_or(0),
                v4l_sequence_gaps,
                drained_frames,
                decode_timing.mjpg_decode_us as f32 / n / 1000.0,
                decode_timing.color_us as f32 / n / 1000.0,
                decode_timing.passthrough_us as f32 / n / 1000.0,
                publish_us as f32 / n / 1000.0
            );
            last_log = Instant::now();
            frames = 0;
            wait_us = 0;
            decode_timing = DecodeTiming::default();
            publish_us = 0;
            drained_frames = 0;
            v4l_timestamp_delta_us = 0;
            v4l_timestamp_deltas = 0;
            v4l_sequence_gaps = 0;
        }
    }
}

const LINUX_POLLIN: i16 = 0x0001;

fn update_v4l_stats(
    meta: &v4l::buffer::Metadata,
    last_v4l_sequence: &mut Option<u32>,
    v4l_sequence_gaps: &mut u32,
    last_v4l_timestamp_us: &mut Option<i64>,
    v4l_timestamp_delta_us: &mut u64,
    v4l_timestamp_deltas: &mut u32,
) {
    if let Some(prev) = *last_v4l_sequence {
        let expected = prev.wrapping_add(1);
        if meta.sequence != expected {
            *v4l_sequence_gaps = v4l_sequence_gaps.saturating_add(1);
        }
    }
    *last_v4l_sequence = Some(meta.sequence);
    let timestamp_us = meta.timestamp.sec.saturating_mul(1_000_000) + meta.timestamp.usec;
    if let Some(prev) = *last_v4l_timestamp_us {
        let delta = timestamp_us.saturating_sub(prev);
        if delta >= 0 {
            *v4l_timestamp_delta_us = v4l_timestamp_delta_us.saturating_add(delta as u64);
            *v4l_timestamp_deltas = v4l_timestamp_deltas.saturating_add(1);
        }
    }
    *last_v4l_timestamp_us = Some(timestamp_us);
}

fn publish_frame(shared: &SharedRawFrame, fmt: &v4l::Format, decoded: DecodedFrame<'_>, seq: u64) {
    let mut frame = shared.lock().unwrap();
    frame.data.clear();
    frame.data.extend_from_slice(decoded.data);
    frame.width = fmt.width;
    frame.height = fmt.height;
    frame.format = decoded.format;
    frame.seq = seq;
    frame.timestamp = Instant::now();
}

struct DecodedFrame<'a> {
    data: &'a [u8],
    format: PixelFormat,
    timing: DecodeTiming,
}

#[derive(Clone, Copy, Debug, Default)]
struct DecodeTiming {
    mjpg_decode_us: u64,
    color_us: u64,
    passthrough_us: u64,
}

impl DecodeTiming {
    fn add(&mut self, other: Self) {
        self.mjpg_decode_us += other.mjpg_decode_us;
        self.color_us += other.color_us;
        self.passthrough_us += other.passthrough_us;
    }
}

fn decode_frame<'a>(
    format: StreamFormat,
    decoder: DecoderKind,
    data: &'a [u8],
    decoded_owned: &'a mut Vec<u8>,
    turbojpeg: Option<&mut turbojpeg::Decompressor>,
) -> Result<DecodedFrame<'a>> {
    match format {
        StreamFormat::Yuyv => {
            let t = Instant::now();
            Ok(DecodedFrame {
                data,
                format: PixelFormat::Yuyv,
                timing: DecodeTiming {
                    passthrough_us: t.elapsed().as_micros() as u64,
                    ..Default::default()
                },
            })
        }
        StreamFormat::Grey => {
            let t = Instant::now();
            Ok(DecodedFrame {
                data,
                format: PixelFormat::Grey,
                timing: DecodeTiming {
                    passthrough_us: t.elapsed().as_micros() as u64,
                    ..Default::default()
                },
            })
        }
        StreamFormat::Mjpg => match decoder {
            DecoderKind::OpenCv => decode_mjpg_frame_opencv(data, decoded_owned),
            DecoderKind::TurboJpeg => {
                let Some(turbojpeg) = turbojpeg else {
                    anyhow::bail!("TurboJPEG decoder was not initialized");
                };
                decode_mjpg_frame_turbojpeg(data, decoded_owned, turbojpeg)
            }
        },
    }
}

fn decode_mjpg_frame_opencv<'a>(
    data: &[u8],
    decoded_owned: &'a mut Vec<u8>,
) -> Result<DecodedFrame<'a>> {
    let encoded = core::Vector::<u8>::from_slice(data);
    let t_decode = Instant::now();
    let bgr = imgcodecs::imdecode(&encoded, imgcodecs::IMREAD_COLOR)
        .context("decode MJPG frame with OpenCV")?;
    let mjpg_decode_us = t_decode.elapsed().as_micros() as u64;
    let mut bgra = core::Mat::default();
    let t_color = Instant::now();
    imgproc::cvt_color(
        &bgr,
        &mut bgra,
        imgproc::COLOR_BGR2BGRA,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .context("convert decoded MJPG BGR to BGRA")?;
    let color_us = t_color.elapsed().as_micros() as u64;
    let bgra_bytes = bgra.data_bytes()?;
    decoded_owned.clear();
    decoded_owned.extend_from_slice(bgra_bytes);
    Ok(DecodedFrame {
        data: decoded_owned,
        format: PixelFormat::Bgra8,
        timing: DecodeTiming {
            mjpg_decode_us,
            color_us,
            ..Default::default()
        },
    })
}

fn decode_mjpg_frame_turbojpeg<'a>(
    data: &[u8],
    decoded_owned: &'a mut Vec<u8>,
    decompressor: &mut turbojpeg::Decompressor,
) -> Result<DecodedFrame<'a>> {
    let t_decode = Instant::now();
    let header = decompressor
        .read_header(data)
        .context("read MJPG header with TurboJPEG")?;
    let len = header.width * header.height * turbojpeg::PixelFormat::BGRA.size();
    decoded_owned.resize(len, 0);
    decompressor
        .decompress(
            data,
            turbojpeg::Image {
                pixels: decoded_owned.as_mut_slice(),
                width: header.width,
                pitch: header.width * turbojpeg::PixelFormat::BGRA.size(),
                height: header.height,
                format: turbojpeg::PixelFormat::BGRA,
            },
        )
        .context("decode MJPG frame with TurboJPEG")?;
    Ok(DecodedFrame {
        data: decoded_owned,
        format: PixelFormat::Bgra8,
        timing: DecodeTiming {
            mjpg_decode_us: t_decode.elapsed().as_micros() as u64,
            ..Default::default()
        },
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("tron-stream")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.window_size.0,
                self.window_size.1,
            ));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let frame = self.frame.lock().unwrap();
        let renderer = pollster::block_on(StreamRenderer::new(
            window.clone(),
            frame.width,
            frame.height,
            frame.format,
        ))
        .expect("init stream renderer");
        drop(frame);
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                self.loop_timing.redraws += 1;
                if let Some(renderer) = self.renderer.as_mut() {
                    let frame_stats = {
                        let frame = self.frame.lock().unwrap();
                        renderer.upload_frame(&frame)
                    };
                    match renderer.render_present(frame_stats) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            renderer.resize(renderer.size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("render: {e:?}"),
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        let log_start = self
            .loop_timing
            .last_log
            .get_or_insert_with(Instant::now)
            .to_owned();
        if log_start.elapsed() >= Duration::from_secs(2) {
            let elapsed = log_start.elapsed().as_secs_f32();
            eprintln!(
                "stream loop: redraws={:.1}/s",
                self.loop_timing.redraws as f32 / elapsed
            );
            self.loop_timing = LoopTiming {
                last_log: Some(Instant::now()),
                ..Default::default()
            };
        }
    }
}

struct StreamRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    vbuf: wgpu::Buffer,
    frame_width: u32,
    frame_height: u32,
    format: PixelFormat,
    last_seq: u64,
    timing: RenderTiming,
}

#[derive(Default)]
struct RenderTiming {
    last_log: Option<Instant>,
    frames: u32,
    upload_us: u64,
    render_us: u64,
    present_us: u64,
}

impl StreamRenderer {
    async fn new(
        window: Arc<Window>,
        width: u32,
        height: u32,
        format: PixelFormat,
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
        let surface_format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raw-stream"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.texture_format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw-stream-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw-stream-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw-stream-shader"),
            source: wgpu::ShaderSource::Wgsl(STREAM_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raw-stream-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let fragment_entry = match format {
            PixelFormat::Yuyv => "fs_yuyv",
            PixelFormat::Bgra8 => "fs_bgra",
            PixelFormat::Grey => "fs_grey",
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raw-stream-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: fragment_entry,
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw-stream-vbuf"),
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let renderer = Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            bind_group,
            texture,
            vbuf,
            frame_width: width,
            frame_height: height,
            format,
            last_seq: 0,
            timing: RenderTiming::default(),
        };
        renderer.fit();
        Ok(renderer)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.fit();
    }

    fn fit(&self) {
        let verts = quad_letterboxed(
            self.frame_width,
            self.frame_height,
            self.size.width,
            self.size.height,
        );
        self.queue
            .write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    fn upload_frame(&mut self, frame: &RawFrame) -> FrameStats {
        let log_start = self
            .timing
            .last_log
            .get_or_insert_with(Instant::now)
            .to_owned();
        if frame.seq != self.last_seq && !frame.data.is_empty() {
            let t_upload = Instant::now();
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &frame.data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.format.bytes_per_row(self.frame_width)),
                    rows_per_image: Some(self.frame_height),
                },
                wgpu::Extent3d {
                    width: self.frame_width,
                    height: self.frame_height,
                    depth_or_array_layers: 1,
                },
            );
            self.last_seq = frame.seq;
            self.timing.upload_us += t_upload.elapsed().as_micros() as u64;
        }

        FrameStats {
            log_start,
            age_ms: frame.timestamp.elapsed().as_secs_f32() * 1000.0,
        }
    }

    fn render_present(&mut self, frame_stats: FrameStats) -> Result<(), wgpu::SurfaceError> {
        let t_render = Instant::now();
        let surface_frame = self.surface.get_current_texture()?;
        let view = surface_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raw-enc"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raw-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.draw(0..6, 0..1);
        }
        self.timing.render_us += t_render.elapsed().as_micros() as u64;

        let t_present = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        surface_frame.present();
        self.timing.present_us += t_present.elapsed().as_micros() as u64;
        self.timing.frames += 1;
        self.log_timing(frame_stats);
        Ok(())
    }

    fn log_timing(&mut self, frame_stats: FrameStats) {
        let elapsed = frame_stats.log_start.elapsed();
        if elapsed < Duration::from_secs(2) {
            return;
        }
        let n = self.timing.frames.max(1) as f32;
        eprintln!(
            "stream render: fps={:.1} upload={:.3}ms render={:.3}ms present={:.3}ms age={:.1}ms",
            self.timing.frames as f32 / elapsed.as_secs_f32(),
            self.timing.upload_us as f32 / n / 1000.0,
            self.timing.render_us as f32 / n / 1000.0,
            self.timing.present_us as f32 / n / 1000.0,
            frame_stats.age_ms
        );
        self.timing = RenderTiming {
            last_log: Some(Instant::now()),
            ..Default::default()
        };
    }
}

#[derive(Clone, Copy)]
struct FrameStats {
    log_start: Instant,
    age_ms: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
};

fn quad_letterboxed(frame_w: u32, frame_h: u32, win_w: u32, win_h: u32) -> [Vertex; 6] {
    let frame_aspect = frame_w as f32 / frame_h.max(1) as f32;
    let win_aspect = win_w.max(1) as f32 / win_h.max(1) as f32;
    let (w, h) = if win_aspect > frame_aspect {
        (frame_aspect / win_aspect, 1.0)
    } else {
        (1.0, win_aspect / frame_aspect)
    };
    quad(-w, -h, w, h)
}

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

const STREAM_SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@group(0) @binding(0) var raw_tex: texture_2d<f32>;

fn yuv_to_rgb(y: f32, u: f32, v: f32) -> vec3<f32> {
    let uu = u - 0.5;
    let vv = v - 0.5;
    return vec3<f32>(
        y + 1.402 * vv,
        y - 0.344136 * uu - 0.714136 * vv,
        y + 1.772 * uu
    );
}

@fragment
fn fs_yuyv(in: VsOut) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(raw_tex);
    let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
    let max_xy = dims - vec2<i32>(1, 1);
    let uv_px = vec2<i32>(in.uv * vec2<f32>(f32(dims.x), f32(dims.y)));
    let p = clamp(uv_px, vec2<i32>(0, 0), max_xy);
    let even_x = p.x & ~1;
    let y0u = textureLoad(raw_tex, vec2<i32>(even_x, p.y), 0).rg;
    let y1v = textureLoad(raw_tex, vec2<i32>(min(even_x + 1, max_xy.x), p.y), 0).rg;
    let y = select(y0u.r, y1v.r, (p.x & 1) == 1);
    let rgb = yuv_to_rgb(y, y0u.g, y1v.g);
    return vec4<f32>(rgb, 1.0);
}

@fragment
fn fs_bgra(in: VsOut) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(raw_tex);
    let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
    let max_xy = dims - vec2<i32>(1, 1);
    let uv_px = vec2<i32>(in.uv * vec2<f32>(f32(dims.x), f32(dims.y)));
    let p = clamp(uv_px, vec2<i32>(0, 0), max_xy);
    return textureLoad(raw_tex, p, 0);
}

@fragment
fn fs_grey(in: VsOut) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(raw_tex);
    let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
    let max_xy = dims - vec2<i32>(1, 1);
    let uv_px = vec2<i32>(in.uv * vec2<f32>(f32(dims.x), f32(dims.y)));
    let p = clamp(uv_px, vec2<i32>(0, 0), max_xy);
    let g = textureLoad(raw_tex, p, 0).r;
    return vec4<f32>(g, g, g, 1.0);
}
"#;

impl Config {
    fn parse() -> Self {
        let mut cfg = Self {
            camera: None,
            sensor: Sensor::Rgb,
            format: None,
            decoder: DecoderKind::TurboJpeg,
            size: None,
            fps: None,
            buffers: 4,
            drain_latest: false,
            list_cameras: false,
            list_camera_modes: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if let Some(camera) = arg.strip_prefix("--camera=") {
                cfg.camera = Some(camera.to_string());
                continue;
            }
            if let Some(sensor) = arg.strip_prefix("--sensor=") {
                cfg.sensor = parse_sensor(sensor);
                continue;
            }
            if let Some(format) = arg.strip_prefix("--format=") {
                cfg.format = Some(parse_format(format));
                continue;
            }
            if let Some(decoder) = arg.strip_prefix("--decoder=") {
                cfg.decoder = parse_decoder(decoder);
                continue;
            }
            if let Some(fps) = arg.strip_prefix("--fps=") {
                cfg.fps = Some(parse_fps(fps));
                continue;
            }
            if let Some(size) = arg.strip_prefix("--size=") {
                cfg.size = Some(parse_size(size));
                continue;
            }
            if let Some(buffers) = arg.strip_prefix("--buffers=") {
                cfg.buffers = parse_buffers(buffers);
                continue;
            }
            match arg.as_str() {
                "--camera" => {
                    let Some(camera) = args.next() else {
                        eprintln!("--camera requires a name, e.g. --camera Lenovo");
                        std::process::exit(2);
                    };
                    cfg.camera = Some(camera);
                }
                "--sensor" => {
                    let Some(sensor) = args.next() else {
                        eprintln!("--sensor requires rgb or ir");
                        std::process::exit(2);
                    };
                    cfg.sensor = parse_sensor(&sensor);
                }
                "--format" => {
                    let Some(format) = args.next() else {
                        eprintln!("--format requires mjpg, yuyv, or grey");
                        std::process::exit(2);
                    };
                    cfg.format = Some(parse_format(&format));
                }
                "--decoder" => {
                    let Some(decoder) = args.next() else {
                        eprintln!("--decoder requires opencv or turbojpeg");
                        std::process::exit(2);
                    };
                    cfg.decoder = parse_decoder(&decoder);
                }
                "--fps" => {
                    let Some(fps) = args.next() else {
                        eprintln!("--fps requires an integer, e.g. --fps 30");
                        std::process::exit(2);
                    };
                    cfg.fps = Some(parse_fps(&fps));
                }
                "--size" => {
                    let Some(size) = args.next() else {
                        eprintln!("--size requires WIDTHxHEIGHT, e.g. --size 640x480");
                        std::process::exit(2);
                    };
                    cfg.size = Some(parse_size(&size));
                }
                "--buffers" => {
                    let Some(buffers) = args.next() else {
                        eprintln!("--buffers requires an integer, e.g. --buffers 1");
                        std::process::exit(2);
                    };
                    cfg.buffers = parse_buffers(&buffers);
                }
                "--drain-latest" => cfg.drain_latest = true,
                "--list-cameras" => cfg.list_cameras = true,
                "--list-camera-modes" => {
                    cfg.list_cameras = true;
                    cfg.list_camera_modes = true;
                }
                "-h" | "--help" => print_help_and_exit(),
                _ => eprintln!("unknown arg {arg:?}; use --help for options"),
            }
        }
        cfg
    }
}

fn parse_sensor(value: &str) -> Sensor {
    match value {
        "rgb" => Sensor::Rgb,
        "ir" => Sensor::Ir,
        _ => {
            eprintln!("invalid --sensor {value:?}; expected rgb or ir");
            std::process::exit(2);
        }
    }
}

fn parse_format(value: &str) -> StreamFormat {
    match value.to_ascii_lowercase().as_str() {
        "mjpg" | "mjpeg" => StreamFormat::Mjpg,
        "yuyv" | "yuyv422" => StreamFormat::Yuyv,
        "grey" | "gray" => StreamFormat::Grey,
        _ => {
            eprintln!("invalid --format {value:?}; expected mjpg, yuyv, or grey");
            std::process::exit(2);
        }
    }
}

fn parse_decoder(value: &str) -> DecoderKind {
    match value.to_ascii_lowercase().as_str() {
        "opencv" | "cv" => DecoderKind::OpenCv,
        "turbojpeg" | "turbo" | "tj" => DecoderKind::TurboJpeg,
        _ => {
            eprintln!("invalid --decoder {value:?}; expected opencv or turbojpeg");
            std::process::exit(2);
        }
    }
}

fn parse_fps(value: &str) -> u32 {
    match value.parse::<u32>() {
        Ok(fps) if fps > 0 => fps,
        _ => {
            eprintln!("invalid --fps {value:?}; expected a positive integer");
            std::process::exit(2);
        }
    }
}

fn parse_size(value: &str) -> (u32, u32) {
    let Some((width, height)) = value.split_once('x') else {
        eprintln!("invalid --size {value:?}; expected WIDTHxHEIGHT");
        std::process::exit(2);
    };
    let width = parse_dimension(width, "--size width");
    let height = parse_dimension(height, "--size height");
    (width, height)
}

fn parse_buffers(value: &str) -> u32 {
    match value.parse::<u32>() {
        Ok(buffers) if buffers > 0 => buffers,
        _ => {
            eprintln!("invalid --buffers {value:?}; expected a positive integer");
            std::process::exit(2);
        }
    }
}

fn parse_dimension(value: &str, label: &str) -> u32 {
    match value.parse::<u32>() {
        Ok(v) if v > 0 => v,
        _ => {
            eprintln!("invalid {label} {value:?}; expected a positive integer");
            std::process::exit(2);
        }
    }
}

fn with_overrides(
    mut stream: camera::StreamConfig,
    size: Option<(u32, u32)>,
    fps: Option<u32>,
) -> camera::StreamConfig {
    if let Some((width, height)) = size {
        stream.width = width;
        stream.height = height;
    }
    if let Some(fps) = fps {
        stream.fps = Some(fps);
    }
    stream
}

fn print_help_and_exit() -> ! {
    println!(
        "Usage: tron-stream [OPTIONS]\n\n\
         Options:\n\
           --camera NAME              Select a camera set by card/bus name (e.g. Lenovo, NexiGo)\n\
           --sensor rgb|ir            Select one stream to display (default: rgb)\n\
           --format mjpg|yuyv|grey    Select capture format (default: mjpg for RGB, grey for IR)\n\
           --decoder opencv|turbojpeg Select MJPG decoder (default: turbojpeg)\n\
           --size WIDTHxHEIGHT        Override selected stream size\n\
           --fps FPS                  Request a frame rate (default: selected mode target)\n\
           --buffers N                Request V4L mmap buffer count (default: 4)\n\
           --drain-latest             Drain already-ready V4L frames before publishing\n\
           --list-cameras             List visible V4L camera capture nodes and exit\n\
           --list-camera-modes        List selected camera sets plus advertised modes and exit\n\
           -h, --help                 Show this help"
    );
    std::process::exit(0);
}
