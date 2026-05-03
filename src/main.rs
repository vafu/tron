// Many items wired for phase B (real ML model, IR ROI) are unused while the
// MockLandmarker is in place. Re-enable lints after phase B lands.
#![allow(dead_code)]

mod calib;
mod camera;
mod filter;
mod gestures;
mod gfx;
mod inference;
mod landmarker;
mod pipeline;
mod proximity;
mod refiners;
mod roi;
mod types;

use anyhow::Result;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const RGB_W: u32 = 640;
const RGB_H: u32 = 480;
const IR_W: u32 = 640;
const IR_H: u32 = 360;

struct App {
    runtime: RuntimeConfig,
    rgb_src: camera::SharedImage,
    ir_src: camera::SharedImage,
    prox_src: proximity::SharedProx,
    controls: types::SharedPipelineControls,
    hand_src: pipeline::SharedHand,
    mask_src: pipeline::SharedMask,
    pointer_src: pipeline::SharedPointer,
    window: Option<Arc<Window>>,
    gfx: Option<gfx::Gfx>,
    loop_timing: LoopTiming,
}

#[derive(Clone)]
struct RuntimeConfig {
    cube: bool,
    skeleton: bool,
    classifier_debug: bool,
    perfetto_path: Option<PathBuf>,
    perfetto_open: bool,
}

#[derive(Default)]
struct LoopTiming {
    last_log: Option<Instant>,
    waits: u32,
    redraws: u32,
}

impl RuntimeConfig {
    fn parse() -> Self {
        let mut cfg = Self {
            cube: true,
            skeleton: true,
            classifier_debug: true,
            perfetto_path: None,
            perfetto_open: false,
        };
        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            if let Some(path) = arg.strip_prefix("--perfetto=") {
                cfg.perfetto_path = Some(PathBuf::from(path));
                continue;
            }
            match arg.as_str() {
                "--no-cube" => cfg.cube = false,
                "--cube" => cfg.cube = true,
                "--no-skeleton" => cfg.skeleton = false,
                "--skeleton" => cfg.skeleton = true,
                "--no-classifier-debug" => cfg.classifier_debug = false,
                "--classifier-debug" => cfg.classifier_debug = true,
                "--classifier-only" => {
                    cfg.cube = false;
                    cfg.skeleton = true;
                    cfg.classifier_debug = true;
                }
                "--perfetto" => {
                    let Some(path) = args.next() else {
                        eprintln!("--perfetto requires a path");
                        std::process::exit(2);
                    };
                    cfg.perfetto_path = Some(PathBuf::from(path));
                }
                "--perfetto-open" => cfg.perfetto_open = true,
                "--help" | "-h" => {
                    print_help_and_exit();
                }
                _ => eprintln!("unknown arg {arg:?}; use --help for options"),
            }
        }
        cfg
    }
}

fn print_help_and_exit() -> ! {
    println!(
        "Usage: tron [OPTIONS]\n\n\
         Options:\n\
           --classifier-only          Disable cube; keep skeleton + classifier diagnostics\n\
           --no-cube / --cube         Disable or enable cube simulation/rendering\n\
           --no-skeleton / --skeleton Disable or enable hand skeleton overlay\n\
           --no-classifier-debug      Hide classifier feature values in the window title\n\
           --perfetto PATH            Write tracing spans/events to a Perfetto .pftrace file\n\
           --perfetto-open            Open the Perfetto trace in the browser after shutdown\n\
           -h, --help                 Show this help"
    );
    std::process::exit(0);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("tron")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 600));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let g = pollster::block_on(gfx::Gfx::new(
            window.clone(),
            self.rgb_src.clone(),
            self.ir_src.clone(),
            self.prox_src.clone(),
            self.controls.clone(),
            self.hand_src.clone(),
            self.mask_src.clone(),
            self.pointer_src.clone(),
            gfx::RenderOptions {
                cube: self.runtime.cube,
                skeleton: self.runtime.skeleton,
                classifier_debug: self.runtime.classifier_debug,
            },
            (RGB_W, RGB_H),
            (IR_W, IR_H),
        ))
        .expect("init gfx");
        self.window = Some(window);
        self.gfx = Some(g);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat,
                        ..
                    },
                ..
            } => {
                handle_key(code, repeat, &self.controls);
            }
            WindowEvent::RedrawRequested => {
                self.loop_timing.redraws += 1;
                if let Some(g) = self.gfx.as_mut() {
                    match g.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            g.resize(g.size);
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
        let log_start = self
            .loop_timing
            .last_log
            .get_or_insert_with(Instant::now)
            .to_owned();
        self.loop_timing.waits += 1;
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        if log_start.elapsed() >= Duration::from_secs(2) {
            let elapsed = log_start.elapsed().as_secs_f32();
            tracing::debug!(
                target: "tron::event_loop",
                waits_per_s = self.loop_timing.waits as f32 / elapsed,
                redraws_per_s = self.loop_timing.redraws as f32 / elapsed,
                "event loop timing"
            );
            self.loop_timing = LoopTiming {
                last_log: Some(Instant::now()),
                ..Default::default()
            };
        }
    }
}

/// Defaults to `info`. Set `RUST_LOG=tron=debug` to include app-level
/// profiling spans in Perfetto output.
fn init_tracing(perfetto_path: Option<PathBuf>) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,tron=info"));

    let fmt_layer = fmt::layer().with_target(true).compact();

    match perfetto_path {
        Some(path) => {
            let file = std::fs::File::create(&path)?;
            let perfetto_layer = tracing_perfetto::PerfettoLayer::new(std::sync::Mutex::new(file));
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(perfetto_layer)
                .init();
            eprintln!("perfetto: writing trace to {}", path.display());
        }
        None => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
        }
    }
    Ok(())
}

fn handle_key(code: KeyCode, repeat: bool, controls: &types::PipelineControls) {
    if repeat {
        return;
    }
    if code == KeyCode::KeyI {
        let enabled = controls.toggle_ir_mask();
        eprintln!("ir mask: {}", if enabled { "on" } else { "off" });
        return;
    }

    const NUDGE_OFFSET: f32 = 0.005;
    const NUDGE_SCALE: f32 = 0.01;
    match code {
        KeyCode::ArrowLeft => calib::modify(|c| c.offset_x -= NUDGE_OFFSET),
        KeyCode::ArrowRight => calib::modify(|c| c.offset_x += NUDGE_OFFSET),
        KeyCode::ArrowUp => calib::modify(|c| c.offset_y -= NUDGE_OFFSET),
        KeyCode::ArrowDown => calib::modify(|c| c.offset_y += NUDGE_OFFSET),
        KeyCode::KeyA => calib::modify(|c| c.scale_x -= NUDGE_SCALE),
        KeyCode::KeyD => calib::modify(|c| c.scale_x += NUDGE_SCALE),
        KeyCode::KeyW => calib::modify(|c| c.scale_y -= NUDGE_SCALE),
        KeyCode::KeyS => calib::modify(|c| c.scale_y += NUDGE_SCALE),
        KeyCode::KeyB => calib::modify(|c| c.use_binary = !c.use_binary),
        KeyCode::KeyR => calib::reset(),
        KeyCode::KeyP => eprintln!("calib: {:?}", calib::current()),
        _ => {}
    }
}

fn maybe_run_perfetto_helper() -> Result<bool> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("--open-perfetto-trace") {
        return Ok(false);
    }
    let Some(path) = args.next() else {
        anyhow::bail!("--open-perfetto-trace requires a path");
    };
    open_perfetto_trace(PathBuf::from(path))?;
    Ok(true)
}

fn spawn_perfetto_helper(path: PathBuf) -> Result<()> {
    let exe = std::env::current_exe()?;
    Command::new(exe)
        .arg("--open-perfetto-trace")
        .arg(path)
        .spawn()?;
    Ok(())
}

fn open_perfetto_trace(path: PathBuf) -> Result<()> {
    const PORT: u16 = 9001;
    const TRACE_ROUTE: &str = "/trace.pftrace";
    let path = path.canonicalize()?;
    let listener = TcpListener::bind(("127.0.0.1", PORT))?;
    let url = format!(
        "https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{PORT}{TRACE_ROUTE}&referrer=tron"
    );
    eprintln!("perfetto: serving {} on {}", path.display(), TRACE_ROUTE);
    eprintln!("perfetto: opening {url}");
    open_browser(&url)?;

    for stream in listener.incoming() {
        let mut stream = stream?;
        if serve_trace_request(&mut stream, &path, TRACE_ROUTE)? {
            break;
        }
    }
    Ok(())
}

fn serve_trace_request(
    stream: &mut TcpStream,
    path: &std::path::Path,
    route: &str,
) -> Result<bool> {
    let mut buf = [0u8; 2048];
    let n = stream.read(&mut buf)?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let Some(first_line) = request.lines().next() else {
        return Ok(false);
    };
    let expected = format!("GET {route} ");
    if !first_line.starts_with(&expected) {
        write_http_response(stream, 404, "text/plain", b"not found")?;
        return Ok(false);
    }

    let body = std::fs::read(path)?;
    write_http_response(stream, 200, "application/octet-stream", &body)?;
    Ok(true)
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {status_text}\r\n\
         Access-Control-Allow-Origin: https://ui.perfetto.dev\r\n\
         Cache-Control: no-cache\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(url).spawn()?;
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn()?;
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd").args(["/C", "start", "", url]).spawn()?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        eprintln!("perfetto: open this URL manually: {url}");
        Ok(())
    }
}

fn main() -> Result<()> {
    if maybe_run_perfetto_helper()? {
        return Ok(());
    }
    let runtime = RuntimeConfig::parse();
    if runtime.perfetto_open && runtime.perfetto_path.is_none() {
        anyhow::bail!("--perfetto-open requires --perfetto PATH");
    }
    let perfetto_path = runtime.perfetto_path.clone();
    let perfetto_open = runtime.perfetto_open;
    init_tracing(runtime.perfetto_path.clone())?;
    let rgb_src = camera::spawn_rgb("/dev/video0", RGB_W, RGB_H)?;
    let ir_src = camera::spawn_ir("/dev/video2", IR_W, IR_H)?;
    let prox_src = proximity::spawn("prox", "proximity1")?;
    let controls = types::PipelineControls::new();

    // Try the real MediaPipe model; fall back to mock if the file is missing
    // or the load fails. Run `scripts/download_models.sh` to fetch it.
    let lm: Box<dyn landmarker::HandLandmarker> =
        match landmarker::mediapipe::MediaPipeHandLandmarker::new(
            "models/hand_landmark/hand_landmark.onnx",
        ) {
            Ok(m) => {
                eprintln!("landmarker: MediaPipe (ort)");
                Box::new(m)
            }
            Err(e) => {
                eprintln!("landmarker: falling back to mock — {e:#}");
                Box::new(landmarker::mock::MockLandmarker::new())
            }
        };

    // Indoor experiment: keep RGB as the primary tracking image, but use the
    // IR foreground signal to dim non-hand background before ROI + landmark
    // stages. Raw RGB/IR frames are still captured into FrameContext first.
    let refiners: Vec<Box<dyn refiners::FrameContextRefiner>> = vec![
        Box::new(refiners::FlashlightDetectorRefiner::new()),
        Box::new(refiners::TemporalSubtractionRefiner::new()),
        Box::new(refiners::RgbMaskingRefiner::new()),
    ];

    let detector =
        roi::detector::PalmDetector::new("models/hand_detector/model.onnx").expect("load detector");

    // ROI chain: 1. Neural Palm Detector on RGB, 2. Previous-frame track.
    let pipe = pipeline::GesturePipeline::new(
        refiners,
        Box::new(roi::CompositeRoiHinter::new(vec![
            Box::new(detector),
            Box::new(roi::track::TrackFromLastRoi::new()),
        ])),
        lm,
        Box::new(filter::OneEuroFilter::default()),
        Box::new(gestures::RuleBasedClassifier::new()),
    );
    let pipeline::PipelineOutputs {
        hand: hand_src,
        mask: mask_src,
        pointer: pointer_src,
    } = pipeline::spawn(
        rgb_src.clone(),
        ir_src.clone(),
        prox_src.clone(),
        controls.clone(),
        pipe,
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        runtime,
        rgb_src,
        ir_src,
        prox_src,
        controls,
        hand_src,
        mask_src,
        pointer_src,
        window: None,
        gfx: None,
        loop_timing: LoopTiming::default(),
    };
    event_loop.run_app(&mut app)?;
    if perfetto_open {
        if let Some(path) = perfetto_path {
            spawn_perfetto_helper(path)?;
            std::process::exit(0);
        }
    }
    Ok(())
}
