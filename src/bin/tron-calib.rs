#![allow(dead_code)]

#[path = "../calib/mod.rs"]
mod calib;
#[path = "../camera/mod.rs"]
mod camera;
#[path = "../depth/mod.rs"]
mod depth;
#[path = "../diagnostics/mod.rs"]
mod diagnostics;
#[path = "../filter/mod.rs"]
mod filter;
#[path = "../gestures/mod.rs"]
mod gestures;
#[path = "../gfx/mod.rs"]
mod gfx;
#[path = "../inference/mod.rs"]
mod inference;
#[path = "../landmarker/mod.rs"]
mod landmarker;
#[path = "../pipeline/mod.rs"]
mod pipeline;
#[path = "../proximity/mod.rs"]
mod proximity;
#[path = "../refiners/mod.rs"]
mod refiners;
#[path = "../roi/mod.rs"]
mod roi;
#[path = "../types/mod.rs"]
mod types;

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Clone)]
struct Config {
    camera: Option<String>,
    list_cameras: bool,
    list_camera_modes: bool,
    checkerboard: (i32, i32),
    square_size: f32,
}

struct App {
    raw_rgb: camera::SharedImage,
    raw_ir: camera::SharedImage,
    display_rgb: camera::SharedImage,
    display_debug_rgb: camera::SharedImage,
    display_ir: camera::SharedImage,
    prox_src: proximity::SharedProx,
    controls: types::SharedPipelineControls,
    hand_src: pipeline::SharedHand,
    mask_src: pipeline::SharedMask,
    pointer_src: pipeline::SharedPointer,
    rgb_size: (u32, u32),
    ir_size: (u32, u32),
    pattern: (i32, i32),
    square_size: f32,
    stereo: calib::stereo::StereoCalibrationSession,
    window: Option<Arc<Window>>,
    gfx: Option<gfx::Gfx>,
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
    eprintln!(
        "camera: {} rgb={} {}x{} ir={} {}x{}",
        camera_set.label,
        camera_set.rgb.path,
        camera_set.rgb.width,
        camera_set.rgb.height,
        camera_set.ir.path,
        camera_set.ir.width,
        camera_set.ir.height
    );
    let rgb_size = (camera_set.rgb.width, camera_set.rgb.height);
    let ir_size = (camera_set.ir.width, camera_set.ir.height);
    calib::init(&camera_set.label, rgb_size, ir_size);

    let raw_rgb = camera::spawn_config(camera_set.rgb)?;
    let raw_ir = camera::spawn_config(camera_set.ir)?;
    let display_rgb = Arc::new(Mutex::new(None));
    let display_debug_rgb = Arc::new(Mutex::new(None));
    let display_ir = Arc::new(Mutex::new(None));
    spawn_display_mirror(
        raw_rgb.clone(),
        raw_ir.clone(),
        display_rgb.clone(),
        display_ir.clone(),
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    eprintln!("tron-calib: event loop ready");
    eprintln!("tron-calib: starting app");
    let controls = types::PipelineControls::new();
    let mut app = App {
        raw_rgb,
        raw_ir,
        display_rgb,
        display_debug_rgb,
        display_ir,
        prox_src: Arc::new(Mutex::new(None)),
        controls,
        hand_src: Arc::new(Mutex::new(None)),
        mask_src: Arc::new(Mutex::new(None)),
        pointer_src: Arc::new(Mutex::new(None)),
        rgb_size,
        ir_size,
        pattern: cfg.checkerboard,
        square_size: cfg.square_size,
        stereo: calib::stereo::StereoCalibrationSession::new(
            cfg.checkerboard,
            cfg.square_size,
            rgb_size,
            ir_size,
        ),
        window: None,
        gfx: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        eprintln!("tron-calib: creating window");
        let attrs = Window::default_attributes()
            .with_title("tron-calib")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 600));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        eprintln!("tron-calib: initializing renderer");
        let gfx = pollster::block_on(gfx::Gfx::new(
            window.clone(),
            self.display_rgb.clone(),
            self.display_debug_rgb.clone(),
            self.display_ir.clone(),
            self.prox_src.clone(),
            self.controls.clone(),
            self.hand_src.clone(),
            self.mask_src.clone(),
            self.pointer_src.clone(),
            gfx::RenderOptions {
                title: "tron-calib",
                cube: false,
                skeleton: false,
                classifier_debug: false,
            },
            self.rgb_size,
            self.ir_size,
        ))
        .expect("init gfx");
        eprintln!("tron-calib: window ready");
        self.window = Some(window);
        self.gfx = Some(gfx);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.resize(size);
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
                if !repeat {
                    match code {
                        KeyCode::KeyC => self.capture_checkerboard(),
                        KeyCode::KeyV => self.solve_stereo(),
                        KeyCode::KeyR => {
                            self.stereo.reset(
                                self.pattern,
                                self.square_size,
                                self.rgb_size,
                                self.ir_size,
                            );
                            eprintln!("stereo: reset samples");
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gfx) = self.gfx.as_mut() {
                    match gfx.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            gfx.resize(gfx.size);
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
    }
}

impl App {
    fn capture_checkerboard(&mut self) {
        match capture_checkerboard_window(&self.raw_rgb, &self.raw_ir, self.pattern) {
            Ok((rgb, ir, sample, attempts)) => self.accept_sample(rgb, ir, sample, attempts),
            Err(e) => eprintln!("checkerboard: capture failed: {e:#}"),
        }
    }

    fn accept_sample(
        &mut self,
        rgb: types::Image,
        ir: types::Image,
        sample: calib::checkerboard::CheckerboardSample,
        attempts: u32,
    ) {
        match calib::checkerboard::calibrate_affine_from_sample(&sample, &rgb, &ir) {
            Ok(result) => {
                calib::set(result.calib);
                if let Err(e) = calib::save() {
                    eprintln!("checkerboard: affine save failed: {e}");
                }
                eprintln!(
                    "checkerboard: affine {} corners rms={:.5} attempts={attempts}",
                    result.corners, result.rms_error
                );
            }
            Err(e) => eprintln!("checkerboard: affine fit failed: {e:#}"),
        }
        match self.stereo.push(sample) {
            Ok(n) => eprintln!("stereo: captured sample {n}; press V to solve"),
            Err(e) => eprintln!("stereo: sample rejected: {e:#}"),
        }
    }

    fn solve_stereo(&self) {
        eprintln!("stereo: solving from {} samples", self.stereo.len());
        match self.stereo.solve() {
            Ok(result) => {
                eprintln!(
                    "stereo: solved samples={} rgb_rms={:.4} ir_rms={:.4} stereo_rms={:.4} {}",
                    result.sample_count,
                    result.rgb_rms,
                    result.ir_rms,
                    result.stereo_rms,
                    result.error_summary()
                );
                if let Err(e) = calib::save_stereo(&result.to_text()) {
                    eprintln!("stereo: save failed: {e}");
                }
            }
            Err(e) => eprintln!("stereo: solve failed: {e:#}"),
        }
    }
}

fn capture_checkerboard_window(
    raw_rgb: &camera::SharedImage,
    raw_ir: &camera::SharedImage,
    pattern: (i32, i32),
) -> anyhow::Result<(
    types::Image,
    types::Image,
    calib::checkerboard::CheckerboardSample,
    u32,
)> {
    let deadline = Instant::now() + Duration::from_millis(350);
    let mut attempts = 0u32;
    let mut last_pair = None;
    let mut last_error = None;

    while Instant::now() < deadline {
        let rgb = raw_rgb.lock().unwrap().clone();
        let ir = raw_ir.lock().unwrap().clone();
        let (Some(rgb), Some(ir)) = (rgb, ir) else {
            thread::sleep(Duration::from_millis(4));
            continue;
        };
        let pair = (rgb.seq, ir.seq);
        if last_pair == Some(pair) {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        last_pair = Some(pair);
        attempts += 1;
        match calib::checkerboard::capture_sample(&rgb, &ir, pattern) {
            Ok(sample) => return Ok((rgb, ir, sample, attempts)),
            Err(e) => last_error = Some(e),
        }
        thread::sleep(Duration::from_millis(2));
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("no RGB/IR frames available during capture window"))
        .context(format!(
            "no checkerboard pair detected in {attempts} attempts over 350ms"
        )))
}

impl Config {
    fn parse() -> Self {
        let mut cfg = Self {
            camera: None,
            list_cameras: false,
            list_camera_modes: false,
            checkerboard: (9, 6),
            square_size: 1.0,
        };
        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            if let Some(camera) = arg.strip_prefix("--camera=") {
                cfg.camera = Some(camera.to_string());
                continue;
            }
            if let Some(pattern) = arg.strip_prefix("--checkerboard=") {
                cfg.checkerboard = parse_checkerboard(pattern);
                continue;
            }
            if let Some(square) = arg.strip_prefix("--checkerboard-square=") {
                cfg.square_size = parse_positive_f32(square, "--checkerboard-square");
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
                "--checkerboard" => {
                    if let Some(pattern) = args.next_if(|arg| !arg.starts_with('-')) {
                        cfg.checkerboard = parse_checkerboard(&pattern);
                    }
                }
                "--checkerboard-square" => {
                    let Some(square) = args.next() else {
                        eprintln!("--checkerboard-square requires a positive number");
                        std::process::exit(2);
                    };
                    cfg.square_size = parse_positive_f32(&square, "--checkerboard-square");
                }
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

fn spawn_display_mirror(
    raw_rgb: camera::SharedImage,
    raw_ir: camera::SharedImage,
    display_rgb: camera::SharedImage,
    display_ir: camera::SharedImage,
) {
    thread::Builder::new()
        .name("calib-display".into())
        .spawn(move || {
            loop {
                if let Some(rgb) = raw_rgb.lock().unwrap().clone() {
                    *display_rgb.lock().unwrap() = Some(rgb);
                }
                if let Some(ir) = raw_ir.lock().unwrap().clone() {
                    *display_ir.lock().unwrap() = Some(ir);
                }
                thread::sleep(Duration::from_millis(8));
            }
        })
        .expect("spawn calibration display mirror");
}

fn parse_checkerboard(value: &str) -> (i32, i32) {
    let Some((cols, rows)) = value.split_once('x') else {
        eprintln!("--checkerboard requires COLSxROWS inner corners, e.g. 9x6");
        std::process::exit(2);
    };
    let parsed = cols
        .parse::<i32>()
        .ok()
        .zip(rows.parse::<i32>().ok())
        .filter(|(cols, rows)| *cols >= 2 && *rows >= 2);
    match parsed {
        Some(pattern) => pattern,
        None => {
            eprintln!("invalid checkerboard pattern {value:?}; expected COLSxROWS, e.g. 9x6");
            std::process::exit(2);
        }
    }
}

fn parse_positive_f32(value: &str, flag: &str) -> f32 {
    match value.parse::<f32>() {
        Ok(value) if value > 0.0 => value,
        _ => {
            eprintln!("{flag} requires a positive number, got {value:?}");
            std::process::exit(2);
        }
    }
}

fn print_help_and_exit() -> ! {
    println!(
        "Usage: tron-calib [OPTIONS]\n\n\
         Options:\n\
           --camera NAME              Select a camera set by card/bus name (e.g. Lenovo, NexiGo)\n\
           --list-cameras             List visible V4L camera capture nodes and exit\n\
           --list-camera-modes        List selected camera sets plus advertised modes and exit\n\
           --checkerboard [COLSxROWS] Checkerboard inner-corner pattern (default: 9x6)\n\
           --checkerboard-square N    Checkerboard square size for stereo units (default: 1.0)\n\
           -h, --help                 Show this help\n\n\
         Keys:\n\
           C                          Capture checkerboard sample\n\
           V                          Solve stereo calibration from captured samples\n\
           R                          Reset captured samples"
    );
    std::process::exit(0);
}
