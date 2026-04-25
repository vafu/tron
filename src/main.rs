// Many items wired for phase B (real ML model, IR ROI) are unused while the
// MockLandmarker is in place. Re-enable lints after phase B lands.
#![allow(dead_code)]

mod calib;
mod camera;
mod filter;
mod gestures;
mod gfx;
mod landmarker;
mod pipeline;
mod proximity;
mod roi;
mod skeleton_render;
mod types;

use anyhow::Result;
use std::sync::Arc;
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
    rgb_src: camera::SharedImage,
    ir_src: camera::SharedImage,
    prox_src: proximity::SharedProx,
    hand_src: pipeline::SharedHand,
    window: Option<Arc<Window>>,
    gfx: Option<gfx::Gfx>,
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
        let g = gfx::Gfx::new(
            window.clone(),
            self.rgb_src.clone(),
            self.ir_src.clone(),
            self.prox_src.clone(),
            self.hand_src.clone(),
            (RGB_W, RGB_H),
            (IR_W, IR_H),
        )
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
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, repeat, .. },
                ..
            } => {
                handle_calib_key(code, repeat);
            }
            WindowEvent::RedrawRequested => {
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
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn handle_calib_key(code: KeyCode, _repeat: bool) {
    const NUDGE_OFFSET: f32 = 0.005;
    const NUDGE_SCALE: f32 = 0.01;
    match code {
        KeyCode::ArrowLeft  => calib::modify(|c| c.offset_x -= NUDGE_OFFSET),
        KeyCode::ArrowRight => calib::modify(|c| c.offset_x += NUDGE_OFFSET),
        KeyCode::ArrowUp    => calib::modify(|c| c.offset_y -= NUDGE_OFFSET),
        KeyCode::ArrowDown  => calib::modify(|c| c.offset_y += NUDGE_OFFSET),
        KeyCode::KeyA       => calib::modify(|c| c.scale_x  -= NUDGE_SCALE),
        KeyCode::KeyD       => calib::modify(|c| c.scale_x  += NUDGE_SCALE),
        KeyCode::KeyW       => calib::modify(|c| c.scale_y  -= NUDGE_SCALE),
        KeyCode::KeyS       => calib::modify(|c| c.scale_y  += NUDGE_SCALE),
        KeyCode::KeyR       => calib::reset(),
        KeyCode::KeyP       => eprintln!("calib: {:?}", calib::current()),
        _ => {}
    }
}

fn main() -> Result<()> {
    let rgb_src = camera::spawn_rgb("/dev/video0", RGB_W, RGB_H)?;
    let ir_src = camera::spawn_ir("/dev/video2", IR_W, IR_H)?;
    let prox_src = proximity::spawn("prox", "proximity1")?;

    // Try the real MediaPipe model; fall back to mock if the file is missing
    // or the load fails. Run `scripts/download_models.sh` to fetch it.
    let lm: Box<dyn landmarker::HandLandmarker> = match landmarker::mediapipe::MediaPipeHandLandmarker::new("models/hand_landmark.onnx") {
        Ok(m) => {
            eprintln!("landmarker: MediaPipe (ort)");
            Box::new(m)
        }
        Err(e) => {
            eprintln!("landmarker: falling back to mock — {e:#}");
            Box::new(landmarker::mock::MockLandmarker::new())
        }
    };

    // ROI chain: IR blob (mapped to RGB coords via calib::IR_TO_RGB) → previous-
    // frame tracking → full frame fallback.
    let pipe = pipeline::GesturePipeline {
        roi: Box::new(roi::CompositeRoiHinter(vec![
            Box::new(roi::ir::IrRoiHinter::new()),
            Box::new(roi::track::TrackFromLastRoi::new()),
            Box::new(roi::FullFrameRoi),
        ])),
        lm,
        filter: Box::new(filter::OneEuroFilter::default()),
        gestures: Box::new(gestures::RuleBasedClassifier::new()),
    };
    let hand_src = pipeline::spawn(rgb_src.clone(), ir_src.clone(), prox_src.clone(), pipe);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        rgb_src,
        ir_src,
        prox_src,
        hand_src,
        window: None,
        gfx: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
