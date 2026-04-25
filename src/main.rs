// Many items wired for phase B (real ML model, IR ROI) are unused while the
// MockLandmarker is in place. Re-enable lints after phase B lands.
#![allow(dead_code)]

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
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
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

fn main() -> Result<()> {
    let rgb_src = camera::spawn_rgb("/dev/video0", RGB_W, RGB_H)?;
    let ir_src = camera::spawn_ir("/dev/video2", IR_W, IR_H)?;
    let prox_src = proximity::spawn("prox", "proximity1")?;

    // v1 pipeline — MockLandmarker so the app runs without any model file.
    let pipe = pipeline::GesturePipeline {
        roi: Box::new(roi::CompositeRoiHinter(vec![
            Box::new(roi::track::TrackFromLastRoi::new()),
            Box::new(roi::FullFrameRoi),
        ])),
        lm: Box::new(landmarker::mock::MockLandmarker::new()),
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
