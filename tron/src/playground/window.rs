use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;
use tron_api::{FrameStats, FrameViewModel, NamedFrame, PixelFormat, Presenter};
use tron_core::pipeline::FrameStream;
use tron_core::present::wgpu::WgpuPresenter;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

pub fn run(stream: impl FrameStream + 'static) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(stream);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<S> {
    stream: S,
    window_id: Option<WindowId>,
    presenter: Option<WgpuPresenter>,
    result: Result<()>,
}

impl<S> WindowApp<S> {
    fn new(stream: S) -> Self {
        Self {
            stream,
            window_id: None,
            presenter: None,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl<S> ApplicationHandler for WindowApp<S>
where
    S: FrameStream,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.presenter.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron playground");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        match pollster::block_on(WgpuPresenter::new(window)) {
            Ok(presenter) => self.presenter = Some(presenter),
            Err(err) => self.set_error(event_loop, err),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }
        let Some(presenter) = self.presenter.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => presenter.resize(size),
            WindowEvent::RedrawRequested => {
                if let Err(err) = present_next_frame(&mut self.stream, presenter) {
                    self.set_error(event_loop, err);
                    return;
                }
                presenter.window().request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(presenter) = self.presenter.as_ref() {
            presenter.window().request_redraw();
        }
    }
}

fn present_next_frame(stream: &mut impl FrameStream, presenter: &mut WgpuPresenter) -> Result<()> {
    let start = Instant::now();
    let frame = stream.next_frame()?;
    if frame.format != PixelFormat::Bgra8 {
        anyhow::bail!(
            "window presenter expects BGRA8 frames; set --format mjpg --decode-format bgra8"
        );
    }
    let frames = [NamedFrame {
        name: "camera",
        frame,
    }];
    presenter.present(FrameViewModel {
        frames: &frames,
        metadata: FrameStats {
            acquire_us: start.elapsed().as_micros() as u64,
        },
    })
}
