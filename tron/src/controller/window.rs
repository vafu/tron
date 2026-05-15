use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tron_api::{EventProducerChannels, PointerInput, PointerOutput, Sink, Size};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::pipeline::{ControllerFrame, Tick};
use crate::renderer::Renderer;

pub type ComboSink = tron_core::sink::ComboSink<dyn for<'a> Sink<&'a ControllerFrame<'a>>>;

pub fn run<T>(
    ticker: T,
    pointer: EventProducerChannels<PointerInput, PointerOutput>,
    sinks: ComboSink,
) -> Result<()>
where
    T: Tick,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(ticker, pointer, sinks);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<T> {
    ticker: T,
    pointer_input: mpsc::Sender<PointerInput>,
    pointer_output: mpsc::Receiver<PointerOutput>,
    _pointer_task: JoinHandle<Result<()>>,
    sinks: ComboSink,
    rendered_frame_id: Option<u64>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<Renderer>,
    result: Result<()>,
}

impl<T> WindowApp<T> {
    fn new(
        ticker: T,
        pointer: EventProducerChannels<PointerInput, PointerOutput>,
        sinks: ComboSink,
    ) -> Self {
        Self {
            ticker,
            pointer_input: pointer.input,
            pointer_output: pointer.output,
            _pointer_task: pointer.task,
            sinks,
            rendered_frame_id: None,
            window_id: None,
            window: None,
            renderer: None,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }

    fn drain_pointer_output(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let Some(renderer) = self.renderer.as_mut() else {
            return true;
        };
        while let Ok(event) = self.pointer_output.try_recv() {
            if let Err(err) = pollster::block_on(renderer.consume(event)) {
                self.set_error(event_loop, err);
                return false;
            }
        }
        true
    }
}

impl<T> ApplicationHandler for WindowApp<T>
where
    T: Tick,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron controller");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(Renderer::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
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

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    let size = Size {
                        width: size.width,
                        height: size.height,
                    };
                    renderer.resize(size);
                    self.rendered_frame_id = None;
                }
            }
            WindowEvent::RedrawRequested => {
                if !self.drain_pointer_output(event_loop) {
                    return;
                }

                let frame = match self.ticker.tick() {
                    Ok(frame) => frame,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let Some(frame) = frame else {
                    if let Some(renderer) = self.renderer.as_mut() {
                        if let Err(err) = pollster::block_on(renderer.render_cached()) {
                            self.set_error(event_loop, err);
                            return;
                        }
                    }
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                };

                let frame_id = frame.frame_id();
                if self.rendered_frame_id == Some(frame_id) {
                    return;
                }

                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(err) = self.pointer_input.try_send(PointerInput {
                    gesture: frame.gesture.clone(),
                }) {
                    tracing::debug!("controller pointer input dropped: {err}");
                }
                while let Ok(event) = self.pointer_output.try_recv() {
                    if let Err(err) = pollster::block_on(renderer.consume(event)) {
                        self.set_error(event_loop, err);
                        return;
                    }
                }
                if let Err(err) = pollster::block_on(renderer.consume(&frame)) {
                    self.set_error(event_loop, err);
                    return;
                }
                if let Err(err) = pollster::block_on(self.sinks.consume(&frame)) {
                    self.set_error(event_loop, err);
                    return;
                }

                self.rendered_frame_id = Some(frame_id);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
