use std::sync::Arc;

use anyhow::{Context, Result};
use tron_api::{Sink, Size};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::pipeline::Tick;
use crate::renderer::Renderer;
use crate::sink::ComboSink;

pub fn run<T>(ticker: T, sinks: ComboSink) -> Result<()>
where
    T: Tick,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(ticker, sinks);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<T> {
    ticker: T,
    sinks: ComboSink,
    sinks_ready: bool,
    rendered_pair_id: Option<(u64, u64)>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    result: Result<()>,
}

impl<T> WindowApp<T> {
    fn new(ticker: T, sinks: ComboSink) -> Self {
        Self {
            ticker,
            sinks,
            sinks_ready: false,
            rendered_pair_id: None,
            window_id: None,
            window: None,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl<T> ApplicationHandler for WindowApp<T>
where
    T: Tick,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.sinks_ready {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron collector");
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
                self.sinks.push_front(renderer);
                self.sinks_ready = true;
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
                if self.sinks_ready {
                    self.sinks.resize(Size {
                        width: size.width,
                        height: size.height,
                    });
                    self.rendered_pair_id = None;
                }
            }
            WindowEvent::RedrawRequested => {
                let aggregate = match self.ticker.tick() {
                    Ok(aggregate) => aggregate,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let Some(aggregate) = aggregate else {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                };

                let pair_id = aggregate.pair_id();
                if self.rendered_pair_id == Some(pair_id) {
                    return;
                }

                if !self.sinks_ready {
                    return;
                }
                if let Err(err) = pollster::block_on(self.sinks.consume(&aggregate)) {
                    self.set_error(event_loop, err);
                    return;
                }

                self.rendered_pair_id = Some(pair_id);
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
