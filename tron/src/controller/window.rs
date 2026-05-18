use std::sync::Arc;

use anyhow::{Context, Result};
use tron_api::{EventProducerChannels, PointerInput, PointerOutput, Sink, Size};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{WindowAttributes, WindowId};

use crate::pipeline::Tick;
use crate::renderer::Renderer;
use crate::runtime::{ComboSink, ControllerRuntime, PointerSink};

pub fn run<T>(
    ticker: T,
    pointer: EventProducerChannels<PointerInput, PointerOutput>,
    sinks: ComboSink,
    pointer_sinks: PointerSink,
) -> Result<()>
where
    T: Tick,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(ticker, pointer, sinks, pointer_sinks);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<T> {
    runtime: ControllerRuntime<T>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<Renderer>,
    occluded: bool,
    result: Result<()>,
}

impl<T> WindowApp<T> {
    fn new(
        ticker: T,
        pointer: EventProducerChannels<PointerInput, PointerOutput>,
        sinks: ComboSink,
        pointer_sinks: PointerSink,
    ) -> Self {
        Self {
            runtime: ControllerRuntime::new(ticker, pointer, sinks, pointer_sinks),
            window_id: None,
            window: None,
            renderer: None,
            occluded: false,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }

    fn drain_pointer_output(&mut self, event_loop: &ActiveEventLoop) -> Option<bool> {
        match self
            .runtime
            .drain_pointer_output(self.renderer.as_mut().map(|sink| sink as &mut dyn Sink<_>))
        {
            Ok(drained) => Some(drained),
            Err(err) => {
                self.set_error(event_loop, err);
                None
            }
        }
    }

    fn process_next_frame(&mut self, event_loop: &ActiveEventLoop) -> Option<bool>
    where
        T: Tick,
    {
        let preview_sink = if self.occluded {
            None
        } else {
            self.renderer
                .as_mut()
                .map(|sink| sink as &mut dyn for<'a> Sink<&'a crate::pipeline::ControllerFrame<'a>>)
        };
        match self.runtime.process_next_frame(preview_sink) {
            Ok(processed) => Some(processed),
            Err(err) => {
                self.set_error(event_loop, err);
                None
            }
        }
    }

    fn request_redraw_if_visible(&self) {
        if self.occluded {
            return;
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
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
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                let moved = match event.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.runtime.prev_frame(),
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.runtime.next_frame(),
                    _ => return,
                };
                match moved {
                    Ok(true) => {
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                    Ok(false) => {}
                    Err(err) => self.set_error(event_loop, err),
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    let size = Size {
                        width: size.width,
                        height: size.height,
                    };
                    renderer.resize(size);
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    self.request_redraw_if_visible();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.drain_pointer_output(event_loop).is_none() {
                    return;
                }

                if self.occluded {
                    return;
                }
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Err(err) = pollster::block_on(renderer.render_cached()) {
                        self.set_error(event_loop, err);
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(pointer_changed) = self.drain_pointer_output(event_loop) else {
            return;
        };
        let Some(frame_processed) = self.process_next_frame(event_loop) else {
            return;
        };
        if pointer_changed && !frame_processed {
            self.request_redraw_if_visible();
        }
    }
}
