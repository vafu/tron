use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tron_api::{EventProducerChannels, PointerInput, PointerOutput, Sink};

use crate::pipeline::{ControllerFrame, Tick};

pub type ComboSink = tron_core::sink::ComboSink<dyn for<'a> Sink<&'a ControllerFrame<'a>>>;
pub type PointerSink = tron_core::sink::ComboSink<dyn Sink<PointerOutput>>;

pub fn run<T>(
    ticker: T,
    pointer: EventProducerChannels<PointerInput, PointerOutput>,
    sinks: ComboSink,
    pointer_sinks: PointerSink,
) -> Result<()>
where
    T: Tick,
{
    let mut runtime = ControllerRuntime::new(ticker, pointer, sinks, pointer_sinks);
    loop {
        runtime.drain_pointer_output(None)?;
        if !runtime.process_next_frame(None)? {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

pub struct ControllerRuntime<T> {
    ticker: T,
    pointer_input: mpsc::Sender<PointerInput>,
    pointer_output: mpsc::Receiver<PointerOutput>,
    _pointer_task: JoinHandle<Result<()>>,
    sinks: ComboSink,
    pointer_sinks: PointerSink,
}

impl<T> ControllerRuntime<T> {
    pub fn new(
        ticker: T,
        pointer: EventProducerChannels<PointerInput, PointerOutput>,
        sinks: ComboSink,
        pointer_sinks: PointerSink,
    ) -> Self {
        Self {
            ticker,
            pointer_input: pointer.input,
            pointer_output: pointer.output,
            _pointer_task: pointer.task,
            sinks,
            pointer_sinks,
        }
    }

    pub fn drain_pointer_output(
        &mut self,
        mut preview_sink: Option<&mut dyn Sink<PointerOutput>>,
    ) -> Result<bool> {
        let mut drained = false;
        while let Ok(output) = self.pointer_output.try_recv() {
            drained = true;
            if let Some(sink) = preview_sink.as_mut() {
                pollster::block_on(sink.consume(output))?;
            }
            pollster::block_on(self.pointer_sinks.consume(output))?;
        }
        Ok(drained)
    }

    pub fn process_next_frame(
        &mut self,
        preview_sink: Option<&mut dyn for<'a> Sink<&'a ControllerFrame<'a>>>,
    ) -> Result<bool>
    where
        T: Tick,
    {
        let Some(frame) = self.ticker.tick()? else {
            return Ok(false);
        };

        if let Err(err) = self.pointer_input.try_send(PointerInput {
            gesture: frame.gesture.clone(),
        }) {
            tracing::debug!("controller pointer input dropped: {err}");
        }
        if let Some(sink) = preview_sink {
            pollster::block_on(sink.consume(&frame))?;
        }
        pollster::block_on(self.sinks.consume(&frame))?;

        Ok(true)
    }

    pub fn next_frame(&mut self) -> Result<bool>
    where
        T: Tick,
    {
        self.ticker.next_frame()
    }

    pub fn prev_frame(&mut self) -> Result<bool>
    where
        T: Tick,
    {
        self.ticker.prev_frame()
    }
}
