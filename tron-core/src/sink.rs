use tron_api::Size;

use crate::render::http::HttpJsonSink;

pub trait ControlledSink {
    fn resize(&mut self, _size: Size) {}

    fn toggle_enabled(&mut self) -> Option<bool> {
        None
    }
}

impl ControlledSink for HttpJsonSink {}

pub struct ComboSink<S: ?Sized> {
    sinks: Vec<Box<S>>,
}

impl<S: ?Sized> Default for ComboSink<S> {
    fn default() -> Self {
        Self { sinks: Vec::new() }
    }
}

impl<S: ?Sized> ComboSink<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_box(&mut self, sink: Box<S>) {
        self.sinks.push(sink);
    }

    pub fn push_front_box(&mut self, sink: Box<S>) {
        self.sinks.insert(0, sink);
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<S>> {
        self.sinks.iter_mut()
    }
}

impl<S> ComboSink<S>
where
    S: ControlledSink + ?Sized,
{
    pub fn resize(&mut self, size: Size) {
        for sink in &mut self.sinks {
            sink.resize(size);
        }
    }

    pub fn toggle_enabled(&mut self) -> Option<bool> {
        let mut latest = None;
        for sink in &mut self.sinks {
            latest = sink.toggle_enabled().or(latest);
        }
        latest
    }
}
