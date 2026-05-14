use anyhow::Result;
use tron_api::{Sink, Size};

use crate::aggregate::Aggregate;
use crate::persistence::Persistence;
use crate::renderer::Renderer;

#[async_trait::async_trait(?Send)]
pub trait AggregateSink {
    async fn consume(&mut self, aggregate: &Aggregate<'_>) -> Result<()>;

    fn resize(&mut self, _size: Size) {}

    fn toggle_enabled(&mut self) -> Option<bool> {
        None
    }
}

#[derive(Default)]
pub struct ComboSink {
    sinks: Vec<Box<dyn AggregateSink>>,
}

impl ComboSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<S>(&mut self, sink: S)
    where
        S: AggregateSink + 'static,
    {
        self.sinks.push(Box::new(sink));
    }

    pub fn push_front<S>(&mut self, sink: S)
    where
        S: AggregateSink + 'static,
    {
        self.sinks.insert(0, Box::new(sink));
    }

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

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a Aggregate<'a>> for ComboSink {
    async fn consume(&mut self, aggregate: &'a Aggregate<'a>) -> Result<()> {
        for sink in &mut self.sinks {
            sink.consume(aggregate).await?;
        }
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl AggregateSink for Renderer {
    async fn consume(&mut self, aggregate: &Aggregate<'_>) -> Result<()> {
        <Self as Sink<&Aggregate<'_>>>::consume(self, aggregate).await
    }

    fn resize(&mut self, size: Size) {
        self.resize(size);
    }
}

#[async_trait::async_trait(?Send)]
impl AggregateSink for Persistence {
    async fn consume(&mut self, aggregate: &Aggregate<'_>) -> Result<()> {
        <Self as Sink<&Aggregate<'_>>>::consume(self, aggregate).await
    }
}

pub struct ToggleSink<S> {
    inner: S,
    enabled: bool,
}

impl<S> ToggleSink<S> {
    pub fn new(inner: S, enabled: bool) -> Self {
        Self { inner, enabled }
    }
}

#[async_trait::async_trait(?Send)]
impl<S> AggregateSink for ToggleSink<S>
where
    S: AggregateSink,
{
    async fn consume(&mut self, aggregate: &Aggregate<'_>) -> Result<()> {
        if self.enabled {
            self.inner.consume(aggregate).await?;
        }
        Ok(())
    }

    fn resize(&mut self, size: Size) {
        self.inner.resize(size);
    }

    fn toggle_enabled(&mut self) -> Option<bool> {
        self.enabled = !self.enabled;
        Some(self.enabled)
    }
}
