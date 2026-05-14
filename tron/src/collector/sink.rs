use anyhow::Result;
use tron_api::{Sink, Size};

use crate::aggregate::Aggregate;
use crate::persistence::Persistence;
use crate::renderer::Renderer;

#[async_trait::async_trait(?Send)]
pub trait AggregateSink {
    async fn consume(&mut self, aggregate: &Aggregate<'_>) -> Result<()>;

    fn resize(&mut self, _size: Size) {}
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
