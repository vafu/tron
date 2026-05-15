use anyhow::Result;
use tron_api::Sink;

use crate::aggregate::Aggregate;

pub type ComboSink = tron_core::sink::ComboSink<dyn for<'a> Sink<&'a Aggregate<'a>>>;

pub struct ToggleSink<S> {
    inner: S,
    enabled: bool,
}

impl<S> ToggleSink<S> {
    pub fn new(inner: S, enabled: bool) -> Self {
        Self { inner, enabled }
    }

    pub fn toggle_enabled(&mut self) -> bool {
        self.enabled = !self.enabled;
        self.enabled
    }
}

#[async_trait::async_trait(?Send)]
impl<'view, S> Sink<&'view Aggregate<'view>> for ToggleSink<S>
where
    S: Sink<&'view Aggregate<'view>>,
{
    async fn consume<'a>(&'a mut self, aggregate: &'view Aggregate<'view>) -> Result<()>
    where
        &'view Aggregate<'view>: 'a,
    {
        if self.enabled {
            self.inner.consume(aggregate).await?;
        }
        Ok(())
    }
}
