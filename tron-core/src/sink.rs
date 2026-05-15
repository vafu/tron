use anyhow::Result;
use tron_api::Sink;

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
}

#[async_trait::async_trait(?Send)]
impl<V, S> Sink<V> for ComboSink<S>
where
    V: Clone,
    S: Sink<V> + ?Sized,
{
    async fn consume<'a>(&'a mut self, view: V) -> Result<()>
    where
        V: 'a,
    {
        for sink in &mut self.sinks {
            sink.consume(view.clone()).await?;
        }
        Ok(())
    }
}
