use anyhow::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoContext;

#[async_trait::async_trait(?Send)]
pub trait Sink<V> {
    async fn consume<'a>(&'a mut self, view: V) -> Result<()>
    where
        V: 'a;
}
