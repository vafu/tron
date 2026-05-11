use anyhow::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoContext;

pub trait Renderer<V> {
    fn render(&mut self, view: V) -> Result<()>;
}
