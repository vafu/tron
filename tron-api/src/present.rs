use anyhow::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoContext;

pub trait Presenter<V> {
    fn present(&mut self, view: V) -> Result<()>;
}
