use anyhow::Result;

use crate::stream::process::FrameContext;

pub trait RenderSink {
    fn submit(&mut self, context: &FrameContext<'_>) -> Result<()>;
}
