use anyhow::Result;
use tron_api::{Frame, FrameSource, OpenedCameraInfo, OwnedFrame, ProjectionMapSource};

use crate::projection::FrameProjectionMap;

use super::frame_projection::project_frame;

pub struct ProjectedFrameSource<S, M> {
    source: S,
    info: OpenedCameraInfo,
    map_source: M,
    current_map: FrameProjectionMap,
    current: Option<OwnedFrame>,
}

impl<S, M> ProjectedFrameSource<S, M>
where
    S: FrameSource + Send,
    M: ProjectionMapSource<Map = FrameProjectionMap> + Send,
{
    pub fn new(source: S, mut map_source: M) -> Result<Self> {
        let current_map = pollster::block_on(map_source.next_map(std::time::Instant::now()))?
            .ok_or_else(|| anyhow::anyhow!("projection map source did not provide initial map"))?;
        let mut info = source.info().clone();
        info.size = current_map.output_size;
        Ok(Self {
            source,
            info,
            map_source,
            current_map,
            current: None,
        })
    }
}

#[async_trait::async_trait]
impl<S, M> FrameSource for ProjectedFrameSource<S, M>
where
    S: FrameSource + Send,
    M: ProjectionMapSource<Map = FrameProjectionMap> + Send,
{
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let Some(frame) = self.source.next_frame().await? else {
            return Ok(None);
        };
        if let Some(map) = self
            .map_source
            .next_map(frame.meta.timestamp.received_at)
            .await?
        {
            self.current_map = map;
        }
        anyhow::ensure!(
            frame.meta.size == self.current_map.input_size,
            "projected source frame size {:?} does not match projection input size {:?}",
            frame.meta.size,
            self.current_map.input_size
        );
        anyhow::ensure!(
            self.current_map.output_size == self.info.size,
            "projection output size {:?} does not match transformed source size {:?}",
            self.current_map.output_size,
            self.info.size
        );
        self.current = Some(project_frame(frame, &self.current_map)?);
        Ok(self.current.as_ref().map(|frame| frame.as_frame()))
    }
}
