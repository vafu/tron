use anyhow::Result;
use tron_api::{
    Frame, FrameMeta, FrameSource, OpenedCameraInfo, OwnedFrame, PixelFormat, ProjectionMapSource,
};

use crate::projection::FrameProjectionMap;
use crate::view::{IntoView, ViewExt};

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
        let current_map = pollster::block_on(map_source.next_map(std::time::Instant::now()))?;
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
        self.current_map = self
            .map_source
            .next_map(frame.meta.timestamp.received_at)
            .await?;
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

fn project_frame(frame: Frame<'_>, map: &FrameProjectionMap) -> Result<OwnedFrame> {
    let input = frame.view();
    let output_size = map.output_size;
    let mut data = vec![0_u8; output_size.width as usize * output_size.height as usize];
    for y in 0..output_size.height {
        let dst_row_start = y as usize * output_size.width as usize;
        for x in 0..output_size.width {
            let Some((src_x, src_y)) = map.get(x, y) else {
                continue;
            };
            let row = input.row(src_y)?;
            data[dst_row_start + x as usize] = gray_at(row, input.format, src_x as usize)?;
        }
    }

    Ok(OwnedFrame {
        meta: FrameMeta {
            size: output_size,
            ..frame.meta
        },
        format: PixelFormat::Gray8,
        stride: output_size.width as usize,
        data,
    })
}

fn gray_at(row: &[u8], format: PixelFormat, x: usize) -> Result<u8> {
    match format {
        PixelFormat::Gray8 => Ok(row[x]),
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok(((row[offset] as u16 + row[offset + 1] as u16 + row[offset + 2] as u16) / 3) as u8)
        }
        PixelFormat::Yuyv422 => anyhow::bail!("projection transform does not support YUYV422"),
    }
}
