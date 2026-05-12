use anyhow::Result;
use tron_api::{Frame, FrameSource, OpenedCameraInfo};

use super::uvcm_metadata::V4lUvcmMetadataSource;

pub struct LitIrFrameStream<S> {
    inner: S,
    metadata: V4lUvcmMetadataSource,
    missing_metadata_count: u32,
}

impl<S> LitIrFrameStream<S> {
    pub fn new(inner: S, metadata: V4lUvcmMetadataSource) -> Self {
        Self {
            inner,
            metadata,
            missing_metadata_count: 0,
        }
    }
}

#[async_trait::async_trait]
impl<S> FrameSource for LitIrFrameStream<S>
where
    S: FrameSource + Send,
{
    fn info(&self) -> &OpenedCameraInfo {
        self.inner.info()
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let Self {
            inner,
            metadata,
            missing_metadata_count,
        } = self;
        let Some(frame) = inner.next_frame().await? else {
            return Ok(None);
        };
        let Some(sequence) = frame.meta.sequence else {
            note_missing_metadata(
                missing_metadata_count,
                "IR frame has no V4L sequence metadata",
            );
            return Ok(None);
        };
        let Some(illumination_on) = metadata.illumination_for_sequence(sequence).await? else {
            note_missing_metadata(
                missing_metadata_count,
                &format!("no UVCM frame illumination metadata for IR frame sequence {sequence}"),
            );
            return Ok(None);
        };
        *missing_metadata_count = 0;
        if illumination_on {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }
}

fn note_missing_metadata(missing_metadata_count: &mut u32, message: &str) {
    *missing_metadata_count = missing_metadata_count.saturating_add(1);
    if *missing_metadata_count == 1 || *missing_metadata_count % 120 == 0 {
        eprintln!(
            "lit-ir: {message}; skipping frame. If this persists, ensure the camera is in face-auth/mode2 alternating illumination mode."
        );
    }
}
