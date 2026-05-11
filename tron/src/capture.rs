use anyhow::Result;
use tron_api::{
    CameraOpenRequest, CameraOpener, Frame, FrameSource, OpenedCameraInfo, PixelFormat, SensorKind,
};
use tron_core::capture::v4l::{V4lCameraOpener, V4lUvcmMetadataSource};

pub fn open_v4l_stream(
    request: CameraOpenRequest,
    decoded_mjpeg_format: PixelFormat,
) -> Result<(OpenedCameraInfo, impl FrameSource)> {
    let source = V4lCameraOpener::with_decoded_mjpeg_format(decoded_mjpeg_format).open(request)?;
    let info = source.info().clone();
    Ok((info, source))
}

pub fn force_sensor(mut request: CameraOpenRequest, sensor: SensorKind) -> CameraOpenRequest {
    request.selector.sensor = sensor;
    request
}

pub struct LitIrFrameStream<S> {
    inner: S,
    metadata: LitIrMetadataFilter,
    sequence_mismatch_count: u32,
}

impl<S> LitIrFrameStream<S> {
    pub fn new(inner: S, metadata: V4lUvcmMetadataSource) -> Self {
        Self {
            inner,
            metadata: LitIrMetadataFilter::new(metadata),
            sequence_mismatch_count: 0,
        }
    }
}

impl<S> FrameSource for LitIrFrameStream<S>
where
    S: FrameSource,
{
    fn info(&self) -> &OpenedCameraInfo {
        self.inner.info()
    }

    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let Self {
            inner,
            metadata,
            sequence_mismatch_count,
        } = self;
        let Some(frame) = inner.next_frame()? else {
            return Ok(None);
        };
        let illumination_on = if let Some(sequence) = frame.meta.sequence {
            match metadata.illumination_for_sequence(sequence)? {
                Some(illumination_on) => illumination_on,
                None => {
                    note_sequence_mismatch(sequence_mismatch_count, sequence);
                    let Some(illumination_on) = metadata.next_illumination()? else {
                        metadata.note_missing_metadata(&format!(
                            "no UVCM frame illumination metadata for IR frame sequence {sequence}"
                        ));
                        return Ok(None);
                    };
                    illumination_on
                }
            }
        } else {
            metadata.note_missing_metadata("IR frame has no V4L sequence metadata");
            let Some(illumination_on) = metadata.next_illumination()? else {
                return Ok(None);
            };
            illumination_on
        };
        metadata.note_matched_metadata();
        if illumination_on {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }
}

fn note_sequence_mismatch(sequence_mismatch_count: &mut u32, sequence: u64) {
    *sequence_mismatch_count = sequence_mismatch_count.saturating_add(1);
    if *sequence_mismatch_count == 1 || *sequence_mismatch_count % 120 == 0 {
        eprintln!(
            "calibration-ir: no exact UVCM metadata sequence match for IR frame sequence {sequence}; falling back to metadata dequeue order"
        );
    }
}

struct LitIrMetadataFilter {
    source: V4lUvcmMetadataSource,
    missing_metadata_count: u32,
}

impl LitIrMetadataFilter {
    fn new(source: V4lUvcmMetadataSource) -> Self {
        Self {
            source,
            missing_metadata_count: 0,
        }
    }

    fn illumination_for_sequence(&mut self, sequence: u64) -> Result<Option<bool>> {
        self.source.illumination_for_sequence(sequence)
    }

    fn next_illumination(&mut self) -> Result<Option<bool>> {
        Ok(self
            .source
            .next_illumination()?
            .map(|illumination| illumination.illumination_on))
    }

    fn note_matched_metadata(&mut self) {
        self.missing_metadata_count = 0;
    }

    fn note_missing_metadata(&mut self, message: &str) {
        self.missing_metadata_count = self.missing_metadata_count.saturating_add(1);
        if self.missing_metadata_count == 1 || self.missing_metadata_count % 120 == 0 {
            eprintln!(
                "calibration-ir: {message}; skipping pair. If this persists, ensure the camera is in face-auth/mode2 alternating illumination mode."
            );
        }
    }
}
