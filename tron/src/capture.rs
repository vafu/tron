use anyhow::Result;
use std::thread;
use std::time::Duration;
use tron_api::{
    CameraOpenRequest, CameraOpener, Frame, FrameSource, OpenedCameraInfo, PixelFormat, SensorKind,
};
use tron_core::capture::v4l::{
    V4lCameraOpener, V4lFrameSource, V4lUvcmMetadataSource, resolve_device as resolve_v4l_device,
};

use crate::face_auth::{FaceAuthMode, set_face_auth_mode};

pub fn open_v4l_stream(
    request: CameraOpenRequest,
    decoded_mjpeg_format: PixelFormat,
) -> Result<(OpenedCameraInfo, V4lFrameSource)> {
    let source = V4lCameraOpener::with_decoded_mjpeg_format(decoded_mjpeg_format).open(request)?;
    let info = source.info().clone();
    Ok((info, source))
}

pub struct WindowsHelloV4lStreams {
    pub rgb_info: OpenedCameraInfo,
    pub rgb_stream: V4lFrameSource,
    pub ir_info: OpenedCameraInfo,
    pub ir_stream: V4lFrameSource,
    pub ir_device_id: String,
    pub ir_metadata_id: String,
}

pub struct WindowsHelloV4lConfig {
    pub rgb_request: CameraOpenRequest,
    pub ir_request: CameraOpenRequest,
    pub ir_metadata_id: Option<String>,
    pub decoded_rgb_format: PixelFormat,
    pub decoded_ir_format: PixelFormat,
}

pub fn open_windows_hello_v4l_streams(
    config: WindowsHelloV4lConfig,
) -> Result<WindowsHelloV4lStreams> {
    let WindowsHelloV4lConfig {
        rgb_request,
        ir_request,
        ir_metadata_id,
        decoded_rgb_format,
        decoded_ir_format,
    } = config;

    let ir_device_id = resolve_v4l_device(&ir_request.selector)?;
    let ir_metadata_id = ir_metadata_id
        .or_else(|| infer_metadata_node(&ir_device_id))
        .ok_or_else(|| {
            anyhow::anyhow!("--ir-metadata-id is required when IR node is not /dev/videoN")
        })?;

    eprintln!("tron: setting face-auth default mode on {ir_device_id}");
    set_face_auth_mode(&ir_device_id, FaceAuthMode::Default)?;

    let (rgb_info, mut rgb_stream) = open_v4l_stream(rgb_request, decoded_rgb_format)?;
    warm_up_stream(&mut rgb_stream, "rgb")?;

    eprintln!("tron: setting face-auth mode2 on {ir_device_id}");
    set_face_auth_mode(&ir_device_id, FaceAuthMode::Mode2)?;

    let (ir_info, ir_stream) = open_v4l_stream(ir_request, decoded_ir_format)?;

    Ok(WindowsHelloV4lStreams {
        rgb_info,
        rgb_stream,
        ir_info,
        ir_stream,
        ir_device_id,
        ir_metadata_id,
    })
}

fn warm_up_stream(source: &mut impl FrameSource, label: &str) -> Result<()> {
    for _ in 0..30 {
        if source.next_frame()?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(1));
    }
    anyhow::bail!("{label} stream did not produce a valid warm-up frame")
}

pub fn force_sensor(mut request: CameraOpenRequest, sensor: SensorKind) -> CameraOpenRequest {
    request.selector.sensor = sensor;
    request
}

pub fn infer_metadata_node(video_node: &str) -> Option<String> {
    let number = video_node.strip_prefix("/dev/video")?.parse::<u32>().ok()?;
    Some(format!("/dev/video{}", number + 1))
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
