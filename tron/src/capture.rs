use anyhow::Result;
use std::thread;
use std::time::Duration;
use tron_api::{
    CameraOpenRequest, CameraOpener, FrameSource, OpenedCameraInfo, PixelFormat, SensorKind,
};
use tron_core::capture::v4l::{
    V4lCameraOpener, V4lFrameSource, resolve_device as resolve_v4l_device,
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
        if pollster::block_on(source.next_frame())?.is_some() {
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
