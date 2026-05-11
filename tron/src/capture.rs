use anyhow::Result;
use tron_api::{
    CameraOpenRequest, CameraOpener, CaptureFormat, FrameSource, OpenedCameraInfo, PixelFormat,
    SensorKind,
};
use tron_core::capture::v4l::V4lCameraOpener;
use tron_core::decode::mjpeg::TurboMjpegDecoder;
use tron_core::pipeline::{DecodeStream, FrameStream, PassthroughStream};

pub fn open_v4l_stream(
    request: CameraOpenRequest,
    decoded_mjpeg_format: PixelFormat,
) -> Result<(OpenedCameraInfo, Box<dyn FrameStream + Send>)> {
    let source = V4lCameraOpener.open(request)?;
    let info = source.info().clone();
    let stream = match info.format {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(decoded_mjpeg_format)?;
            Box::new(DecodeStream::new(source, decoder)) as Box<dyn FrameStream + Send>
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
            Box::new(PassthroughStream::new(source)) as Box<dyn FrameStream + Send>
        }
    };
    Ok((info, stream))
}

pub fn force_sensor(mut request: CameraOpenRequest, sensor: SensorKind) -> CameraOpenRequest {
    request.selector.sensor = sensor;
    request
}
