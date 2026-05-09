use anyhow::Result;
use clap::Parser;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{
    CameraOpenRequest, CameraOpener, CaptureFormat, FrameSource, PixelFormat, SensorKind,
};
use tron_core::capture::v4l::V4lCameraOpener;
use tron_core::decode::mjpeg::TurboMjpegDecoder;
use tron_core::pipeline::{DecodeStream, FrameStream, PassthroughStream};

mod latency;
mod window;

#[derive(Debug, Parser)]
#[command(name = "tron-playground")]
#[command(about = "Composable playground for capture/decode/process/present experiments")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Backend-native IR camera identifier. On V4L this is a path such as /dev/video51.
    #[arg(long)]
    ir_camera_id: Option<String>,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!(
            "tron-playground: resolving --camera {camera:?} for {:?}",
            cli.camera.sensor
        );
    }

    let rgb_request = rgb_request(&cli);
    let rgb_source = V4lCameraOpener.open(rgb_request)?;
    let rgb_info = rgb_source.info();
    eprintln!(
        "tron-playground: opened rgb {} {:?} {}x{}",
        rgb_info.id, rgb_info.format, rgb_info.size.width, rgb_info.size.height
    );
    let rgb_stream = match rgb_info.format {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::from(cli.decode_format))?;
            Box::new(DecodeStream::new(rgb_source, decoder)) as Box<dyn FrameStream>
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
            anyhow::bail!("playground RGB feed currently requires MJPEG")
        }
    };

    let ir_source = V4lCameraOpener.open(ir_request(&cli))?;
    let ir_info = ir_source.info();
    eprintln!(
        "tron-playground: opened ir {} {:?} {}x{}",
        ir_info.id, ir_info.format, ir_info.size.width, ir_info.size.height
    );
    let ir_stream = match ir_info.format {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::Bgra8)?;
            Box::new(DecodeStream::new(ir_source, decoder)) as Box<dyn FrameStream>
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
            Box::new(PassthroughStream::new(ir_source)) as Box<dyn FrameStream>
        }
    };

    window::run(rgb_stream, ir_stream)
}

fn rgb_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Rgb;
    if request.format.is_none() {
        request.format = Some(CaptureFormat::Mjpeg);
    }
    request
}

fn ir_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Ir;
    request.selector.id = cli.ir_camera_id.clone();
    request.format = None;
    request.size = None;
    request
}
