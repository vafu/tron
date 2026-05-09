use anyhow::Result;
use clap::Parser;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpener, CaptureFormat, FrameSource, PixelFormat};
use tron_core::capture::v4l::V4lCameraOpener;
use tron_core::decode::mjpeg::TurboMjpegDecoder;
use tron_core::pipeline::DecodeStream;

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

    let requested_format = cli.camera.requested_format();
    let source = V4lCameraOpener.open(cli.camera.open_request())?;
    let info = source.info();
    eprintln!(
        "tron-playground: opened {} {:?} {}x{}",
        info.id, info.format, info.size.width, info.size.height
    );

    match requested_format.unwrap_or(info.format) {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::from(cli.decode_format))?;
            window::run(DecodeStream::new(source, decoder))
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
            anyhow::bail!("playground window currently requires --format mjpg")
        }
    }
}
