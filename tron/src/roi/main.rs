use anyhow::Result;
use clap::Parser;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{
    CameraOpenRequest, CameraOpener, CaptureFormat, FrameSource, PixelFormat, SensorKind,
};
use tron_core::capture::v4l::V4lCameraOpener;
use tron_core::decode::mjpeg::TurboMjpegDecoder;
use tron_core::pipeline::{DecodeStream, FrameStream, PassthroughStream};

mod overlay;
mod roi;
mod sweep;
mod window;

#[derive(Debug, Parser)]
#[command(name = "roi-control")]
#[command(about = "Single-camera IR stream with live ROI exposure controls")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Initial ROI rectangle as x,y,width,height in frame pixels.
    #[arg(long, value_parser = parse_roi, default_value = "280,140,80,80")]
    roi: roi::RoiRect,

    /// Keyboard movement step in pixels.
    #[arg(long, default_value_t = 10)]
    step: u32,

    /// Initial autopilot sweep speed in pixels per second.
    #[arg(long, default_value_t = 160.0)]
    sweep_speed: f32,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    let request = camera_request(&cli);
    let source = V4lCameraOpener.open(request)?;
    let info = source.info().clone();
    eprintln!(
        "roi-control: opened {} {:?} {}x{}",
        info.id, info.format, info.size.width, info.size.height
    );
    eprintln!(
        "roi-control: left click/drag moves ROI, wheel resizes, arrows move, +/- resize, R reset, 1 enables ROI auto exposure, 0 disables it"
    );
    eprintln!("roi-control: F1..F7 set exact square sizes: 1, 8, 16, 24, 32, 48, 64");
    eprintln!("roi-control: A toggles horizontal ROI sweep, [/] coarse speed, ,/. fine speed");

    let stream = match info.format {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::from(cli.decode_format))?;
            Box::new(DecodeStream::new(source, decoder)) as Box<dyn FrameStream>
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
            Box::new(PassthroughStream::new(source)) as Box<dyn FrameStream>
        }
    };

    let controller = roi::RoiController::new(info.id.clone(), cli.roi, cli.step);
    window::run(stream, controller, cli.sweep_speed)
}

fn camera_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Ir;
    request
}

fn parse_roi(value: &str) -> std::result::Result<roi::RoiRect, String> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| format!("invalid ROI {value:?}: {err}"))?;
    if parts.len() != 4 {
        return Err(format!("invalid ROI {value:?}; expected x,y,width,height"));
    }
    Ok(roi::RoiRect {
        x: parts[0],
        y: parts[1],
        width: parts[2],
        height: parts[3],
    })
}
