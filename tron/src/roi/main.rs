use anyhow::Result;
use clap::Parser;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CameraOpener, FrameSource, PixelFormat, SensorKind};
use tron_core::capture::v4l::V4lCameraOpener;
use tron_core::capture::v4l_control::V4lCameraRoiControl;
use tron_core::render::http::HttpJsonSink;

mod overlay;
mod roi;
mod sweep;
mod uvc_step;
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
    roi: tron_api::Rect,

    /// Keyboard movement step in pixels.
    #[arg(long, default_value_t = 10)]
    step: u32,

    /// Initial autopilot sweep speed in pixels per second.
    #[arg(long, default_value_t = 160.0)]
    sweep_speed: f32,

    /// Enable Enter-to-step UVC mode writes for emitter experiments.
    #[arg(long)]
    uvc_step: bool,

    /// UVC extension unit used by --uvc-step.
    #[arg(long, default_value_t = 4)]
    uvc_unit: u8,

    /// UVC extension selector used by --uvc-step.
    #[arg(long, default_value_t = 6)]
    uvc_selector: u8,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    let source = V4lCameraOpener::with_decoded_mjpeg_format(PixelFormat::from(cli.decode_format))
        .open(camera_request(&cli))?;
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

    let controller = roi::RoiController::new(
        Box::new(V4lCameraRoiControl::open(&info.id)?),
        cli.roi,
        cli.step,
    );
    let uvc_stepper = if cli.uvc_step {
        Some(uvc_step::UvcStepper::new(
            info.id.clone(),
            cli.uvc_unit,
            cli.uvc_selector,
        )?)
    } else {
        None
    };
    let mut sinks = window::ComboSink::new();
    let metadata = HttpJsonSink::bind_available(("127.0.0.1", 8765), 100)?;
    eprintln!(
        "roi-control: metadata http://{}/metadata",
        metadata.local_addr()
    );
    sinks.push_box(Box::new(metadata));
    window::run(
        Box::new(source),
        controller,
        cli.sweep_speed,
        uvc_stepper,
        sinks,
    )
}

fn camera_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Ir;
    request
}

fn parse_roi(value: &str) -> std::result::Result<tron_api::Rect, String> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| format!("invalid ROI {value:?}: {err}"))?;
    if parts.len() != 4 {
        return Err(format!("invalid ROI {value:?}; expected x,y,width,height"));
    }
    Ok(tron_api::Rect {
        x: parts[0],
        y: parts[1],
        size: tron_api::Size {
            width: parts[2],
            height: parts[3],
        },
    })
}
