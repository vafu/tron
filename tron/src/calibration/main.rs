use anyhow::Result;
use clap::Parser;
use tron::capture::open_v4l_stream;
use tron::config::{CameraArgs, PixelFormatArg};
use tron::latest::LatestFrameSource;
use tron_api::{CameraOpenRequest, CaptureFormat, PixelFormat, SensorKind};

mod presenter;
mod window;

#[derive(Debug, Parser)]
#[command(name = "tron-calibration")]
#[command(about = "RGB/IR calibration capture view")]
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
        eprintln!("tron-calibration: resolving --camera {camera:?}");
    }

    let (rgb_info, rgb_stream) =
        open_v4l_stream(rgb_request(&cli), PixelFormat::from(cli.decode_format))?;
    anyhow::ensure!(
        rgb_info.format == CaptureFormat::Mjpeg,
        "calibration RGB feed currently requires MJPEG"
    );
    eprintln!(
        "tron-calibration: opened rgb {} {:?} {}x{}",
        rgb_info.id, rgb_info.format, rgb_info.size.width, rgb_info.size.height
    );

    let (ir_info, ir_stream) = open_v4l_stream(ir_request(&cli), PixelFormat::Bgra8)?;
    eprintln!(
        "tron-calibration: opened ir {} {:?} {}x{}",
        ir_info.id, ir_info.format, ir_info.size.width, ir_info.size.height
    );

    let rgb = LatestFrameSource::spawn("calibration-rgb", rgb_stream);
    let ir = LatestFrameSource::spawn("calibration-ir", ir_stream);
    window::run(rgb, ir)
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
