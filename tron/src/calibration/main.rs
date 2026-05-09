use anyhow::Result;
use clap::Parser;
use tron::config::CameraArgs;
use tron_api::{CameraOpener, FrameSource};
use tron_core::capture::v4l::V4lCameraOpener;

#[derive(Debug, Parser)]
#[command(name = "tron-calibration")]
#[command(about = "Calibration binary scaffold")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    let source = V4lCameraOpener.open(cli.camera.open_request())?;
    let info = source.info();
    eprintln!(
        "tron-calibration: scaffold ready for {:?} on {}",
        info.sensor, info.id
    );
    Ok(())
}
