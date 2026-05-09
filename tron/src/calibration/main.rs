use anyhow::Result;
use clap::Parser;
use tron::config::CameraArgs;

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
    eprintln!(
        "tron-calibration: scaffold ready for {:?} on {}",
        cli.camera.sensor, cli.camera.device
    );
    Ok(())
}
