use anyhow::Result;
use clap::Parser;
use std::time::{Duration, Instant};
use tron::config::{CameraArgs, PixelFormatArg, parse_positive_u32};
use tron_api::{CaptureFormat, FrameStats, FrameViewModel, NamedFrame, PixelFormat, Presenter};
use tron_core::capture::v4l::V4lFrameSource;
use tron_core::decode::mjpeg::TurboMjpegDecoder;
use tron_core::pipeline::{DecodeStream, FrameStream, PassthroughStream};
use tron_core::present::text::TextStatsPresenter;

#[derive(Debug, Parser)]
#[command(name = "tron-playground")]
#[command(about = "Composable playground for capture/decode/process/present experiments")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Text stats presentation interval in seconds.
    #[arg(long, value_parser = parse_positive_u32, default_value = "2")]
    stats_interval_secs: u32,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!(
            "tron-playground: --camera {camera:?} accepted but name resolution is not wired yet; using --device {}",
            cli.camera.device
        );
    }

    let source_config = cli.camera.to_v4l_config();
    eprintln!(
        "tron-playground: opening {} {:?} {:?} {}x{}@{}fps buffers={}",
        source_config.path,
        source_config.sensor,
        source_config.format,
        source_config.width,
        source_config.height,
        cli.camera.fps,
        source_config.buffers
    );

    let source = V4lFrameSource::open(source_config)?;
    match cli.camera.capture_format() {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::from(cli.decode_format))?;
            run_stream(
                DecodeStream::new(source, decoder),
                Duration::from_secs(cli.stats_interval_secs as u64),
            )
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => run_stream(
            PassthroughStream::new(source),
            Duration::from_secs(cli.stats_interval_secs as u64),
        ),
    }
}

fn run_stream(mut stream: impl FrameStream, stats_interval: Duration) -> Result<()> {
    let mut presenter = TextStatsPresenter::new(stats_interval);

    loop {
        let start = Instant::now();
        let frame = stream.next_frame()?;
        let frames = [NamedFrame {
            name: "camera",
            frame,
        }];
        presenter.present(FrameViewModel {
            frames: &frames,
            metadata: FrameStats {
                acquire_us: start.elapsed().as_micros() as u64,
            },
        })?;
    }
}
