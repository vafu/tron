use anyhow::Result;
use std::time::{Duration, Instant};
use tron_api::{FrameStats, FrameViewModel, Presenter};

pub struct TextStatsPresenter {
    interval: Duration,
    last_log: Instant,
    frames: u32,
    acquire_us: u64,
}

impl TextStatsPresenter {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: Instant::now(),
            frames: 0,
            acquire_us: 0,
        }
    }
}

impl<'a> Presenter<FrameViewModel<'a, FrameStats>> for TextStatsPresenter {
    fn present(&mut self, view: FrameViewModel<'a, FrameStats>) -> Result<()> {
        self.frames += 1;
        self.acquire_us += view.metadata.acquire_us;

        if self.last_log.elapsed() >= self.interval {
            let elapsed = self.last_log.elapsed().as_secs_f32();
            let n = self.frames.max(1) as f32;
            let frame_summary = view
                .frames
                .iter()
                .map(|named| {
                    let frame = named.frame;
                    format!(
                        "{}={}x{} {:?} stride={} len={} age={:.2}ms id={} seq={:?}",
                        named.name,
                        frame.meta.size.width,
                        frame.meta.size.height,
                        frame.format,
                        frame.stride,
                        frame.data.len(),
                        frame.meta.timestamp.received_at.elapsed().as_secs_f32() * 1000.0,
                        frame.meta.id,
                        frame.meta.sequence
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            let first = view.frames.first().map(|named| named.frame);
            eprintln!(
                "pipeline: fps={:.1} acquire={:.3}ms {} cam_ts={:?} ts_src={:?}",
                self.frames as f32 / elapsed,
                self.acquire_us as f32 / n / 1000.0,
                frame_summary,
                first.and_then(|frame| frame.meta.timestamp.camera_monotonic_us),
                first.map(|frame| frame.meta.timestamp.source)
            );
            self.last_log = Instant::now();
            self.frames = 0;
            self.acquire_us = 0;
        }

        Ok(())
    }
}
