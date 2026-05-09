use anyhow::Result;
use std::time::{Duration, Instant};
use tron_api::{Frame, Presenter};

#[derive(Clone, Copy, Debug)]
pub struct TextFrameView<'a> {
    pub name: &'static str,
    pub frame: Frame<'a>,
    pub acquire_us: u64,
}

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

impl<'a> Presenter<TextFrameView<'a>> for TextStatsPresenter {
    fn present(&mut self, view: TextFrameView<'a>) -> Result<()> {
        self.frames += 1;
        self.acquire_us += view.acquire_us;

        if self.last_log.elapsed() >= self.interval {
            let elapsed = self.last_log.elapsed().as_secs_f32();
            let n = self.frames.max(1) as f32;
            let frame = view.frame;
            eprintln!(
                "pipeline: fps={:.1} acquire={:.3}ms {}={}x{} {:?} stride={} len={} age={:.2}ms id={} seq={:?} cam_ts={:?} ts_src={:?}",
                self.frames as f32 / elapsed,
                self.acquire_us as f32 / n / 1000.0,
                view.name,
                frame.meta.size.width,
                frame.meta.size.height,
                frame.format,
                frame.stride,
                frame.data.len(),
                frame.meta.timestamp.received_at.elapsed().as_secs_f32() * 1000.0,
                frame.meta.id,
                frame.meta.sequence,
                frame.meta.timestamp.camera_monotonic_us,
                frame.meta.timestamp.source
            );
            self.last_log = Instant::now();
            self.frames = 0;
            self.acquire_us = 0;
        }

        Ok(())
    }
}
