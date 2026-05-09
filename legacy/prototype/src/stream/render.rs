use anyhow::Result;
use std::time::{Duration, Instant};

use crate::stream::frame::Frame;
use crate::stream::process::NoContext;

pub trait RenderSink<I, C = NoContext> {
    fn submit(&mut self, input: I, context: C) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub acquire_us: u64,
}

pub struct TextStatsSink {
    interval: Duration,
    last_log: Instant,
    frames: u32,
    acquire_us: u64,
}

impl TextStatsSink {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: Instant::now(),
            frames: 0,
            acquire_us: 0,
        }
    }
}

impl<'a> RenderSink<Frame<'a>, FrameStats> for TextStatsSink {
    fn submit(&mut self, frame: Frame<'a>, stats: FrameStats) -> Result<()> {
        self.frames += 1;
        self.acquire_us += stats.acquire_us;

        if self.last_log.elapsed() >= self.interval {
            let elapsed = self.last_log.elapsed().as_secs_f32();
            let n = self.frames.max(1) as f32;
            eprintln!(
                "pipeline: fps={:.1} acquire={:.3}ms id={} seq={:?} {:?} {}x{} stride={} len={} age={:.2}ms cam_ts={:?} ts_src={:?}",
                self.frames as f32 / elapsed,
                self.acquire_us as f32 / n / 1000.0,
                frame.meta.id,
                frame.meta.sequence,
                frame.format,
                frame.meta.width,
                frame.meta.height,
                frame.stride,
                frame.data.len(),
                frame.meta.timestamp.received_at.elapsed().as_secs_f32() * 1000.0,
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
