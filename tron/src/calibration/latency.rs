use std::time::{Duration, Instant};

use tron_api::OwnedFrame;

#[derive(Default)]
pub struct CalibrationLatencyLog {
    window_start: Option<Instant>,
    frames: u32,
    latest: DurationStats,
    rgb_detect: DurationStats,
    ir_detect: DurationStats,
    present: DurationStats,
    total: DurationStats,
    rgb_age: DurationStats,
    ir_age: DurationStats,
    sync_delta: MicrosecondStats,
    rgb_detected: u32,
    ir_detected: u32,
}

impl CalibrationLatencyLog {
    pub fn record(&mut self, sample: CalibrationLatencySample<'_>) {
        let window_start = *self.window_start.get_or_insert(sample.finished_at);
        self.frames += 1;
        self.latest.record(sample.latest);
        self.rgb_detect.record(sample.rgb_detect);
        self.ir_detect.record(sample.ir_detect);
        self.present.record(sample.present);
        self.total.record(sample.total);
        if sample.rgb_detected {
            self.rgb_detected += 1;
        }
        if sample.ir_detected {
            self.ir_detected += 1;
        }
        if let Some(frame) = sample.rgb {
            self.rgb_age.record(
                sample
                    .finished_at
                    .saturating_duration_since(frame.meta.timestamp.received_at),
            );
        }
        if let Some(frame) = sample.ir {
            self.ir_age.record(
                sample
                    .finished_at
                    .saturating_duration_since(frame.meta.timestamp.received_at),
            );
        }
        if let Some(delta_us) = sample.sync_delta_us {
            self.sync_delta.record(delta_us);
        }

        let elapsed = sample.finished_at.saturating_duration_since(window_start);
        if elapsed < Duration::from_secs(1) {
            return;
        }

        let fps = self.frames as f64 / elapsed.as_secs_f64().max(0.001);
        tracing::info!(
            target: "calibration::latency",
            "fps={fps:.1} latest={} rgb_detect={} ir_detect={} present={} total={} rgb_age={} ir_age={} sync_delta={} detections=rgb:{}/{} ir:{}/{}",
            self.latest,
            self.rgb_detect,
            self.ir_detect,
            self.present,
            self.total,
            self.rgb_age,
            self.ir_age,
            self.sync_delta,
            self.rgb_detected,
            self.frames,
            self.ir_detected,
            self.frames,
        );
        self.reset(sample.finished_at);
    }

    fn reset(&mut self, now: Instant) {
        *self = Self {
            window_start: Some(now),
            ..Self::default()
        };
    }
}

pub struct CalibrationLatencySample<'a> {
    pub latest: Duration,
    pub rgb_detect: Duration,
    pub ir_detect: Duration,
    pub present: Duration,
    pub total: Duration,
    pub finished_at: Instant,
    pub rgb: Option<&'a OwnedFrame>,
    pub ir: Option<&'a OwnedFrame>,
    pub sync_delta_us: Option<i64>,
    pub rgb_detected: bool,
    pub ir_detected: bool,
}

#[derive(Default)]
struct DurationStats {
    count: u32,
    total: Duration,
    max: Duration,
}

impl DurationStats {
    fn record(&mut self, value: Duration) {
        self.count += 1;
        self.total += value;
        self.max = self.max.max(value);
    }

    fn average(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total / self.count
        }
    }
}

impl std::fmt::Display for DurationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "avg={:.2}ms max={:.2}ms",
            self.average().as_secs_f64() * 1000.0,
            self.max.as_secs_f64() * 1000.0
        )
    }
}

#[derive(Default)]
struct MicrosecondStats {
    count: u32,
    total_abs_us: u128,
    max_abs_us: u64,
}

impl MicrosecondStats {
    fn record(&mut self, value_us: i64) {
        let abs_us = value_us.unsigned_abs();
        self.count += 1;
        self.total_abs_us += abs_us as u128;
        self.max_abs_us = self.max_abs_us.max(abs_us);
    }

    fn average_abs_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_abs_us as f64 / self.count as f64
        }
    }
}

impl std::fmt::Display for MicrosecondStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "avg_abs={:.2}ms max_abs={:.2}ms",
            self.average_abs_us() / 1000.0,
            self.max_abs_us as f64 / 1000.0
        )
    }
}
