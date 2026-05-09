use std::time::{Duration, Instant};
use tron_api::Frame;

#[derive(Clone, Copy, Debug)]
pub struct LatencySample {
    pub rgb_wait: Duration,
    pub ir_wait: Duration,
    pub present: Duration,
    pub rgb_age_after_acquire: u128,
    pub rgb_age_before_present: u128,
    pub ir_age_after_acquire: u128,
    pub camera_delta_us: Option<i64>,
}

#[derive(Debug)]
pub struct LatencyProbe {
    interval: Duration,
    last_log: Instant,
    frames: u32,
    rgb_wait: DurationStats,
    ir_wait: DurationStats,
    present: DurationStats,
    rgb_age_after_acquire: UsStats,
    rgb_age_before_present: UsStats,
    ir_age_after_acquire: UsStats,
    camera_delta_abs: UsStats,
}

impl LatencyProbe {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: Instant::now(),
            frames: 0,
            rgb_wait: DurationStats::default(),
            ir_wait: DurationStats::default(),
            present: DurationStats::default(),
            rgb_age_after_acquire: UsStats::default(),
            rgb_age_before_present: UsStats::default(),
            ir_age_after_acquire: UsStats::default(),
            camera_delta_abs: UsStats::default(),
        }
    }

    pub fn record(&mut self, sample: LatencySample) {
        self.frames += 1;
        self.rgb_wait.add(sample.rgb_wait);
        self.ir_wait.add(sample.ir_wait);
        self.present.add(sample.present);
        self.rgb_age_after_acquire.add(sample.rgb_age_after_acquire);
        self.rgb_age_before_present
            .add(sample.rgb_age_before_present);
        self.ir_age_after_acquire.add(sample.ir_age_after_acquire);
        if let Some(delta) = sample.camera_delta_us {
            self.camera_delta_abs.add(delta.unsigned_abs() as u128);
        }

        if self.last_log.elapsed() < self.interval {
            return;
        }

        let elapsed = self.last_log.elapsed().as_secs_f32();
        eprintln!(
            "latency: fps={:.1} rgb_wait={} ir_wait={} present={} rgb_age_after={} rgb_age_before_present={} ir_age_after={} cam_delta_abs={}",
            self.frames as f32 / elapsed,
            self.rgb_wait,
            self.ir_wait,
            self.present,
            self.rgb_age_after_acquire,
            self.rgb_age_before_present,
            self.ir_age_after_acquire,
            self.camera_delta_abs,
        );
        self.reset();
    }

    fn reset(&mut self) {
        self.last_log = Instant::now();
        self.frames = 0;
        self.rgb_wait = DurationStats::default();
        self.ir_wait = DurationStats::default();
        self.present = DurationStats::default();
        self.rgb_age_after_acquire = UsStats::default();
        self.rgb_age_before_present = UsStats::default();
        self.ir_age_after_acquire = UsStats::default();
        self.camera_delta_abs = UsStats::default();
    }
}

pub fn age_us(frame: Frame<'_>, now: Instant) -> u128 {
    now.duration_since(frame.meta.timestamp.received_at)
        .as_micros()
}

pub fn camera_delta_us(rgb: Frame<'_>, ir: Frame<'_>) -> Option<i64> {
    let rgb_ts = rgb.meta.timestamp.camera_monotonic_us?;
    let ir_ts = ir.meta.timestamp.camera_monotonic_us?;
    Some(rgb_ts - ir_ts)
}

#[derive(Clone, Copy, Debug, Default)]
struct DurationStats {
    count: u32,
    total: Duration,
    max: Duration,
}

impl DurationStats {
    fn add(&mut self, value: Duration) {
        self.count += 1;
        self.total += value;
        self.max = self.max.max(value);
    }
}

impl std::fmt::Display for DurationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            return write!(f, "n/a");
        }
        write!(
            f,
            "avg={:.2}ms max={:.2}ms",
            self.total.as_secs_f64() * 1000.0 / f64::from(self.count),
            self.max.as_secs_f64() * 1000.0,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct UsStats {
    count: u32,
    total: u128,
    max: u128,
}

impl UsStats {
    fn add(&mut self, value: u128) {
        self.count += 1;
        self.total += value;
        self.max = self.max.max(value);
    }
}

impl std::fmt::Display for UsStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            return write!(f, "n/a");
        }
        write!(
            f,
            "avg={:.2}ms max={:.2}ms",
            self.total as f64 / f64::from(self.count) / 1000.0,
            self.max as f64 / 1000.0,
        )
    }
}
