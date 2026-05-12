use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::task::JoinHandle;
use tokio_serial::SerialPortBuilderExt;
use tron_api::{DepthSample, DepthSource};

const DEFAULT_SAMPLE_CAPACITY: usize = 128;

pub struct Vl53l5cxSerialDepthSource {
    shared: Arc<Mutex<SharedSamples>>,
    reader_task: JoinHandle<()>,
}

struct SharedSamples {
    samples: VecDeque<DepthSample>,
    capacity: usize,
    error: Option<anyhow::Error>,
}

impl Vl53l5cxSerialDepthSource {
    pub fn open(path: impl AsRef<Path>, baud_rate: u32, timeout: Duration) -> Result<Self> {
        Self::open_with_capacity(path, baud_rate, timeout, DEFAULT_SAMPLE_CAPACITY)
    }

    pub fn open_with_capacity(
        path: impl AsRef<Path>,
        baud_rate: u32,
        timeout: Duration,
        capacity: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let port = tokio_serial::new(path.to_string_lossy(), baud_rate)
            .timeout(timeout)
            .open_native_async()
            .with_context(|| format!("open VL53L5CX serial port {}", path.display()))?;
        let capacity = capacity.max(1);
        let shared = Arc::new(Mutex::new(SharedSamples {
            samples: VecDeque::with_capacity(capacity),
            capacity,
            error: None,
        }));
        let reader_task = tokio::spawn(read_samples(BufReader::new(port), shared.clone()));
        Ok(Self {
            shared,
            reader_task,
        })
    }

    fn closest_sample(&self, timestamp: Instant) -> Result<Option<DepthSample>> {
        let samples: Vec<_> = match self.shared.try_lock() {
            Ok(mut shared) => {
                if let Some(err) = shared.error.take() {
                    return Err(err);
                }
                shared.samples.iter().copied().collect()
            }
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => anyhow::bail!("VL53L5CX sample lock poisoned"),
        };

        Ok(samples
            .iter()
            .min_by_key(|sample| instant_distance(sample.sampled_at, timestamp))
            .copied())
    }
}

impl Drop for Vl53l5cxSerialDepthSource {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

#[async_trait::async_trait]
impl DepthSource for Vl53l5cxSerialDepthSource {
    async fn depth_at(&mut self, timestamp: Instant) -> Result<Option<DepthSample>> {
        self.closest_sample(timestamp)
    }
}

async fn read_samples(
    mut reader: BufReader<tokio_serial::SerialStream>,
    shared: Arc<Mutex<SharedSamples>>,
) {
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => return,
            Ok(_) => {
                let received_at = Instant::now();
                match parse_sample(&line, received_at) {
                    Ok(Some(sample)) => {
                        if let Ok(mut shared) = shared.try_lock() {
                            if shared.samples.len() >= shared.capacity {
                                shared.samples.pop_front();
                            }
                            shared.samples.push_back(sample);
                        }
                    }
                    Ok(None) => {}
                    Err(err) => {
                        set_error(&shared, err);
                        return;
                    }
                }
            }
            Err(err) => {
                set_error(
                    &shared,
                    anyhow::Error::new(err).context("read VL53L5CX serial sample"),
                );
                return;
            }
        }
        tokio::task::yield_now().await;
    }
}

fn set_error(shared: &Arc<Mutex<SharedSamples>>, err: anyhow::Error) {
    if let Ok(mut shared) = shared.lock() {
        shared.error = Some(err);
    }
}

fn instant_distance(left: Instant, right: Instant) -> Duration {
    if left >= right {
        left.duration_since(right)
    } else {
        right.duration_since(left)
    }
}

fn parse_sample(line: &str, received_at: Instant) -> Result<Option<DepthSample>> {
    let Some(rest) = line.trim().strip_prefix("tof ") else {
        return Ok(None);
    };
    if !rest.starts_with("seq=") {
        return Ok(None);
    }

    let mut sequence = None;
    let mut sensor_timestamp_us = None;
    let mut printed_at_ms = None;
    let mut resolution = None;
    let mut center_mm = None;
    let mut min_mm = None;
    let mut max_mm = None;
    let mut valid_zones = None;
    let mut zones = [0; 64];
    let mut zone_count = 0;

    for field in rest.split_whitespace() {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        match key {
            "seq" => sequence = Some(value.parse().context("parse TOF sequence")?),
            "t_ms" => printed_at_ms = Some(value.parse::<u64>().context("parse TOF t_ms")?),
            "irq_us" => sensor_timestamp_us = Some(value.parse().context("parse TOF irq_us")?),
            "res" => resolution = Some(value.parse().context("parse TOF resolution")?),
            "center_mm" => center_mm = Some(value.parse().context("parse TOF center_mm")?),
            "min_mm" => {
                let value = value.parse().context("parse TOF min_mm")?;
                min_mm = Some(value);
            }
            "max_mm" => max_mm = Some(value.parse().context("parse TOF max_mm")?),
            "valid" => valid_zones = Some(value.parse().context("parse TOF valid")?),
            "zones" => zone_count = parse_zones(value, &mut zones)?,
            _ => {}
        }
    }

    let sampled_at = match (printed_at_ms, sensor_timestamp_us) {
        (Some(printed_at_ms), Some(sensor_timestamp_us)) => {
            let printed_at_us = printed_at_ms.saturating_mul(1000);
            let age_us = printed_at_us.saturating_sub(sensor_timestamp_us);
            received_at
                .checked_sub(Duration::from_micros(age_us))
                .unwrap_or(received_at)
        }
        _ => received_at,
    };

    if !min_mm.is_some_and(|value| value > 0) {
        return Ok(None);
    }

    Ok(Some(DepthSample {
        sequence,
        sensor_timestamp_us,
        printed_at_ms,
        resolution,
        center_mm,
        min_mm,
        max_mm,
        valid_zones,
        zones,
        zone_count,
        sampled_at,
        received_at,
    }))
}

fn parse_zones(value: &str, zones: &mut [u16; 64]) -> Result<usize> {
    let mut count = 0;
    for part in value.split(',') {
        anyhow::ensure!(count < zones.len(), "TOF zones has more than 64 entries");
        zones[count] = part.parse().context("parse TOF zone distance")?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sample_line() {
        let received_at = Instant::now();
        let sample = parse_sample(
            "tof seq=438 t_ms=34789 irq_us=34789000 res=64 center_mm=386 min_mm=26 max_mm=624 valid=26 zones=405,0,386\n",
            received_at,
        )
        .unwrap()
        .unwrap();

        assert_eq!(sample.sequence, Some(438));
        assert_eq!(sample.printed_at_ms, Some(34_789));
        assert_eq!(sample.sensor_timestamp_us, Some(34_789_000));
        assert_eq!(sample.resolution, Some(64));
        assert_eq!(sample.center_mm, Some(386));
        assert_eq!(sample.min_mm, Some(26));
        assert_eq!(sample.max_mm, Some(624));
        assert_eq!(sample.valid_zones, Some(26));
        assert_eq!(sample.zone_count, 3);
        assert_eq!(&sample.zones[..3], &[405, 0, 386]);
        assert_eq!(sample.zones[3], 0);
        assert!(sample.sampled_at <= sample.received_at);
    }

    #[test]
    fn parses_full_zone_payload() {
        let received_at = Instant::now();
        let sample = parse_sample(
            "tof seq=166836 t_ms=11004765 irq_us=2414782186 res=64 center_mm=1784 min_mm=736 max_mm=2796 valid=24 zones=0,0,1799,1843,1803,1780,1755,1766,0,0,2059,0,2085,2045,2002,1994,0,0,0,2288,0,0,2387,2284,0,0,0,2796,0,2556,2515,2376,0,0,0,0,0,0,0,0,0,0,0,0,0,0,798,0,0,0,0,0,0,862,745,783,0,0,0,0,0,768,736,0",
            received_at,
        )
        .unwrap()
        .unwrap();

        assert_eq!(sample.sequence, Some(166_836));
        assert_eq!(sample.printed_at_ms, Some(11_004_765));
        assert_eq!(sample.sensor_timestamp_us, Some(2_414_782_186));
        assert_eq!(sample.resolution, Some(64));
        assert_eq!(sample.center_mm, Some(1784));
        assert_eq!(sample.min_mm, Some(736));
        assert_eq!(sample.max_mm, Some(2796));
        assert_eq!(sample.valid_zones, Some(24));
        assert_eq!(sample.zone_count, 64);
        assert_eq!(sample.zones[2], 1799);
        assert_eq!(sample.zones[27], 2796);
        assert_eq!(sample.zones[62], 736);
    }

    #[test]
    fn skips_status_and_empty_depth() {
        let received_at = Instant::now();
        assert!(
            parse_sample("tof status=ready res=64 hz=15 int_pin=16", received_at)
                .unwrap()
                .is_none()
        );
        assert!(
            parse_sample(
                "tof seq=1 t_ms=1 irq_us=2 res=64 center_mm=0 min_mm=0 max_mm=0 valid=0 zones=0",
                received_at,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn rejects_more_than_64_zones() {
        let zones = std::iter::repeat_n("1", 65).collect::<Vec<_>>().join(",");
        let line = format!(
            "tof seq=1 t_ms=1 irq_us=2 res=64 center_mm=1 min_mm=1 max_mm=1 valid=65 zones={zones}"
        );

        assert!(parse_sample(&line, Instant::now()).is_err());
    }
}
