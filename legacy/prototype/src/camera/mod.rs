use crate::types::Image;
use anyhow::Result;
use std::sync::{Arc, Mutex};

pub mod v4l;

pub type SharedImage = Arc<Mutex<Option<Image>>>;

#[derive(Clone, Debug)]
pub struct CameraSet {
    pub label: String,
    pub rgb: StreamConfig,
    pub ir: StreamConfig,
}

#[derive(Clone, Copy, Debug)]
pub enum StreamFormat {
    Rgb,
    Ir,
}

#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub format: StreamFormat,
    pub fps: Option<u32>,
}

impl StreamConfig {
    pub fn rgb(path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            format: StreamFormat::Rgb,
            fps: None,
        }
    }

    pub fn ir(path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            format: StreamFormat::Ir,
            fps: None,
        }
    }

    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = Some(fps);
        self
    }
}

/// Camera backend boundary.
///
/// The rest of the app consumes `SharedImage` and `StreamConfig`; backend
/// details such as V4L, libcamera, or IPU6-specific setup should live behind
/// this trait.
pub trait CameraBackend {
    fn spawn_stream(&self, config: StreamConfig) -> Result<SharedImage>;
}

pub fn spawn_stream<B: CameraBackend>(backend: &B, config: StreamConfig) -> Result<SharedImage> {
    backend.spawn_stream(config)
}

pub fn spawn_config(config: StreamConfig) -> Result<SharedImage> {
    spawn_stream(&v4l::Backend, config)
}

pub fn spawn_rgb(path: &str, width: u32, height: u32) -> Result<SharedImage> {
    spawn_stream(&v4l::Backend, StreamConfig::rgb(path, width, height))
}

pub fn spawn_ir(path: &str, width: u32, height: u32) -> Result<SharedImage> {
    spawn_stream(&v4l::Backend, StreamConfig::ir(path, width, height))
}

pub mod select {
    use super::{CameraSet, StreamConfig};
    use anyhow::{Context, Result, anyhow};
    use std::collections::BTreeMap;
    use v4l::capability::Flags;
    use v4l::context;
    use v4l::frameinterval::FrameIntervalEnum;
    use v4l::framesize::FrameSizeEnum;
    use v4l::video::Capture;
    use v4l::{Device, FourCC};

    const RGB_FOURCC: &[u8; 4] = b"YUYV";
    const IR_FOURCC: &[u8; 4] = b"GREY";
    const RGB_TARGET_PIXELS: u64 = 960 * 540;
    const IR_TARGET_PIXELS: u64 = 640 * 480;
    const RGB_TARGET_FPS: u32 = 30;
    const IR_TARGET_FPS: u32 = 30;

    #[derive(Clone, Debug)]
    struct Candidate {
        path: String,
        card: String,
        bus: String,
        formats: Vec<FormatInfo>,
    }

    #[derive(Clone, Copy, Debug)]
    struct FormatInfo {
        fourcc: FourCC,
        width: u32,
        height: u32,
        max_fps: Option<u32>,
    }

    pub fn by_name(query: &str) -> Result<CameraSet> {
        let query_norm = query.to_lowercase();
        let mut candidates = enumerate_candidates();
        candidates.retain(|c| {
            c.card.to_lowercase().contains(&query_norm)
                || c.bus.to_lowercase().contains(&query_norm)
        });

        if candidates.is_empty() {
            return Err(anyhow!(
                "no V4L camera nodes matched {query:?}; available: {}",
                available_summary()
            ));
        }

        let rgb_configs = candidates
            .iter()
            .flat_map(|c| stream_configs(c, RGB_FOURCC, true))
            .collect::<Vec<_>>();
        if rgb_configs.is_empty() {
            return Err(anyhow!(
                "camera {query:?} matched nodes but none expose YUYV RGB capture; matched: {}",
                candidate_summary(&candidates)
            ));
        }

        let ir_configs = candidates
            .iter()
            .flat_map(|c| stream_configs(c, IR_FOURCC, false))
            .collect::<Vec<_>>();
        if ir_configs.is_empty() {
            return Err(anyhow!(
                "camera {query:?} matched nodes but none expose GREY IR capture; matched: {}",
                candidate_summary(&candidates)
            ));
        }

        let rgb = choose_stream(&rgb_configs, RGB_TARGET_PIXELS).ok_or_else(|| {
            anyhow!(
                "camera {query:?} matched nodes but no usable RGB mode was found; matched: {}",
                candidate_summary(&candidates)
            )
        })?;
        let ir = choose_stream(&ir_configs, IR_TARGET_PIXELS).ok_or_else(|| {
            anyhow!(
                "camera {query:?} matched nodes but no usable IR mode was found; matched: {}",
                candidate_summary(&candidates)
            )
        })?;

        let label = candidates
            .first()
            .map(|c| c.card.clone())
            .unwrap_or_else(|| query.to_string());

        Ok(CameraSet { label, rgb, ir })
    }

    pub fn default_set() -> CameraSet {
        CameraSet {
            label: "default".to_string(),
            rgb: StreamConfig::rgb("/dev/video0", 640, 480).with_fps(RGB_TARGET_FPS),
            ir: StreamConfig::ir("/dev/video2", 640, 360).with_fps(IR_TARGET_FPS),
        }
    }

    fn enumerate_candidates() -> Vec<Candidate> {
        let mut out = Vec::new();
        for node in context::enum_devices() {
            let path = node.path().to_string_lossy().to_string();
            let Ok(dev) = Device::with_path(&path) else {
                continue;
            };
            let Ok(caps) = dev.query_caps() else {
                continue;
            };
            if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
                continue;
            }
            let Ok(formats) = formats(&dev) else {
                continue;
            };
            if formats.is_empty() {
                continue;
            }
            out.push(Candidate {
                path,
                card: caps.card,
                bus: caps.bus,
                formats,
            });
        }
        out.sort_by_key(|c| video_index(&c.path));
        out
    }

    fn formats(dev: &Device) -> Result<Vec<FormatInfo>> {
        let mut out = Vec::new();
        for desc in dev.enum_formats().context("enumerate formats")? {
            for size in dev
                .enum_framesizes(desc.fourcc)
                .with_context(|| format!("enumerate frame sizes for {}", desc.fourcc))?
            {
                match size.size {
                    FrameSizeEnum::Discrete(d) => out.push(FormatInfo {
                        fourcc: desc.fourcc,
                        width: d.width,
                        height: d.height,
                        max_fps: max_fps(dev, desc.fourcc, d.width, d.height),
                    }),
                    FrameSizeEnum::Stepwise(s) => out.push(FormatInfo {
                        fourcc: desc.fourcc,
                        width: s.min_width,
                        height: s.min_height,
                        max_fps: max_fps(dev, desc.fourcc, s.min_width, s.min_height),
                    }),
                }
            }
        }
        Ok(out)
    }

    fn max_fps(dev: &Device, fourcc: FourCC, width: u32, height: u32) -> Option<u32> {
        dev.enum_frameintervals(fourcc, width, height)
            .ok()?
            .iter()
            .filter_map(|interval| match &interval.interval {
                FrameIntervalEnum::Discrete(frac) => {
                    fps_from_interval(frac.numerator, frac.denominator)
                }
                FrameIntervalEnum::Stepwise(stepwise) => {
                    fps_from_interval(stepwise.min.numerator, stepwise.min.denominator)
                }
            })
            .max()
    }

    fn fps_from_interval(numerator: u32, denominator: u32) -> Option<u32> {
        if numerator == 0 {
            return None;
        }
        Some(((denominator as f64 / numerator as f64) + 0.5) as u32)
    }

    fn stream_configs(candidate: &Candidate, fourcc: &[u8; 4], rgb: bool) -> Vec<StreamConfig> {
        let wanted = FourCC::new(fourcc);
        let target_fps = if rgb { RGB_TARGET_FPS } else { IR_TARGET_FPS };
        let mut sizes: Vec<(u32, u32, Option<u32>)> = candidate
            .formats
            .iter()
            .filter(|f| f.fourcc == wanted)
            .map(|f| (f.width, f.height, f.max_fps))
            .collect();
        sizes.sort_unstable();
        sizes.dedup();

        sizes
            .into_iter()
            .map(|(width, height, max_fps)| {
                let fps = max_fps.map(|fps| fps.min(target_fps));
                if rgb {
                    with_optional_fps(
                        StreamConfig::rgb(candidate.path.clone(), width, height),
                        fps,
                    )
                } else {
                    with_optional_fps(StreamConfig::ir(candidate.path.clone(), width, height), fps)
                }
            })
            .collect()
    }

    fn with_optional_fps(config: StreamConfig, fps: Option<u32>) -> StreamConfig {
        match fps {
            Some(fps) => config.with_fps(fps),
            None => config,
        }
    }

    fn choose_stream(configs: &[StreamConfig], target_pixels: u64) -> Option<StreamConfig> {
        let native = configs.iter().max_by_key(|cfg| pixels(cfg))?;
        configs
            .iter()
            .min_by_key(|cfg| stream_score(cfg, native, target_pixels))
            .cloned()
    }

    fn stream_score(
        cfg: &StreamConfig,
        native: &StreamConfig,
        target_pixels: u64,
    ) -> (u64, u64, u64, u32) {
        (
            fps_penalty(cfg, target_pixels),
            pixels(cfg).abs_diff(target_pixels),
            aspect_diff_ppm(cfg, native),
            video_index(&cfg.path),
        )
    }

    fn fps_penalty(cfg: &StreamConfig, target_pixels: u64) -> u64 {
        let target_fps = match cfg.format {
            super::StreamFormat::Rgb => RGB_TARGET_FPS,
            super::StreamFormat::Ir => IR_TARGET_FPS,
        };
        let fps = cfg.fps.unwrap_or(0);
        let shortfall = target_fps.saturating_sub(fps) as u64;
        if shortfall == 0 {
            return 0;
        }
        shortfall * target_pixels
    }

    fn aspect_diff_ppm(a: &StreamConfig, b: &StreamConfig) -> u64 {
        let lhs = a.width as u64 * b.height as u64 * 1_000_000;
        let rhs = b.width as u64 * a.height as u64 * 1_000_000;
        lhs.abs_diff(rhs) / (a.height as u64 * b.height as u64).max(1)
    }

    fn pixels(cfg: &StreamConfig) -> u64 {
        cfg.width as u64 * cfg.height as u64
    }

    pub fn available_summary() -> String {
        let candidates = enumerate_candidates();
        if candidates.is_empty() {
            return "<none>".to_string();
        }
        let mut groups: BTreeMap<String, Vec<Candidate>> = BTreeMap::new();
        for c in candidates {
            groups
                .entry(format!("{} [{}]", c.card, c.bus))
                .or_default()
                .push(c);
        }

        let mut sets = Vec::new();
        for (label, candidates) in groups {
            let rgb_configs = candidates
                .iter()
                .flat_map(|c| stream_configs(c, RGB_FOURCC, true))
                .collect::<Vec<_>>();
            let ir_configs = candidates
                .iter()
                .flat_map(|c| stream_configs(c, IR_FOURCC, false))
                .collect::<Vec<_>>();
            if let (Some(rgb), Some(ir)) = (
                choose_stream(&rgb_configs, RGB_TARGET_PIXELS),
                choose_stream(&ir_configs, IR_TARGET_PIXELS),
            ) {
                sets.push(format!(
                    "{label}: rgb={} {}x{}{}, ir={} {}x{}{}",
                    rgb.path,
                    rgb.width,
                    rgb.height,
                    fps_suffix(rgb.fps),
                    ir.path,
                    ir.width,
                    ir.height,
                    fps_suffix(ir.fps)
                ));
            }
        }

        if sets.is_empty() {
            return "no usable RGB+IR camera sets found".to_string();
        }
        sets.join("\n")
    }

    pub fn available_summary_detailed() -> String {
        let candidates = enumerate_candidates();
        if candidates.is_empty() {
            return "<none>".to_string();
        }

        let mut out = Vec::new();
        out.push("selected sets:".to_string());
        out.push(available_summary());
        out.push(String::new());
        out.push("capture nodes:".to_string());
        for c in candidates {
            let mut formats = c
                .formats
                .iter()
                .map(|f| {
                    format!(
                        "{}:{}x{}{}",
                        f.fourcc,
                        f.width,
                        f.height,
                        fps_suffix(f.max_fps)
                    )
                })
                .collect::<Vec<_>>();
            formats.sort();
            formats.dedup();
            out.push(format!(
                "{} [{}; {}] {}",
                c.path,
                c.card,
                c.bus,
                formats.join(", ")
            ));
        }
        out.join("\n")
    }

    fn candidate_summary(candidates: &[Candidate]) -> String {
        candidates
            .iter()
            .map(|c| {
                let formats = c
                    .formats
                    .iter()
                    .map(|f| {
                        format!(
                            "{}:{}x{}{}",
                            f.fourcc,
                            f.width,
                            f.height,
                            fps_suffix(f.max_fps)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{} [{}] ({formats})", c.path, c.card)
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn video_index(path: &str) -> u32 {
        path.rsplit_once("video")
            .and_then(|(_, n)| n.parse().ok())
            .unwrap_or(u32::MAX)
    }

    fn fps_suffix(fps: Option<u32>) -> String {
        fps.map(|fps| format!("@{fps}fps")).unwrap_or_default()
    }
}
