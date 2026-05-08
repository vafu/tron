use crate::types::RectNorm;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub mod checkerboard;
pub mod stereo;

/// Quick-and-dirty IR↔RGB camera registration. `AffineCalib` maps normalized IR
/// coordinates into normalized RGB coordinates; `unmap_*` does the reverse.
///
/// This is intentionally a first-order model. It handles translation and scale
/// while we build a real measured RGB↔IR calibration workflow.
pub const FALLBACK: AffineCalib = AffineCalib {
    scale_x: 1.0,
    scale_y: 1.0,
    offset_x: 0.0,
    offset_y: 0.0,
    use_binary: false,
};

#[derive(Clone, Debug)]
struct ActiveProfile {
    camera_label: String,
    path: PathBuf,
    default: AffineCalib,
}

static IR_TO_RGB: RwLock<AffineCalib> = RwLock::new(FALLBACK);
static PROFILE: RwLock<Option<ActiveProfile>> = RwLock::new(None);

pub fn init(camera_label: &str, rgb_size: (u32, u32), ir_size: (u32, u32)) {
    let default = default_from_sizes(rgb_size, ir_size);
    let path = profile_path(camera_label);
    let loaded = load_from_path(&path);
    let calib = loaded.unwrap_or(default);

    *IR_TO_RGB.write().unwrap() = calib;
    *PROFILE.write().unwrap() = Some(ActiveProfile {
        camera_label: camera_label.to_string(),
        path: path.clone(),
        default,
    });

    eprintln!(
        "calib: {} {} scale=({:.3}, {:.3}) offset=({:.3}, {:.3}) binary={}",
        if loaded.is_some() {
            "loaded"
        } else {
            "default"
        },
        path.display(),
        calib.scale_x,
        calib.scale_y,
        calib.offset_x,
        calib.offset_y,
        calib.use_binary
    );
}

pub fn current() -> AffineCalib {
    *IR_TO_RGB.read().unwrap()
}

pub fn modify(f: impl FnOnce(&mut AffineCalib)) {
    let mut c = IR_TO_RGB.write().unwrap();
    f(&mut c);
    eprintln!(
        "calib: scale=({:.3}, {:.3}) offset=({:.3}, {:.3}) binary={}",
        c.scale_x, c.scale_y, c.offset_x, c.offset_y, c.use_binary
    );
}

pub fn set(calib: AffineCalib) {
    *IR_TO_RGB.write().unwrap() = calib;
    eprintln!(
        "calib: set scale=({:.3}, {:.3}) offset=({:.3}, {:.3}) binary={}",
        calib.scale_x, calib.scale_y, calib.offset_x, calib.offset_y, calib.use_binary
    );
}

pub fn reset() {
    let default = PROFILE
        .read()
        .unwrap()
        .as_ref()
        .map(|p| p.default)
        .unwrap_or(FALLBACK);
    *IR_TO_RGB.write().unwrap() = default;
    eprintln!("calib: reset to profile default");
}

pub fn save() -> std::io::Result<()> {
    let calib = current();
    let profile = PROFILE.read().unwrap().clone();
    let Some(profile) = profile else {
        eprintln!("calib: no active profile");
        return Ok(());
    };
    if let Some(parent) = profile.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&profile.path, calib.to_text())?;
    eprintln!(
        "calib: saved {} for {}",
        profile.path.display(),
        profile.camera_label
    );
    Ok(())
}

pub fn save_stereo(text: &str) -> std::io::Result<()> {
    let profile = PROFILE.read().unwrap().clone();
    let Some(profile) = profile else {
        eprintln!("calib: no active profile for stereo save");
        return Ok(());
    };
    let path = profile.path.with_extension("stereo");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, text)?;
    eprintln!("calib: saved stereo {}", path.display());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct AffineCalib {
    pub scale_x: f32,
    pub scale_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub use_binary: bool,
}

impl AffineCalib {
    pub fn map_rect(&self, r: RectNorm) -> RectNorm {
        RectNorm {
            x: r.x * self.scale_x + self.offset_x,
            y: r.y * self.scale_y + self.offset_y,
            w: r.w * self.scale_x,
            h: r.h * self.scale_y,
        }
    }

    pub fn unmap_rect(&self, r: RectNorm) -> RectNorm {
        RectNorm {
            x: (r.x - self.offset_x) / self.scale_x,
            y: (r.y - self.offset_y) / self.scale_y,
            w: r.w / self.scale_x,
            h: r.h / self.scale_y,
        }
    }

    fn to_text(self) -> String {
        format!(
            "scale_x={:.8}\nscale_y={:.8}\noffset_x={:.8}\noffset_y={:.8}\nuse_binary={}\n",
            self.scale_x, self.scale_y, self.offset_x, self.offset_y, self.use_binary
        )
    }
}

fn default_from_sizes(rgb_size: (u32, u32), ir_size: (u32, u32)) -> AffineCalib {
    let rgb_ar = aspect(rgb_size);
    let ir_ar = aspect(ir_size);
    if rgb_ar <= 0.0 || ir_ar <= 0.0 {
        return FALLBACK;
    }

    if rgb_ar > ir_ar {
        let scale_x = ir_ar / rgb_ar;
        AffineCalib {
            scale_x,
            scale_y: 1.0,
            offset_x: (1.0 - scale_x) * 0.5,
            offset_y: 0.0,
            use_binary: false,
        }
    } else {
        let scale_y = rgb_ar / ir_ar;
        AffineCalib {
            scale_x: 1.0,
            scale_y,
            offset_x: 0.0,
            offset_y: (1.0 - scale_y) * 0.5,
            use_binary: false,
        }
    }
}

fn aspect(size: (u32, u32)) -> f32 {
    size.0 as f32 / size.1.max(1) as f32
}

fn profile_path(camera_label: &str) -> PathBuf {
    PathBuf::from("calib").join(format!("{}.calib", slug(camera_label)))
}

fn slug(label: &str) -> String {
    let mut out = String::new();
    for c in label.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "camera".to_string()
    } else {
        out
    }
}

fn load_from_path(path: &Path) -> Option<AffineCalib> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut calib = FALLBACK;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "scale_x" => calib.scale_x = value.parse().ok()?,
            "scale_y" => calib.scale_y = value.parse().ok()?,
            "offset_x" => calib.offset_x = value.parse().ok()?,
            "offset_y" => calib.offset_y = value.parse().ok()?,
            "use_binary" => calib.use_binary = value.parse().ok()?,
            _ => {}
        }
    }
    Some(calib)
}
