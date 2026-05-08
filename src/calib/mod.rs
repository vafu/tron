use crate::types::RectNorm;
use std::sync::RwLock;

/// Quick-and-dirty IR↔RGB camera registration.
///
/// Both cameras share a module but expose different sensor crops:
///   RGB: 640×480 (4:3)
///   IR : 640×360 (16:9)
///
/// First-pass guess: same horizontal sensor coverage; IR is the central
/// horizontal strip of an equivalent vertical extent → vertical letterbox of
/// (480-360)/2 = 60 px on each side ⇒ scale_y = 360/480 = 0.75, offset_y =
/// 60/480 = 0.125. Live-tunable at runtime via keyboard (see `main.rs`); the
/// final values can then be pasted back into `DEFAULT` for persistence.
pub const DEFAULT: AffineCalib = AffineCalib {
    scale_x: 1.22,
    scale_y: 0.89,
    offset_x: -0.12,
    offset_y: 0.03,
    use_binary: false,
};

static IR_TO_RGB: RwLock<AffineCalib> = RwLock::new(DEFAULT);

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

pub fn reset() {
    *IR_TO_RGB.write().unwrap() = DEFAULT;
    eprintln!("calib: reset to default");
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
}
