use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8,
    R8,
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Rgba8 => 4,
            PixelFormat::R8 => 1,
        }
    }
}

#[derive(Clone)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub timestamp: Instant,
    /// Monotonic frame counter from the producing source.
    pub seq: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RectNorm {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl RectNorm {
    pub const FULL: RectNorm = RectNorm { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };

    pub fn padded(self, frac: f32) -> Self {
        let dx = self.w * frac;
        let dy = self.h * frac;
        RectNorm {
            x: (self.x - dx).max(0.0),
            y: (self.y - dy).max(0.0),
            w: (self.w + 2.0 * dx).min(1.0 - (self.x - dx).max(0.0)),
            h: (self.h + 2.0 * dy).min(1.0 - (self.y - dy).max(0.0)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
    Unknown,
}

#[derive(Clone)]
pub struct HandLandmarks {
    /// 21 keypoints in normalized 0..1 source-image coords; z is relative depth.
    pub points: [Vec3; 21],
    pub presence: f32,
    pub handedness: Handedness,
    pub timestamp: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gesture {
    Open,
    Fist,
    Pinch,
    Point,
    ThumbsUp,
    Unknown,
}

impl Gesture {
    pub fn name(self) -> &'static str {
        match self {
            Gesture::Open => "open",
            Gesture::Fist => "fist",
            Gesture::Pinch => "pinch",
            Gesture::Point => "point",
            Gesture::ThumbsUp => "thumbs-up",
            Gesture::Unknown => "?",
        }
    }
}

#[derive(Clone)]
pub struct HandState {
    pub roi: Option<RectNorm>,
    pub landmarks: Option<HandLandmarks>,
    pub gesture: Option<Gesture>,
}
