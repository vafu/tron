use super::HandLandmarker;
use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, Handedness, RectNorm, Vec3};
use std::time::Instant;

/// Generates a deterministic, animated hand for bring-up and tests. Renders an
/// "open palm" by default, with a configurable pose so the gesture path can be
/// exercised without a model file.
pub struct MockLandmarker {
    pub pose: MockPose,
    pub center: (f32, f32),
    pub scale: f32,
}

#[derive(Clone, Copy)]
pub enum MockPose {
    Open,
    Fist,
    Pinch,
    Point,
    ThumbsUp,
}

impl MockLandmarker {
    pub fn new() -> Self {
        Self {
            pose: MockPose::Open,
            center: (0.5, 0.5),
            scale: 0.30,
        }
    }
}

impl Default for MockLandmarker {
    fn default() -> Self {
        Self::new()
    }
}

impl HandLandmarker for MockLandmarker {
    fn run(&mut self, ctx: &FrameContext, _roi: Option<RectNorm>) -> Option<HandLandmarks> {
        // Slow drift so the renderer shows motion.
        let t = ctx.now.elapsed().as_secs_f32();
        let cx = self.center.0 + 0.05 * (t * 0.3).cos();
        let cy = self.center.1 + 0.05 * (t * 0.3).sin();
        let pts = canonical_pose(self.pose, cx, cy, self.scale);
        Some(HandLandmarks {
            points: pts,
            presence: 1.0,
            handedness: Handedness::Right,
            timestamp: Instant::now(),
        })
    }
}

/// MediaPipe Hands index → name (informational):
///   0 wrist
///   1..4   thumb (cmc, mcp, ip, tip)
///   5..8   index (mcp, pip, dip, tip)
///   9..12  middle
///  13..16  ring
///  17..20  pinky
fn canonical_pose(pose: MockPose, cx: f32, cy: f32, s: f32) -> [Vec3; 21] {
    // Base pose: open hand, palm facing camera, fingers up. Coordinates in a
    // local frame around (cx, cy); scale `s` is roughly hand height.
    // (x, y) — y grows downward (image coords).
    let f = |x: f32, y: f32| Vec3 {
        x: cx + x * s,
        y: cy + y * s,
        z: 0.0,
    };
    // Open palm template.
    let mut p = [
        f(0.00, 0.50),   // 0  wrist
        f(-0.30, 0.40),  // 1  thumb cmc
        f(-0.45, 0.20),  // 2  thumb mcp
        f(-0.55, 0.05),  // 3  thumb ip
        f(-0.62, -0.05), // 4  thumb tip
        f(-0.20, 0.10),  // 5  index mcp
        f(-0.22, -0.10), // 6  index pip
        f(-0.23, -0.30), // 7  index dip
        f(-0.24, -0.50), // 8  index tip
        f(0.00, 0.05),   // 9  middle mcp
        f(0.00, -0.18),  // 10 middle pip
        f(0.00, -0.40),  // 11 middle dip
        f(0.00, -0.60),  // 12 middle tip
        f(0.18, 0.10),   // 13 ring mcp
        f(0.20, -0.10),  // 14 ring pip
        f(0.21, -0.28),  // 15 ring dip
        f(0.22, -0.46),  // 16 ring tip
        f(0.32, 0.18),   // 17 pinky mcp
        f(0.36, 0.02),   // 18 pinky pip
        f(0.38, -0.14),  // 19 pinky dip
        f(0.40, -0.30),  // 20 pinky tip
    ];

    // Curl helpers: collapse a finger's three distal joints onto its mcp.
    let curl = |p: &mut [Vec3; 21], mcp: usize, pip: usize, dip: usize, tip: usize| {
        let m = p[mcp];
        let toward = |a: Vec3, t: f32| Vec3 {
            x: a.x * (1.0 - t) + m.x * t,
            y: a.y * (1.0 - t) + m.y * t,
            z: 0.0,
        };
        p[pip] = toward(p[pip], 0.6);
        p[dip] = toward(p[dip], 0.85);
        p[tip] = toward(p[tip], 0.95);
    };

    match pose {
        MockPose::Open => {}
        MockPose::Fist => {
            curl(&mut p, 5, 6, 7, 8);
            curl(&mut p, 9, 10, 11, 12);
            curl(&mut p, 13, 14, 15, 16);
            curl(&mut p, 17, 18, 19, 20);
        }
        MockPose::Pinch => {
            // index tip and thumb tip meet.
            let mid = Vec3 {
                x: (p[4].x + p[8].x) * 0.5,
                y: (p[4].y + p[8].y) * 0.5,
                z: 0.0,
            };
            p[4] = mid;
            p[8] = mid;
        }
        MockPose::Point => {
            // index extended, others curled.
            curl(&mut p, 9, 10, 11, 12);
            curl(&mut p, 13, 14, 15, 16);
            curl(&mut p, 17, 18, 19, 20);
        }
        MockPose::ThumbsUp => {
            curl(&mut p, 5, 6, 7, 8);
            curl(&mut p, 9, 10, 11, 12);
            curl(&mut p, 13, 14, 15, 16);
            curl(&mut p, 17, 18, 19, 20);
            // thumb pointing up (negative y).
            p[4] = Vec3 {
                x: cx - 0.30 * s,
                y: cy - 0.40 * s,
                z: 0.0,
            };
        }
    }
    p
}
