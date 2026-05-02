use crate::pipeline::FrameContext;
use crate::types::{Gesture, HandLandmarks, Vec3};

/// Gesture classification stage.
///
/// Classifiers consume fully processed landmarks and may use any contextual
/// sensor state in `FrameContext`; replacing rules with a learned model should
/// only require swapping this trait object in `main.rs`.
pub trait GestureClassifier: Send {
    fn classify(&mut self, ctx: &FrameContext, lm: &HandLandmarks) -> Option<Gesture>;
}

/// Geometric rules over the 21-keypoint topology. Cheap, deterministic, no
/// training data — a strong v1 baseline that's trivial to replace with an MLP
/// later via the same trait.
pub struct RuleBasedClassifier {
    pub pinch_threshold: f32, // normalized distance — fraction of palm size
}

impl RuleBasedClassifier {
    pub fn new() -> Self {
        Self {
            pinch_threshold: 0.20,
        }
    }
}

impl Default for RuleBasedClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl GestureClassifier for RuleBasedClassifier {
    fn classify(&mut self, _ctx: &FrameContext, lm: &HandLandmarks) -> Option<Gesture> {
        if lm.presence < 0.5 {
            return None;
        }
        let p = &lm.points;
        let palm = palm_size(p).max(1e-3);

        let pinch = dist(p[4], p[8]) / palm;
        let index_ext = is_extended(p, 5, 6, 7, 8);
        let middle_ext = is_extended(p, 9, 10, 11, 12);
        let ring_ext = is_extended(p, 13, 14, 15, 16);
        let pinky_ext = is_extended(p, 17, 18, 19, 20);
        let thumb_ext = is_thumb_extended(p);

        let extended = [index_ext, middle_ext, ring_ext, pinky_ext];
        let n_ext = extended.iter().filter(|x| **x).count();

        // Pinch dominates: thumb tip and index tip touching.
        if pinch < self.pinch_threshold && !middle_ext && !ring_ext && !pinky_ext {
            return Some(Gesture::Pinch);
        }
        // Thumbs up: only thumb extended, others curled, thumb roughly above wrist.
        if thumb_ext && n_ext == 0 && p[4].y < p[0].y {
            return Some(Gesture::ThumbsUp);
        }
        // Point: only index extended.
        if index_ext && !middle_ext && !ring_ext && !pinky_ext {
            return Some(Gesture::Point);
        }
        // Open palm: all four fingers extended.
        if n_ext == 4 {
            return Some(Gesture::Open);
        }
        // Fist: none extended.
        if n_ext == 0 && !thumb_ext {
            return Some(Gesture::Fist);
        }
        Some(Gesture::Unknown)
    }
}

fn palm_size(p: &[Vec3; 21]) -> f32 {
    // Wrist (0) to middle MCP (9) — a stable proxy for hand scale.
    dist(p[0], p[9])
}

fn dist(a: Vec3, b: Vec3) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// A finger is "extended" when its tip is meaningfully farther from the wrist
/// than its PIP joint is. Robust to global hand scale.
fn is_extended(p: &[Vec3; 21], _mcp: usize, pip: usize, _dip: usize, tip: usize) -> bool {
    let wrist = p[0];
    dist(p[tip], wrist) > dist(p[pip], wrist) * 1.10
}

fn is_thumb_extended(p: &[Vec3; 21]) -> bool {
    // Thumb runs ~lateral; use distance from wrist with a slightly looser threshold.
    let wrist = p[0];
    dist(p[4], wrist) > dist(p[2], wrist) * 1.10
}
