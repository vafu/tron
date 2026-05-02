use crate::pipeline::FrameContext;
use crate::types::{Gesture, GestureClassification, GestureFeatures, HandLandmarks, Vec3};

/// Gesture classification stage.
///
/// Classifiers consume fully processed landmarks and may use any contextual
/// sensor state in `FrameContext`; replacing rules with a learned model should
/// only require swapping this trait object in `main.rs`.
pub trait GestureClassifier: Send {
    fn classify(&mut self, ctx: &FrameContext, lm: &HandLandmarks) -> GestureClassification;
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
    fn classify(&mut self, _ctx: &FrameContext, lm: &HandLandmarks) -> GestureClassification {
        if lm.presence < 0.5 {
            return GestureClassification::default();
        }
        let p = &lm.points;
        let palm = palm_size(p).max(1e-3);

        let pinch = dist(p[4], p[8]) / palm;
        let index_ext = is_extended(p, 5, 6, 7, 8);
        let middle_ext = is_extended(p, 9, 10, 11, 12);
        let ring_ext = is_extended(p, 13, 14, 15, 16);
        let pinky_ext = is_extended(p, 17, 18, 19, 20);
        let index_curled = is_curled(p, 5, 8, palm);
        let middle_curled = is_curled(p, 9, 12, palm);
        let ring_curled = is_curled(p, 13, 16, palm);
        let pinky_curled = is_curled(p, 17, 20, palm);

        let extended = [index_ext, middle_ext, ring_ext, pinky_ext];
        let n_ext = extended.iter().filter(|x| **x).count();
        let curled = [index_curled, middle_curled, ring_curled, pinky_curled];
        let n_curled = curled.iter().filter(|x| **x).count();
        let fist_score = fist_score(p, palm, &curled);
        let thumb_up = n_curled >= 3 && is_thumb_up(p, palm);
        let features = GestureFeatures {
            pinch,
            extended: n_ext as u8,
            curled: n_curled as u8,
            thumb_up,
            fist_score,
        };

        // Demo priority: make grab reliable first. A closed hand wins over
        // incidental thumb/index proximity, which currently makes pinch noisy.
        if fist_score >= 0.72 {
            return GestureClassification {
                gesture: Some(Gesture::Fist),
                features,
            };
        }
        if n_ext == 4 && fist_score < 0.35 {
            return GestureClassification {
                gesture: Some(Gesture::Open),
                features,
            };
        }
        if index_ext && !middle_ext && !ring_ext && !pinky_ext && fist_score < 0.55 {
            return GestureClassification {
                gesture: Some(Gesture::Point),
                features,
            };
        }
        if pinch < self.pinch_threshold && fist_score < 0.50 {
            return GestureClassification {
                gesture: Some(Gesture::Pinch),
                features,
            };
        }
        GestureClassification {
            gesture: Some(Gesture::Unknown),
            features,
        }
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

fn is_thumb_up(p: &[Vec3; 21], palm: f32) -> bool {
    // Image y grows downward. Require a distinctly vertical thumb, not just a
    // thumb that happens to sit slightly above the wrist in a closed fist.
    p[4].y < p[2].y - palm * 0.20 && p[4].y < p[0].y - palm * 0.25
}

fn is_curled(p: &[Vec3; 21], mcp: usize, tip: usize, palm: f32) -> bool {
    let center = palm_center(p);
    dist(p[tip], center) < dist(p[mcp], center) + palm * 0.55
}

fn fist_score(p: &[Vec3; 21], palm: f32, curled: &[bool; 4]) -> f32 {
    let center = palm_center(p);
    let curl_count = curled.iter().filter(|x| **x).count() as f32 / curled.len() as f32;
    let tips = [8, 12, 16, 20];
    let mean_tip_dist = tips.iter().map(|&i| dist(p[i], center)).sum::<f32>() / tips.len() as f32;
    let compactness = (1.25 - mean_tip_dist / palm).clamp(0.0, 1.0);
    (curl_count * 0.65 + compactness * 0.35).clamp(0.0, 1.0)
}

fn palm_center(p: &[Vec3; 21]) -> Vec3 {
    let ids = [0, 5, 9, 13, 17];
    let mut c = Vec3::default();
    for i in ids {
        c.x += p[i].x;
        c.y += p[i].y;
        c.z += p[i].z;
    }
    Vec3 {
        x: c.x / ids.len() as f32,
        y: c.y / ids.len() as f32,
        z: c.z / ids.len() as f32,
    }
}
