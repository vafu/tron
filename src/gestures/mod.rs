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
    last_gesture: Option<Gesture>,
}

impl RuleBasedClassifier {
    pub fn new() -> Self {
        Self {
            pinch_threshold: 0.20,
            last_gesture: None,
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
            self.last_gesture = None;
            return GestureClassification::default();
        }
        let p = &lm.points;
        let palm = palm_size(p).max(1e-3);

        let pinch = dist(p[4], p[8]) / palm;
        let index = finger_state(p, 5, 6, 8, palm);
        let middle = finger_state(p, 9, 10, 12, palm);
        let ring = finger_state(p, 13, 14, 16, palm);
        let pinky = finger_state(p, 17, 18, 20, palm);

        let extended = [
            index.extended,
            middle.extended,
            ring.extended,
            pinky.extended,
        ];
        let n_ext = extended.iter().filter(|x| **x).count();
        let curled = [index.curled, middle.curled, ring.curled, pinky.curled];
        let n_curled = curled.iter().filter(|x| **x).count();
        let curl_mean = (index.curl + middle.curl + ring.curl + pinky.curl) * 0.25;
        let compactness = fist_compactness(p, palm);
        let fist_score = (curl_mean * 0.70 + compactness * 0.30).clamp(0.0, 1.0);
        let thumb_up = n_curled >= 3 && is_thumb_up(p, palm);
        let features = GestureFeatures {
            pinch,
            extended: n_ext as u8,
            curled: n_curled as u8,
            thumb_up,
            fist_score,
            index_curl: index.curl,
            middle_curl: middle.curl,
            ring_curl: ring.curl,
            pinky_curl: pinky.curl,
        };

        let fist_enter = fist_score >= 0.78 && n_curled >= 3 && n_ext <= 1;
        let fist_stay =
            self.last_gesture == Some(Gesture::Fist) && fist_score >= 0.64 && n_curled >= 2;
        let pinch_enter = pinch <= self.pinch_threshold
            && index.curl <= 0.78
            && middle.curl <= 0.88
            && !(n_curled >= 3 && fist_score >= 0.72);
        let pinch_stay = self.last_gesture == Some(Gesture::Pinch)
            && pinch <= self.pinch_threshold * 1.35
            && !(n_curled >= 3 && fist_score >= 0.76);
        let gesture = if fist_enter || fist_stay {
            Some(Gesture::Fist)
        } else if pinch_enter || pinch_stay {
            Some(Gesture::Pinch)
        } else if n_ext >= 3 && fist_score <= 0.42 {
            Some(Gesture::Open)
        } else if index.extended
            && middle.curled
            && ring.curled
            && pinky.curled
            && fist_score <= 0.70
        {
            Some(Gesture::Point)
        } else {
            Some(Gesture::Unknown)
        };
        self.last_gesture = gesture;
        GestureClassification { gesture, features }
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

#[derive(Clone, Copy)]
struct FingerState {
    curl: f32,
    curled: bool,
    extended: bool,
}

fn finger_state(p: &[Vec3; 21], mcp: usize, pip: usize, tip: usize, palm: f32) -> FingerState {
    let wrist = p[0];
    let tip_wrist = dist(p[tip], wrist);
    let pip_wrist = dist(p[pip], wrist).max(1e-3);
    let tip_mcp = dist(p[tip], p[mcp]) / palm;
    let reach = tip_wrist / pip_wrist;
    let curl_by_reach = ((1.18 - reach) / 0.45).clamp(0.0, 1.0);
    let curl_by_fold = ((0.78 - tip_mcp) / 0.45).clamp(0.0, 1.0);
    let curl = (curl_by_reach * 0.60 + curl_by_fold * 0.40).clamp(0.0, 1.0);
    FingerState {
        curl,
        curled: curl >= 0.58,
        extended: reach >= 1.16 && curl <= 0.35,
    }
}

fn is_thumb_up(p: &[Vec3; 21], palm: f32) -> bool {
    // Image y grows downward. Require a distinctly vertical thumb, not just a
    // thumb that happens to sit slightly above the wrist in a closed fist.
    p[4].y < p[2].y - palm * 0.20 && p[4].y < p[0].y - palm * 0.25
}

fn fist_compactness(p: &[Vec3; 21], palm: f32) -> f32 {
    let center = palm_center(p);
    let tips = [8, 12, 16, 20];
    let mean_tip_dist = tips.iter().map(|&i| dist(p[i], center)).sum::<f32>() / tips.len() as f32;
    (1.15 - mean_tip_dist / palm).clamp(0.0, 1.0)
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
