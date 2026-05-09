use crate::types::{HandLandmarks, Vec3};
use std::time::Instant;

/// Landmark post-processing stage.
///
/// Filters should preserve the landmark coordinate contract and only adjust
/// temporal stability, confidence, or equivalent post-inference attributes.
pub trait LandmarkFilter: Send {
    fn apply(&mut self, lm: HandLandmarks) -> HandLandmarks;
}

pub struct Identity;

impl LandmarkFilter for Identity {
    fn apply(&mut self, lm: HandLandmarks) -> HandLandmarks {
        lm
    }
}

/// One-Euro filter — temporal smoothing whose cutoff scales with speed, so
/// jitter is suppressed when still and lag is small when moving.
/// See Casiez et al., "1€ Filter".
pub struct OneEuroFilter {
    pub min_cutoff: f32, // Hz
    pub beta: f32,       // speed coefficient
    pub d_cutoff: f32,   // Hz, derivative LPF cutoff
    state: Option<State>,
}

struct State {
    last: Instant,
    x: [Vec3; 21],
    dx: [Vec3; 21],
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            state: None,
        }
    }
}

impl Default for OneEuroFilter {
    fn default() -> Self {
        Self::new(1.0, 0.05, 1.0)
    }
}

impl LandmarkFilter for OneEuroFilter {
    fn apply(&mut self, mut lm: HandLandmarks) -> HandLandmarks {
        let now = lm.timestamp;
        let s = match self.state.as_mut() {
            None => {
                self.state = Some(State {
                    last: now,
                    x: lm.points,
                    dx: [Vec3::default(); 21],
                });
                return lm;
            }
            Some(s) => s,
        };
        let dt = (now - s.last).as_secs_f32().max(1e-3);
        s.last = now;

        for i in 0..21 {
            let p = lm.points[i];
            let prev = s.x[i];
            // velocity estimate, low-pass with d_cutoff
            let raw_dx = Vec3 {
                x: (p.x - prev.x) / dt,
                y: (p.y - prev.y) / dt,
                z: (p.z - prev.z) / dt,
            };
            let a_d = alpha(self.d_cutoff, dt);
            let dx = lerp_v3(s.dx[i], raw_dx, a_d);
            s.dx[i] = dx;

            let cutoff = self.min_cutoff + self.beta * speed(dx);
            let a = alpha(cutoff, dt);
            let smoothed = lerp_v3(prev, p, a);
            s.x[i] = smoothed;
            lm.points[i] = smoothed;
        }
        lm
    }
}

fn alpha(cutoff_hz: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    1.0 / (1.0 + tau / dt)
}

fn lerp_v3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3 {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
    }
}

fn speed(v: Vec3) -> f32 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}
