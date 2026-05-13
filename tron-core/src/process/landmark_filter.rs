use std::time::Instant;

use super::one_euro::OneEuroFilter;
use crate::roi::mediapipe::HandLandmarks;
use tron_api::{NoContext, Processor};

pub struct MediaPipeLandmarkFilter {
    filters: Vec<OneEuroFilter>,
    start_time: Instant,
}

impl MediaPipeLandmarkFilter {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        let mut filters = Vec::with_capacity(63);
        for _ in 0..63 {
            filters.push(OneEuroFilter::new(min_cutoff, beta, 1.0));
        }
        Self {
            filters,
            start_time: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

impl Processor<Option<HandLandmarks>> for MediaPipeLandmarkFilter {
    type Output = Option<HandLandmarks>;

    fn process(
        &mut self,
        input: Option<HandLandmarks>,
        _context: NoContext,
    ) -> anyhow::Result<Self::Output> {
        let Some(mut landmarks) = input else {
            self.reset();
            return Ok(None);
        };

        let t = (landmarks.timestamp - self.start_time).as_secs_f32();

        for i in 0..21 {
            let p = &mut landmarks.points[i];
            p.x = self.filters[i * 3].filter(p.x, t);
            p.y = self.filters[i * 3 + 1].filter(p.y, t);
            p.z = self.filters[i * 3 + 2].filter(p.z, t);
        }

        Ok(Some(landmarks))
    }
}

impl Default for MediaPipeLandmarkFilter {
    fn default() -> Self {
        // Reasonable defaults for hand tracking:
        // min_cutoff: 1.0 Hz (lower = more smoothing at rest)
        // beta: 0.05 (higher = less lag during movement)
        Self::new(1.0, 0.05)
    }
}
