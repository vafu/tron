pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    t_prev: f32,
    initialized: bool,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: 0.0,
            dx_prev: 0.0,
            t_prev: 0.0,
            initialized: false,
        }
    }

    pub fn filter(&mut self, x: f32, t: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.t_prev = t;
            self.initialized = true;
            return x;
        }

        let dt = t - self.t_prev;
        if dt <= 0.0 {
            return self.x_prev;
        }

        let dx = (x - self.x_prev) / dt;
        let edx = self.low_pass(dx, self.dx_prev, self.alpha(dt, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        let result = self.low_pass(x, self.x_prev, self.alpha(dt, cutoff));

        self.x_prev = result;
        self.dx_prev = edx;
        self.t_prev = t;

        result
    }

    fn alpha(&self, dt: f32, cutoff: f32) -> f32 {
        let r = 2.0 * std::f32::consts::PI * cutoff * dt;
        r / (r + 1.0)
    }

    fn low_pass(&self, x: f32, prev: f32, alpha: f32) -> f32 {
        alpha * x + (1.0 - alpha) * prev
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }
}
