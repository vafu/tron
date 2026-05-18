use tron_api::Point2d;

#[derive(Clone, Copy, Debug)]
pub struct OneEuroConfig {
    pub min_cutoff: f64,
    pub beta: f64,
    pub derivative_cutoff: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuroScalar {
    previous_raw: Option<f64>,
    previous_filtered: Option<f64>,
    previous_derivative: Option<f64>,
}

impl OneEuroScalar {
    pub fn filter(&mut self, value: f64, dt: f64, config: OneEuroConfig) -> f64 {
        if !value.is_finite() {
            self.reset();
            return f64::NAN;
        }
        if dt <= 0.0 || !dt.is_finite() {
            self.previous_raw = Some(value);
            self.previous_filtered = Some(value);
            return value;
        }

        let Some(previous_raw) = self.previous_raw else {
            self.previous_raw = Some(value);
            self.previous_filtered = Some(value);
            self.previous_derivative = Some(0.0);
            return value;
        };

        let derivative = (value - previous_raw) / dt;
        let filtered_derivative = low_pass(
            derivative,
            self.previous_derivative.unwrap_or(derivative),
            alpha(config.derivative_cutoff, dt),
        );
        let cutoff = config.min_cutoff + config.beta * filtered_derivative.abs();
        let filtered = low_pass(
            value,
            self.previous_filtered.unwrap_or(value),
            alpha(cutoff.max(f64::EPSILON), dt),
        );

        self.previous_raw = Some(value);
        self.previous_filtered = Some(filtered);
        self.previous_derivative = Some(filtered_derivative);
        filtered
    }

    pub fn derivative(&self) -> Option<f64> {
        self.previous_derivative
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuroPoint2d {
    x: OneEuroScalar,
    y: OneEuroScalar,
}

impl OneEuroPoint2d {
    pub fn filter(&mut self, value: Point2d, dt: f64, config: OneEuroConfig) -> Point2d {
        Point2d::new(
            self.x.filter(value.x, dt, config),
            self.y.filter(value.y, dt, config),
        )
    }

    pub fn derivative(&self) -> Option<Point2d> {
        Some(Point2d::new(self.x.derivative()?, self.y.derivative()?))
    }

    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }
}

fn low_pass(value: f64, previous: f64, alpha: f64) -> f64 {
    alpha * value + (1.0 - alpha) * previous
}

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (std::f64::consts::TAU * cutoff);
    1.0 / (1.0 + tau / dt)
}
