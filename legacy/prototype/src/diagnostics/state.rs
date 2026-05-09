use crate::pipeline::{HandState, Image, PointerState};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct DiagnosticsHandle {
    inner: Arc<Mutex<DiagnosticsSnapshot>>,
}

#[derive(Clone, Debug)]
pub struct DiagnosticsSnapshot {
    pub updated_unix_ms: u64,
    pub frame: u64,
    pub rgb_seq: u64,
    pub ir_seq: u64,
    pub ir_diff_seq: Option<u64>,
    pub interval_ms: f32,
    pub input_age_ms: f32,
    pub proximity: Option<i64>,
    pub calibration: DiagnosticsCalibration,
    pub hand: Option<DiagnosticsHand>,
    pub ir_depth: Option<DiagnosticsIrDepth>,
    pub pointer: Option<DiagnosticsPointer>,
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticsCalibration {
    pub scale_x: f32,
    pub scale_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub use_binary: bool,
}

#[derive(Clone, Debug)]
pub struct DiagnosticsHand {
    pub gesture: Option<&'static str>,
    pub presence: f32,
    pub handedness: String,
    pub roi_x: f32,
    pub roi_y: f32,
    pub roi_w: f32,
    pub roi_h: f32,
    pub gesture_features: String,
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticsIrDepth {
    pub hand_diff_mean: f32,
    pub hand_diff_median: f32,
    pub background_diff_mean: f32,
    pub background_diff_median: f32,
    pub raw_hand_mean: f32,
    pub clip_fraction: f32,
    pub corrected_signal: f32,
    pub delta: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticsPointer {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub grabbed: bool,
    pub confidence: f32,
}

impl DiagnosticsHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(DiagnosticsSnapshot::default())),
        }
    }

    pub fn publish(&self, snapshot: DiagnosticsSnapshot) {
        *self.inner.lock().unwrap() = snapshot;
    }

    pub fn snapshot_json(&self) -> String {
        self.inner.lock().unwrap().to_json()
    }
}

impl DiagnosticsSnapshot {
    pub fn from_pipeline(
        frame: u64,
        rgb_seq: u64,
        ir_seq: u64,
        interval_ms: f32,
        input_age_ms: f32,
        proximity: Option<i64>,
        state: Option<&HandState>,
        pointer: Option<&PointerState>,
        ir_diff: Option<&Image>,
    ) -> Self {
        let ir_depth = state.and_then(|s| s.ir_depth).map(|d| DiagnosticsIrDepth {
            hand_diff_mean: d.hand_diff_mean,
            hand_diff_median: d.hand_diff_median,
            background_diff_mean: d.background_diff_mean,
            background_diff_median: d.background_diff_median,
            raw_hand_mean: d.raw_hand_mean,
            clip_fraction: d.clip_fraction,
            corrected_signal: d.corrected_signal,
            delta: d.delta,
            confidence: d.confidence,
        });
        let calib = crate::calib::current();

        Self {
            updated_unix_ms: now_unix_ms(),
            frame,
            rgb_seq,
            ir_seq,
            ir_diff_seq: ir_diff.map(|img| img.seq),
            interval_ms,
            input_age_ms,
            proximity,
            calibration: DiagnosticsCalibration {
                scale_x: calib.scale_x,
                scale_y: calib.scale_y,
                offset_x: calib.offset_x,
                offset_y: calib.offset_y,
                use_binary: calib.use_binary,
            },
            hand: state.map(|s| DiagnosticsHand {
                gesture: s.gesture.map(|g| g.name()),
                presence: s.landmarks.presence,
                handedness: format!("{:?}", s.landmarks.handedness),
                roi_x: s.roi.x,
                roi_y: s.roi.y,
                roi_w: s.roi.w,
                roi_h: s.roi.h,
                gesture_features: s.gesture_features.summary(),
            }),
            ir_depth,
            pointer: pointer.map(|p| DiagnosticsPointer {
                x: p.position.x,
                y: p.position.y,
                z: p.position.z,
                grabbed: p.grabbed,
                confidence: p.confidence,
            }),
        }
    }

    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push('{');
        push_num(&mut out, "updated_unix_ms", self.updated_unix_ms);
        push_num(&mut out, "frame", self.frame);
        push_num(&mut out, "rgb_seq", self.rgb_seq);
        push_num(&mut out, "ir_seq", self.ir_seq);
        push_opt_num(&mut out, "ir_diff_seq", self.ir_diff_seq);
        push_float(&mut out, "interval_ms", self.interval_ms);
        push_float(&mut out, "input_age_ms", self.input_age_ms);
        push_opt_i64(&mut out, "proximity", self.proximity);

        out.push_str("\"calibration\":{");
        push_float(&mut out, "scale_x", self.calibration.scale_x);
        push_float(&mut out, "scale_y", self.calibration.scale_y);
        push_float(&mut out, "offset_x", self.calibration.offset_x);
        push_float(&mut out, "offset_y", self.calibration.offset_y);
        push_bool_last(&mut out, "use_binary", self.calibration.use_binary);
        out.push_str("},");

        out.push_str("\"hand\":");
        if let Some(hand) = &self.hand {
            out.push('{');
            out.push_str("\"gesture\":");
            push_opt_str_value(&mut out, hand.gesture);
            out.push(',');
            push_float(&mut out, "presence", hand.presence);
            push_str(&mut out, "handedness", &hand.handedness);
            out.push_str("\"roi\":{");
            push_float(&mut out, "x", hand.roi_x);
            push_float(&mut out, "y", hand.roi_y);
            push_float(&mut out, "w", hand.roi_w);
            push_float_last(&mut out, "h", hand.roi_h);
            out.push_str("},");
            push_str_last(&mut out, "gesture_features", &hand.gesture_features);
            out.push('}');
        } else {
            out.push_str("null");
        }
        out.push(',');

        out.push_str("\"ir_depth\":");
        if let Some(depth) = self.ir_depth {
            out.push('{');
            push_float(&mut out, "corrected_signal", depth.corrected_signal);
            push_float(&mut out, "delta", depth.delta);
            push_float(&mut out, "confidence", depth.confidence);
            push_float(&mut out, "clip_fraction", depth.clip_fraction);
            push_float(&mut out, "hand_diff_mean", depth.hand_diff_mean);
            push_float(&mut out, "hand_diff_median", depth.hand_diff_median);
            push_float(&mut out, "background_diff_mean", depth.background_diff_mean);
            push_float(
                &mut out,
                "background_diff_median",
                depth.background_diff_median,
            );
            push_float_last(&mut out, "raw_hand_mean", depth.raw_hand_mean);
            out.push('}');
        } else {
            out.push_str("null");
        }
        out.push(',');

        out.push_str("\"pointer\":");
        if let Some(pointer) = self.pointer {
            out.push('{');
            push_float(&mut out, "x", pointer.x);
            push_float(&mut out, "y", pointer.y);
            push_float(&mut out, "z", pointer.z);
            push_bool(&mut out, "grabbed", pointer.grabbed);
            push_float_last(&mut out, "confidence", pointer.confidence);
            out.push('}');
        } else {
            out.push_str("null");
        }
        out.push('}');
        out
    }
}

impl Default for DiagnosticsSnapshot {
    fn default() -> Self {
        Self {
            updated_unix_ms: now_unix_ms(),
            frame: 0,
            rgb_seq: 0,
            ir_seq: 0,
            ir_diff_seq: None,
            interval_ms: 0.0,
            input_age_ms: 0.0,
            proximity: None,
            calibration: DiagnosticsCalibration {
                scale_x: 1.0,
                scale_y: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
                use_binary: false,
            },
            hand: None,
            ir_depth: None,
            pointer: None,
        }
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn push_num(out: &mut String, key: &str, value: u64) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
    out.push(',');
}

fn push_opt_num(out: &mut String, key: &str, value: Option<u64>) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    if let Some(value) = value {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
    out.push(',');
}

fn push_opt_i64(out: &mut String, key: &str, value: Option<i64>) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    if let Some(value) = value {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
    out.push(',');
}

fn push_float(out: &mut String, key: &str, value: f32) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    push_float_value(out, value);
    out.push(',');
}

fn push_float_last(out: &mut String, key: &str, value: f32) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    push_float_value(out, value);
}

fn push_float_value(out: &mut String, value: f32) {
    if value.is_finite() {
        out.push_str(&format!("{value:.6}"));
    } else {
        out.push_str("null");
    }
}

fn push_bool(out: &mut String, key: &str, value: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(if value { "true" } else { "false" });
    out.push(',');
}

fn push_bool_last(out: &mut String, key: &str, value: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(if value { "true" } else { "false" });
}

fn push_str(out: &mut String, key: &str, value: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    push_escaped(out, value);
    out.push_str("\",");
}

fn push_str_last(out: &mut String, key: &str, value: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    push_escaped(out, value);
    out.push('"');
}

fn push_opt_str_value(out: &mut String, value: Option<&str>) {
    if let Some(value) = value {
        out.push('"');
        push_escaped(out, value);
        out.push('"');
    } else {
        out.push_str("null");
    }
}

fn push_escaped(out: &mut String, value: &str) {
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}
