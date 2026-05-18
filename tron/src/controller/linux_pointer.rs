use anyhow::{Context, Result};
use evdev::{
    AttributeSet, EventType, InputEvent, KeyCode, RelativeAxisCode, uinput::VirtualDevice,
};
use tron_api::{Point2d, PointerEvent, PointerOutput, Sink};

#[derive(Clone, Copy, Debug)]
pub struct LinuxPointerConfig {
    pub units_per_delta: f64,
}

impl Default for LinuxPointerConfig {
    fn default() -> Self {
        Self {
            units_per_delta: 1400.0,
        }
    }
}

pub struct LinuxPointerSink {
    device: VirtualDevice,
    config: LinuxPointerConfig,
    remainder: Point2d,
    left_down: bool,
}

impl LinuxPointerSink {
    pub fn new(config: LinuxPointerConfig) -> Result<Self> {
        let device = VirtualDevice::builder()
            .context("open /dev/uinput for tron pointer")?
            .name("tron pointer")
            .with_keys(&AttributeSet::from_iter([KeyCode::BTN_LEFT]))
            .context("configure tron pointer button")?
            .with_relative_axes(&AttributeSet::from_iter([
                RelativeAxisCode::REL_X,
                RelativeAxisCode::REL_Y,
            ]))
            .context("configure tron pointer axes")?
            .build()
            .context("create tron pointer uinput device")?;

        Ok(Self {
            device,
            config,
            remainder: Point2d::ZERO,
            left_down: false,
        })
    }

    fn consume_event(&mut self, event: PointerEvent) -> Result<()> {
        match event {
            PointerEvent::Move { delta, .. } => self.emit_move(delta)?,
            PointerEvent::Down { .. } => self.emit_left_button(true)?,
            PointerEvent::Up { .. } | PointerEvent::Cancel { .. } => {
                self.emit_left_button(false)?;
                self.remainder = Point2d::ZERO;
            }
            PointerEvent::Click { .. } => {
                self.emit_left_button(true)?;
                self.emit_left_button(false)?;
            }
        }
        Ok(())
    }

    fn emit_move(&mut self, delta: Point2d) -> Result<()> {
        let scaled = self.remainder + delta * self.config.units_per_delta;
        let dx = scaled.x.trunc() as i32;
        let dy = scaled.y.trunc() as i32;
        self.remainder = Point2d::new(scaled.x - f64::from(dx), scaled.y - f64::from(dy));

        if dx == 0 && dy == 0 {
            return Ok(());
        }

        let mut events = Vec::with_capacity(2);
        if dx != 0 {
            events.push(relative_event(RelativeAxisCode::REL_X, dx));
        }
        if dy != 0 {
            events.push(relative_event(RelativeAxisCode::REL_Y, dy));
        }
        self.device.emit(&events).context("emit pointer move")?;
        Ok(())
    }

    fn emit_left_button(&mut self, down: bool) -> Result<()> {
        if self.left_down == down {
            return Ok(());
        }

        let value = if down { 1 } else { 0 };
        self.device
            .emit(&[InputEvent::new(
                EventType::KEY.0,
                KeyCode::BTN_LEFT.0,
                value,
            )])
            .context("emit pointer button")?;
        self.left_down = down;
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl Sink<PointerOutput> for LinuxPointerSink {
    async fn consume(&mut self, output: PointerOutput) -> Result<()> {
        if let PointerOutput::Event(event) = output {
            self.consume_event(event)?;
        }
        Ok(())
    }
}

fn relative_event(axis: RelativeAxisCode, value: i32) -> InputEvent {
    InputEvent::new(EventType::RELATIVE.0, axis.0, value)
}
