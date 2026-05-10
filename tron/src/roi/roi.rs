use anyhow::{Context, Result};
use std::process::Command;
use tron_api::Size;

#[derive(Clone, Copy, Debug)]
pub struct RoiRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl RoiRect {
    pub fn clamp_to(self, size: Size) -> Self {
        let width = self.width.min(size.width).max(1);
        let height = self.height.min(size.height).max(1);
        let x = self.x.min(size.width.saturating_sub(width));
        let y = self.y.min(size.height.saturating_sub(height));
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn non_empty_or_default(self) -> Self {
        if self.width > 0 && self.height > 0 {
            self
        } else {
            Self {
                x: self.x,
                y: self.y,
                width: 80,
                height: 80,
            }
        }
    }
}

pub struct RoiController {
    device: String,
    rect: RoiRect,
    step: u32,
}

impl RoiController {
    pub fn new(device: String, rect: RoiRect, step: u32) -> Self {
        Self {
            device,
            rect,
            step: step.max(1),
        }
    }

    pub fn rect(&self) -> RoiRect {
        self.rect
    }

    pub fn apply(&self) -> Result<()> {
        self.apply_rect()
    }

    pub fn apply_rect(&self) -> Result<()> {
        set_roi_rect(&self.device, self.rect)
    }

    pub fn set_auto(&self, enabled: bool) -> Result<()> {
        set_roi_auto(&self.device, enabled)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.rect = RoiRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
        set_roi_rect(&self.device, self.rect)
    }

    pub fn move_by(&mut self, dx: i32, dy: i32, frame_size: Size) -> Result<()> {
        let x = self.rect.x.saturating_add_signed(dx * self.step as i32);
        let y = self.rect.y.saturating_add_signed(dy * self.step as i32);
        self.rect.x = x;
        self.rect.y = y;
        self.rect = self.rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn resize_by_step(&mut self, delta: i32, frame_size: Size) -> Result<()> {
        self.resize_by_pixels(delta * self.step as i32, frame_size)
    }

    pub fn resize_by_pixels(&mut self, amount: i32, frame_size: Size) -> Result<()> {
        self.rect.width = self.rect.width.saturating_add_signed(amount).max(1);
        self.rect.height = self.rect.height.saturating_add_signed(amount).max(1);
        self.rect = self.rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn set_square_size(&mut self, edge: u32, frame_size: Size) -> Result<()> {
        let edge = edge.max(1);
        let cx = self.rect.x + self.rect.width / 2;
        let cy = self.rect.y + self.rect.height / 2;
        self.rect = RoiRect {
            x: cx.saturating_sub(edge / 2),
            y: cy.saturating_sub(edge / 2),
            width: edge,
            height: edge,
        }
        .clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn set_rect(&mut self, rect: RoiRect, frame_size: Size) -> Result<()> {
        self.rect = rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn center_on(&mut self, x: u32, y: u32, frame_size: Size) -> Result<()> {
        let rect = self.rect.non_empty_or_default();
        self.rect = RoiRect {
            x: x.saturating_sub(rect.width / 2),
            y: y.saturating_sub(rect.height / 2),
            width: rect.width,
            height: rect.height,
        }
        .clamp_to(frame_size);
        self.apply_rect()
    }
}

fn set_roi_auto(device: &str, enabled: bool) -> Result<()> {
    let value = if enabled { "1" } else { "0" };
    run_v4l2_ctl(device, &format!("region_of_interest_auto_ctrls={value}"))
}

fn set_roi_rect(device: &str, rect: RoiRect) -> Result<()> {
    run_v4l2_ctl(
        device,
        &format!(
            "region_of_interest_rectangle=({},{})/{}x{}",
            rect.x, rect.y, rect.width, rect.height
        ),
    )
}

fn run_v4l2_ctl(device: &str, control: &str) -> Result<()> {
    let status = Command::new("v4l2-ctl")
        .arg("-d")
        .arg(device)
        .arg(format!("--set-ctrl={control}"))
        .status()
        .with_context(|| format!("run v4l2-ctl for {control}"))?;
    anyhow::ensure!(status.success(), "v4l2-ctl failed for {control}");
    Ok(())
}
