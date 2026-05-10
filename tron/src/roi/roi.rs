use anyhow::Result;
use tron_api::{CameraRoiControl, Rect, Size};

const DEFAULT_ROI_SIZE: Size = Size {
    width: 80,
    height: 80,
};

pub struct RoiController {
    control: Box<dyn CameraRoiControl>,
    rect: Rect,
    step: u32,
}

impl RoiController {
    pub fn new(control: Box<dyn CameraRoiControl>, rect: Rect, step: u32) -> Self {
        Self {
            control,
            rect,
            step: step.max(1),
        }
    }

    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn apply(&mut self) -> Result<()> {
        self.apply_rect()
    }

    pub fn apply_rect(&mut self) -> Result<()> {
        self.control.set_roi_rect(self.rect)
    }

    pub fn set_auto(&mut self, enabled: bool) -> Result<()> {
        self.control.set_roi_auto(enabled)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.rect = Rect {
            x: 0,
            y: 0,
            size: Size {
                width: 0,
                height: 0,
            },
        };
        self.apply_rect()
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
        self.rect.size.width = self.rect.size.width.saturating_add_signed(amount).max(1);
        self.rect.size.height = self.rect.size.height.saturating_add_signed(amount).max(1);
        self.rect = self.rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn set_square_size(&mut self, edge: u32, frame_size: Size) -> Result<()> {
        let edge = edge.max(1);
        let cx = self.rect.x + self.rect.size.width / 2;
        let cy = self.rect.y + self.rect.size.height / 2;
        self.rect = Rect {
            x: cx.saturating_sub(edge / 2),
            y: cy.saturating_sub(edge / 2),
            size: Size {
                width: edge,
                height: edge,
            },
        };
        self.rect = self.rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn set_rect(&mut self, rect: Rect, frame_size: Size) -> Result<()> {
        self.rect = rect.clamp_to(frame_size);
        self.apply_rect()
    }

    pub fn center_on(&mut self, x: u32, y: u32, frame_size: Size) -> Result<()> {
        let rect = self.rect.non_empty_or(DEFAULT_ROI_SIZE);
        self.rect = Rect {
            x: x.saturating_sub(rect.size.width / 2),
            y: y.saturating_sub(rect.size.height / 2),
            size: rect.size,
        };
        self.rect = self.rect.clamp_to(frame_size);
        self.apply_rect()
    }
}
