use std::ffi::CStr;
use std::mem;
use std::path::Path;

use anyhow::{Context, Result};
use tron_api::{CameraRoiControl, Rect, Size};
use v4l::Device;
use v4l::v4l_sys::{
    V4L2_CTRL_FLAG_NEXT_COMPOUND, V4L2_CTRL_FLAG_NEXT_CTRL, v4l2_ext_control,
    v4l2_ext_control__bindgen_ty_1, v4l2_ext_controls, v4l2_ext_controls__bindgen_ty_1,
    v4l2_query_ext_ctrl, v4l2_rect,
};
use v4l::v4l2;

const ROI_RECT_CONTROL_NAME: &str = "regionofinterestrectangle";
const ROI_AUTO_CONTROL_NAME: &str = "regionofinterestautoctrls";

pub struct V4lCameraRoiControl {
    device: Device,
    roi_rect_id: u32,
    roi_auto_id: Option<u32>,
}

impl V4lCameraRoiControl {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let device = Device::with_path(path)
            .with_context(|| format!("open V4L ROI control {}", path.display()))?;
        let roi_rect_id = find_control_id(&device, ROI_RECT_CONTROL_NAME)
            .with_context(|| format!("find region_of_interest_rectangle on {}", path.display()))?;
        let roi_auto_id = find_control_id_optional(&device, ROI_AUTO_CONTROL_NAME)
            .with_context(|| format!("find region_of_interest_auto_ctrls on {}", path.display()))?;
        Ok(Self {
            device,
            roi_rect_id,
            roi_auto_id,
        })
    }
}

impl CameraRoiControl for V4lCameraRoiControl {
    fn roi_rect(&self) -> Result<Rect> {
        let mut rect = v4l2_rect {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
        };
        let mut control = ext_rect_control(self.roi_rect_id, &mut rect);
        let mut controls = ext_controls(self.roi_rect_id, &mut control);
        unsafe {
            v4l2::ioctl(
                self.device.handle().fd(),
                v4l2::vidioc::VIDIOC_G_EXT_CTRLS,
                &mut controls as *mut _ as *mut std::os::raw::c_void,
            )
        }
        .context("read V4L ROI rectangle")?;
        rect_from_v4l2_rect(rect)
    }

    fn set_roi_rect(&mut self, rect: Rect) -> Result<()> {
        let mut rect = v4l2_rect_from_rect(rect)?;
        let mut control = ext_rect_control(self.roi_rect_id, &mut rect);
        let mut controls = ext_controls(self.roi_rect_id, &mut control);
        unsafe {
            v4l2::ioctl(
                self.device.handle().fd(),
                v4l2::vidioc::VIDIOC_S_EXT_CTRLS,
                &mut controls as *mut _ as *mut std::os::raw::c_void,
            )
        }
        .context("set V4L ROI rectangle")?;
        Ok(())
    }

    fn set_roi_auto(&mut self, enabled: bool) -> Result<()> {
        let id = self
            .roi_auto_id
            .ok_or_else(|| anyhow::anyhow!("region_of_interest_auto_ctrls is not exposed"))?;
        set_integer_control(&self.device, id, if enabled { 1 } else { 0 })?;
        Ok(())
    }
}

fn find_control_id(device: &Device, normalized_name: &str) -> Result<u32> {
    find_control_id_optional(device, normalized_name)?
        .ok_or_else(|| anyhow::anyhow!("control {normalized_name:?} is not exposed"))
}

fn find_control_id_optional(device: &Device, normalized_name: &str) -> Result<Option<u32>> {
    let mut id = 0;
    loop {
        let mut control = v4l2_query_ext_ctrl {
            id: id | V4L2_CTRL_FLAG_NEXT_CTRL | V4L2_CTRL_FLAG_NEXT_COMPOUND,
            ..unsafe { mem::zeroed() }
        };
        let result = unsafe {
            v4l2::ioctl(
                device.handle().fd(),
                v4l2::vidioc::VIDIOC_QUERY_EXT_CTRL,
                &mut control as *mut _ as *mut std::os::raw::c_void,
            )
        };
        if result.is_err() {
            break;
        }

        let name = unsafe { CStr::from_ptr(control.name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if normalize_control_name(&name) == normalized_name {
            return Ok(Some(control.id));
        }
        id = control.id;
    }

    Ok(None)
}

fn normalize_control_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn ext_rect_control(id: u32, rect: &mut v4l2_rect) -> v4l2_ext_control {
    v4l2_ext_control {
        id,
        size: mem::size_of::<v4l2_rect>() as u32,
        reserved2: [0],
        __bindgen_anon_1: v4l2_ext_control__bindgen_ty_1 { p_rect: rect },
    }
}

fn ext_controls(id: u32, control: &mut v4l2_ext_control) -> v4l2_ext_controls {
    v4l2_ext_controls {
        __bindgen_anon_1: v4l2_ext_controls__bindgen_ty_1 {
            ctrl_class: control_class(id),
        },
        count: 1,
        error_idx: 0,
        request_fd: 0,
        reserved: [0],
        controls: control,
    }
}

fn set_integer_control(device: &Device, id: u32, value: i64) -> Result<()> {
    let mut control = v4l2_ext_control {
        id,
        size: mem::size_of::<i64>() as u32,
        reserved2: [0],
        __bindgen_anon_1: v4l2_ext_control__bindgen_ty_1 { value64: value },
    };
    let mut controls = ext_controls(id, &mut control);
    unsafe {
        v4l2::ioctl(
            device.handle().fd(),
            v4l2::vidioc::VIDIOC_S_EXT_CTRLS,
            &mut controls as *mut _ as *mut std::os::raw::c_void,
        )
    }
    .context("set V4L integer control")?;
    Ok(())
}

fn control_class(id: u32) -> u32 {
    id & 0xffff0000
}

fn rect_from_v4l2_rect(value: v4l2_rect) -> Result<Rect> {
    anyhow::ensure!(value.left >= 0, "V4L ROI left is negative: {}", value.left);
    anyhow::ensure!(value.top >= 0, "V4L ROI top is negative: {}", value.top);
    Ok(Rect {
        x: value.left as u32,
        y: value.top as u32,
        size: Size {
            width: value.width,
            height: value.height,
        },
    })
}

fn v4l2_rect_from_rect(value: Rect) -> Result<v4l2_rect> {
    Ok(v4l2_rect {
        left: i32::try_from(value.x).context("V4L ROI x does not fit i32")?,
        top: i32::try_from(value.y).context("V4L ROI y does not fit i32")?,
        width: value.size.width,
        height: value.size.height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_v4l_control_names() {
        assert_eq!(
            normalize_control_name("Region of Interest Rectangle"),
            "regionofinterestrectangle"
        );
        assert_eq!(
            normalize_control_name("region_of_interest_rectangle"),
            "regionofinterestrectangle"
        );
    }

    #[test]
    fn converts_rect_to_v4l_rect() {
        let rect = v4l2_rect_from_rect(Rect {
            x: 10,
            y: 20,
            size: Size {
                width: 30,
                height: 40,
            },
        })
        .unwrap();

        assert_eq!(rect.left, 10);
        assert_eq!(rect.top, 20);
        assert_eq!(rect.width, 30);
        assert_eq!(rect.height, 40);
    }
}
