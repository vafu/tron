mod checkerboard;
mod hand;
mod map;
mod roi;
mod source;

pub use checkerboard::CheckerboardDepthProjection;
pub use hand::{
    HandProjectionConfig, HandProjectionInput, HandProjectionOutput, HandProjectionProcessor,
    LandmarkDepthEstimate,
};
pub use map::FrameProjectionMap;
pub use roi::{project_horizontally_mirrored_roi_at_depth, project_roi_at_depth};
pub use source::{DepthProjectionMapSource, StaticProjectionMapSource};
