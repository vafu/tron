use anyhow::{Context, Result};
use glam::Vec2;
use tron_api::{DepthPointProjection, OrientedBoundingBox, Rect, RoiResult, Size};

pub fn project_roi_at_depth<P>(
    projection: &P,
    roi: RoiResult,
    target_size: Size,
    depth_mm: f64,
) -> Result<Option<RoiResult>>
where
    P: DepthPointProjection,
{
    project_corners_at_depth(projection, roi_corners(roi), target_size, depth_mm)
}

pub fn project_horizontally_mirrored_roi_at_depth<P>(
    projection: &P,
    roi: RoiResult,
    source_size: Size,
    target_size: Size,
    depth_mm: f64,
) -> Result<Option<RoiResult>>
where
    P: DepthPointProjection,
{
    let corners = mirror_corners_x(roi_corners(roi), source_size);
    let Some(projected) = project_corners(projection, corners, depth_mm)? else {
        return Ok(None);
    };
    Ok(roi_from_corners(
        mirror_corners_x(projected, target_size),
        target_size,
    ))
}

fn project_corners_at_depth<P>(
    projection: &P,
    corners: [Vec2; 4],
    target_size: Size,
    depth_mm: f64,
) -> Result<Option<RoiResult>>
where
    P: DepthPointProjection,
{
    let Some(projected) = project_corners(projection, corners, depth_mm)? else {
        return Ok(None);
    };
    Ok(roi_from_corners(projected, target_size))
}

fn project_corners<P>(
    projection: &P,
    corners: [Vec2; 4],
    depth_mm: f64,
) -> Result<Option<[Vec2; 4]>>
where
    P: DepthPointProjection,
{
    let points = corners.map(|point| point.as_dvec2());
    let projected = projection
        .project_points(depth_mm, &points)
        .context("project ROI corners")?;
    anyhow::ensure!(
        projected.len() == 4,
        "ROI projection returned {} points, expected 4",
        projected.len()
    );

    let mut corners = [Vec2::ZERO; 4];
    for (corner, point) in corners.iter_mut().zip(projected) {
        let Some(point) = point else {
            return Ok(None);
        };
        if !point.x.is_finite() || !point.y.is_finite() {
            return Ok(None);
        }
        *corner = point.as_vec2();
    }
    Ok(Some(corners))
}

fn roi_corners(roi: RoiResult) -> [Vec2; 4] {
    roi.oriented_box
        .unwrap_or_else(|| rect_to_oriented_box(roi.rect))
        .corners
}

fn roi_from_corners(corners: [Vec2; 4], target_size: Size) -> Option<RoiResult> {
    let oriented_box = OrientedBoundingBox { corners };
    let rect = oriented_box.enclosing_rect(target_size)?;
    Some(RoiResult {
        rect,
        oriented_box: Some(oriented_box),
    })
}

fn rect_to_oriented_box(rect: Rect) -> OrientedBoundingBox {
    let x0 = rect.x as f32;
    let y0 = rect.y as f32;
    let x1 = (rect.x + rect.size.width) as f32;
    let y1 = (rect.y + rect.size.height) as f32;
    OrientedBoundingBox {
        corners: [
            Vec2::new(x0, y0),
            Vec2::new(x1, y0),
            Vec2::new(x1, y1),
            Vec2::new(x0, y1),
        ],
    }
}

fn mirror_corners_x(mut corners: [Vec2; 4], size: Size) -> [Vec2; 4] {
    let width = size.width as f32;
    for corner in &mut corners {
        corner.x = width - corner.x;
    }
    corners
}

#[cfg(test)]
mod tests {
    use tron_api::Point2d;

    use super::*;

    struct OffsetProjection {
        x: f64,
        y: f64,
    }

    impl DepthPointProjection for OffsetProjection {
        fn project_points(
            &self,
            _depth_mm: f64,
            points: &[Point2d],
        ) -> Result<Vec<Option<Point2d>>> {
            Ok(points
                .iter()
                .map(|point| Some(Point2d::new(point.x + self.x, point.y + self.y)))
                .collect())
        }
    }

    #[test]
    fn projects_axis_aligned_roi_to_oriented_box() {
        let projection = OffsetProjection { x: 10.0, y: 20.0 };
        let roi = RoiResult {
            rect: Rect {
                x: 5,
                y: 7,
                size: Size {
                    width: 11,
                    height: 13,
                },
            },
            oriented_box: None,
        };

        let projected = project_roi_at_depth(
            &projection,
            roi,
            Size {
                width: 100,
                height: 100,
            },
            700.0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            projected.rect,
            Rect {
                x: 15,
                y: 27,
                size: Size {
                    width: 11,
                    height: 13,
                },
            }
        );
        assert_eq!(
            projected.oriented_box.unwrap().corners,
            [
                Vec2::new(15.0, 27.0),
                Vec2::new(26.0, 27.0),
                Vec2::new(26.0, 40.0),
                Vec2::new(15.0, 40.0),
            ]
        );
    }

    #[test]
    fn accounts_for_horizontally_mirrored_source_and_target() {
        let projection = OffsetProjection { x: 0.0, y: 0.0 };
        let roi = RoiResult {
            rect: Rect {
                x: 10,
                y: 20,
                size: Size {
                    width: 30,
                    height: 40,
                },
            },
            oriented_box: None,
        };

        let projected = project_horizontally_mirrored_roi_at_depth(
            &projection,
            roi,
            Size {
                width: 100,
                height: 80,
            },
            Size {
                width: 200,
                height: 80,
            },
            700.0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            projected.rect,
            Rect {
                x: 110,
                y: 20,
                size: Size {
                    width: 30,
                    height: 40,
                },
            }
        );
    }
}
