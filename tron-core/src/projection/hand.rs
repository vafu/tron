use anyhow::{Context, Result};
use tron_api::{DepthPointProjection, DepthSample, NoContext, Point2d, Processor, RoiResult, Size};

use crate::roi::mediapipe::HandLandmarks;

use super::project_roi_at_depth;

#[derive(Clone, Copy, Debug)]
pub struct HandProjectionConfig {
    pub fallback_depth_mm: f64,
    pub landmark_z_scale_mm: f64,
    pub source_mirrored_x: bool,
    pub target_mirrored_x: bool,
}

impl Default for HandProjectionConfig {
    fn default() -> Self {
        Self {
            fallback_depth_mm: 700.0,
            landmark_z_scale_mm: 1000.0,
            source_mirrored_x: false,
            target_mirrored_x: false,
        }
    }
}

pub struct HandProjectionInput<'a> {
    pub roi: Option<RoiResult>,
    pub landmarks: Option<&'a HandLandmarks>,
    pub depth_sample: Option<DepthSample>,
    pub source_size: Size,
    pub target_size: Size,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HandProjectionOutput {
    pub roi: Option<RoiResult>,
    pub landmarks: Option<HandLandmarks>,
    pub depth: LandmarkDepthEstimate,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LandmarkDepthEstimate {
    pub anchor_depth_mm: f64,
    pub anchor_landmark: Option<usize>,
    pub closest_relative_z: Option<f64>,
    pub landmark_depths_mm: [Option<f64>; 21],
    pub used_depth_sample: bool,
}

pub struct HandProjectionProcessor<P> {
    projection: P,
    config: HandProjectionConfig,
    latest_depth_mm: f64,
}

impl<P> HandProjectionProcessor<P> {
    pub fn new(projection: P, config: HandProjectionConfig) -> Result<Self> {
        anyhow::ensure!(
            config.fallback_depth_mm >= 0.0,
            "fallback projection depth must be non-negative"
        );
        anyhow::ensure!(
            config.landmark_z_scale_mm >= 0.0,
            "landmark z scale must be non-negative"
        );
        Ok(Self {
            projection,
            config,
            latest_depth_mm: config.fallback_depth_mm,
        })
    }

    pub fn latest_depth_mm(&self) -> f64 {
        self.latest_depth_mm
    }
}

impl<P> Processor<HandProjectionInput<'_>, NoContext> for HandProjectionProcessor<P>
where
    P: DepthPointProjection,
{
    type Output = HandProjectionOutput;

    fn process(
        &mut self,
        input: HandProjectionInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let used_depth_sample = update_latest_depth(input.depth_sample, &mut self.latest_depth_mm);
        let roi = project_roi(
            input.roi,
            input.source_size,
            input.target_size,
            &self.projection,
            &self.config,
            self.latest_depth_mm,
        )?;
        let (landmarks, depth) = project_landmarks(
            input.landmarks,
            input.source_size,
            input.target_size,
            &self.projection,
            &self.config,
            self.latest_depth_mm,
            used_depth_sample,
        )?;

        Ok(HandProjectionOutput {
            roi,
            landmarks,
            depth,
        })
    }
}

fn update_latest_depth(sample: Option<DepthSample>, latest_depth_mm: &mut f64) -> bool {
    let Some(sample) = sample else {
        return false;
    };
    let Some(depth_mm) = tof_projection_depth_mm(sample) else {
        return false;
    };
    *latest_depth_mm = depth_mm;
    true
}

fn tof_projection_depth_mm(sample: DepthSample) -> Option<f64> {
    sample
        .min_mm
        .filter(|depth| *depth > 0)
        .or_else(|| sample.center_mm.filter(|depth| *depth > 0))
        .map(f64::from)
}

fn project_roi<P>(
    roi: Option<RoiResult>,
    source_size: Size,
    target_size: Size,
    projection: &P,
    config: &HandProjectionConfig,
    depth_mm: f64,
) -> Result<Option<RoiResult>>
where
    P: DepthPointProjection,
{
    let Some(roi) = roi else {
        return Ok(None);
    };
    let roi = if config.source_mirrored_x {
        mirror_roi_x(roi, source_size)
    } else {
        roi
    };
    let projected = project_roi_at_depth(projection, roi, target_size, depth_mm)?;
    Ok(projected.map(|roi| {
        if config.target_mirrored_x {
            mirror_roi_x(roi, target_size)
        } else {
            roi
        }
    }))
}

fn project_landmarks<P>(
    landmarks: Option<&HandLandmarks>,
    source_size: Size,
    target_size: Size,
    projection: &P,
    config: &HandProjectionConfig,
    anchor_depth_mm: f64,
    used_depth_sample: bool,
) -> Result<(Option<HandLandmarks>, LandmarkDepthEstimate)>
where
    P: DepthPointProjection,
{
    let mut landmark_depths_mm = [None; 21];
    let Some(landmarks) = landmarks else {
        return Ok((
            None,
            LandmarkDepthEstimate {
                anchor_depth_mm,
                anchor_landmark: None,
                closest_relative_z: None,
                landmark_depths_mm,
                used_depth_sample,
            },
        ));
    };

    let closest = closest_landmark_z(landmarks);
    let mut projected_landmarks = landmarks.clone();
    let mut valid_points = 0;

    if let Some((anchor_landmark, closest_relative_z)) = closest {
        for (index, point) in landmarks.points.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                projected_landmarks.points[index].x = f64::NAN;
                projected_landmarks.points[index].y = f64::NAN;
                continue;
            }

            let depth_mm =
                anchor_depth_mm + (point.z - closest_relative_z) * config.landmark_z_scale_mm;
            let depth_mm = depth_mm.max(0.0);
            landmark_depths_mm[index] = Some(depth_mm);

            let mut source_point = Point2d::new(point.x, point.y);
            if config.source_mirrored_x {
                source_point.x = source_size.width as f64 - source_point.x;
            }
            let projected = projection
                .project_points(depth_mm, &[source_point])
                .with_context(|| format!("project hand landmark {index}"))?;
            anyhow::ensure!(
                projected.len() == 1,
                "landmark projection returned {} points, expected 1",
                projected.len()
            );

            match projected[0] {
                Some(mut point) if point.x.is_finite() && point.y.is_finite() => {
                    if config.target_mirrored_x {
                        point.x = target_size.width as f64 - point.x;
                    }
                    projected_landmarks.points[index].x = point.x;
                    projected_landmarks.points[index].y = point.y;
                    valid_points += 1;
                }
                _ => {
                    projected_landmarks.points[index].x = f64::NAN;
                    projected_landmarks.points[index].y = f64::NAN;
                }
            }
        }

        return Ok((
            (valid_points > 0).then_some(projected_landmarks),
            LandmarkDepthEstimate {
                anchor_depth_mm,
                anchor_landmark: Some(anchor_landmark),
                closest_relative_z: Some(closest_relative_z),
                landmark_depths_mm,
                used_depth_sample,
            },
        ));
    }

    Ok((
        None,
        LandmarkDepthEstimate {
            anchor_depth_mm,
            anchor_landmark: None,
            closest_relative_z: None,
            landmark_depths_mm,
            used_depth_sample,
        },
    ))
}

fn closest_landmark_z(landmarks: &HandLandmarks) -> Option<(usize, f64)> {
    landmarks
        .points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.x.is_finite() && point.y.is_finite() && point.z.is_finite())
        .min_by(|(_, a), (_, b)| a.z.total_cmp(&b.z))
        .map(|(index, point)| (index, point.z))
}

fn mirror_roi_x(mut roi: RoiResult, size: Size) -> RoiResult {
    if let Some(mut oriented_box) = roi.oriented_box {
        let width = size.width as f32;
        for corner in &mut oriented_box.corners {
            corner.x = width - corner.x;
        }
        roi.oriented_box = Some(oriented_box);
        roi.rect = oriented_box.enclosing_rect(size).unwrap_or(roi.rect);
        return roi;
    }

    roi.rect.x = size
        .width
        .saturating_sub(roi.rect.x.saturating_add(roi.rect.size.width));
    roi
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tron_api::{DepthSample, Point2d, Rect};

    use super::*;
    use crate::roi::mediapipe::HandLandmark;

    struct IdentityProjection;

    impl DepthPointProjection for IdentityProjection {
        fn project_points(
            &self,
            _depth_mm: f64,
            points: &[Point2d],
        ) -> Result<Vec<Option<Point2d>>> {
            Ok(points.iter().copied().map(Some).collect())
        }
    }

    fn sample(min_mm: Option<u16>, center_mm: Option<u16>) -> DepthSample {
        DepthSample {
            sequence: None,
            sensor_timestamp_us: None,
            printed_at_ms: None,
            resolution: None,
            center_mm,
            min_mm,
            max_mm: None,
            valid_zones: None,
            zones: [0; 64],
            zone_count: 0,
            sampled_at: Instant::now(),
            received_at: Instant::now(),
        }
    }

    fn landmarks(points: [(usize, f32, f32, f32); 2]) -> HandLandmarks {
        let mut landmarks = HandLandmarks {
            points: [HandLandmark::splat(f64::NAN); 21],
            presence: 0.9,
            handedness: None,
            timestamp: Instant::now(),
        };
        for (index, x, y, z) in points {
            landmarks.points[index] = HandLandmark::new(x as f64, y as f64, z as f64);
        }
        landmarks
    }

    #[test]
    fn prefers_valid_min_depth_then_center_depth() {
        let mut processor = HandProjectionProcessor::new(
            IdentityProjection,
            HandProjectionConfig {
                fallback_depth_mm: 700.0,
                ..HandProjectionConfig::default()
            },
        )
        .unwrap();

        let output = processor
            .process(
                HandProjectionInput {
                    roi: None,
                    landmarks: None,
                    depth_sample: Some(sample(Some(600), Some(700))),
                    source_size: Size {
                        width: 100,
                        height: 80,
                    },
                    target_size: Size {
                        width: 100,
                        height: 80,
                    },
                },
                NoContext,
            )
            .unwrap();
        assert_eq!(output.depth.anchor_depth_mm, 600.0);

        let output = processor
            .process(
                HandProjectionInput {
                    roi: None,
                    landmarks: None,
                    depth_sample: Some(sample(Some(0), Some(650))),
                    source_size: Size {
                        width: 100,
                        height: 80,
                    },
                    target_size: Size {
                        width: 100,
                        height: 80,
                    },
                },
                NoContext,
            )
            .unwrap();
        assert_eq!(output.depth.anchor_depth_mm, 650.0);
    }

    #[test]
    fn anchors_closest_relative_z_to_tof_min_depth() {
        let mut processor = HandProjectionProcessor::new(
            IdentityProjection,
            HandProjectionConfig {
                fallback_depth_mm: 700.0,
                landmark_z_scale_mm: 1000.0,
                ..HandProjectionConfig::default()
            },
        )
        .unwrap();
        let landmarks = landmarks([(0, 10.0, 20.0, -0.05), (1, 12.0, 22.0, 0.10)]);

        let output = processor
            .process(
                HandProjectionInput {
                    roi: None,
                    landmarks: Some(&landmarks),
                    depth_sample: Some(sample(Some(500), None)),
                    source_size: Size {
                        width: 100,
                        height: 80,
                    },
                    target_size: Size {
                        width: 100,
                        height: 80,
                    },
                },
                NoContext,
            )
            .unwrap();

        assert_eq!(output.depth.anchor_landmark, Some(0));
        assert_eq!(output.depth.landmark_depths_mm[0], Some(500.0));
        let depth = output.depth.landmark_depths_mm[1].unwrap();
        assert!((depth - 650.0).abs() < 0.001);
    }

    #[test]
    fn projects_landmarks_and_roi_with_horizontal_mirror() {
        let mut processor = HandProjectionProcessor::new(
            IdentityProjection,
            HandProjectionConfig {
                fallback_depth_mm: 700.0,
                source_mirrored_x: true,
                target_mirrored_x: true,
                ..HandProjectionConfig::default()
            },
        )
        .unwrap();
        let landmarks = landmarks([(0, 10.0, 20.0, 0.0), (1, 12.0, 22.0, 0.1)]);

        let output = processor
            .process(
                HandProjectionInput {
                    roi: Some(RoiResult {
                        rect: Rect {
                            x: 10,
                            y: 20,
                            size: Size {
                                width: 30,
                                height: 40,
                            },
                        },
                        oriented_box: None,
                    }),
                    landmarks: Some(&landmarks),
                    depth_sample: None,
                    source_size: Size {
                        width: 100,
                        height: 80,
                    },
                    target_size: Size {
                        width: 200,
                        height: 80,
                    },
                },
                NoContext,
            )
            .unwrap();

        let projected = output.landmarks.unwrap();
        assert_eq!(projected.points[0].x, 110.0);
        assert_eq!(projected.points[0].y, 20.0);
        assert_eq!(output.roi.unwrap().rect.x, 110);
    }
}
