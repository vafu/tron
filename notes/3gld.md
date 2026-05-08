---
title: "Checkerboard affine calibration path"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Checkerboard affine calibration path

# Checkerboard affine calibration path

Timestamp: 2026-05-07T19:21:14-07:00

Tags: #tron #task

Question / hypothesis: Manual RGB/IR affine tuning is too tedious; a
checkerboard visible in both streams should provide repeatable point
correspondences for automatic calibration.

Observed result: After rebuilding Gentoo `media-libs/opencv` with `features2d`,
`pkg-config --libs opencv4` exposes `opencv_calib3d` and the Rust `opencv`
bindings generate `opencv::calib3d`. Added a first live checkerboard calibration
path: `--checkerboard COLSxROWS` configures the inner-corner pattern, and `C` in
the app detects the board in the latest RGB and IR frames, fits the existing
IR->RGB affine profile, applies it, and saves it.

Follow-up tasks / instrumentation gaps: Test detection with a physical board in
both Lenovo and NexiGo streams. The current solve fits independent x/y
scale/offset only; once point collection is reliable, upgrade to a homography or
full stereo calibration using multiple board poses.
