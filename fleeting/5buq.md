---
title: Stereo calibration separates baseline and FOV
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Stereo calibration separates baseline and FOV

Timestamp: 2026-05-07T20:58:37-07:00

Question / hypothesis: How should tron solve the two calibration problems: physical camera separation and different RGB/IR FOV?

Observed result: These are separate parameters in the stereo camera model. Physical distance is the stereo extrinsic baseline, represented by rotation R and translation T between IR and RGB camera coordinate systems. Different FOV, resolution, optical center, and lens distortion are camera intrinsics, represented separately for RGB and IR as fx, fy, cx, cy, and distortion coefficients. A 2D affine profile collapses baseline, FOV, optical center, distortion, and depth into one plane-specific image warp, so it cannot be valid across hand depths.

Follow-up tasks / instrumentation gaps: Make tron-calib produce a stereo rig profile with RGB intrinsics, IR intrinsics, distortion, IR-to-RGB R/T, rectification maps, and quality metrics. Runtime should expose APIs for IR pixel to RGB epipolar line, triangulation from matched RGB/IR points, projection with an assumed depth, and rectification. For hand tracking, combine calibrated rays with landmark/ROI correspondences plus IR brightness, landmark scale, or proximity depth priors.

Tags: #codex #calibration #stereo #tracking
