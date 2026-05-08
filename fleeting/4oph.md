---
title: Stereo calibration needs coverage gates
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Stereo calibration needs coverage gates

Timestamp: 2026-05-07T20:54:56-07:00

Question / hypothesis: Why did the initial RGB/IR stereo calibration fail, and what must change for a proper calibration flow?

Observed result: The current calibration path mostly accepts any pair where both cameras detect a checkerboard. That only proves matching 2D corner observations exist; it does not guarantee enough pose diversity or solve quality. Proper stereo calibration must model each camera's intrinsics, distortion, and RGB-to-IR extrinsics, while the capture process must enforce broad image coverage, board size/depth variation, tilted/skewed poses, many accepted samples, and per-view reprojection error filtering. The old affine profile collapses baseline, FOV, optical center, distortion, and depth into one 2D transform, so it can only be valid near the plane/depth where it was captured.

Follow-up tasks / instrumentation gaps: Rework tron-calib to score and display sample coverage, reject near-duplicate board poses, use OpenCV extended calibration APIs for per-view errors, prune bad views, and save stereo profiles only when RMS/epipolar quality is acceptable. Test findChessboardCornersSB and consider ChArUco or asymmetric circle grids if plain checkerboard remains fragile in IR.

Tags: #codex #calibration #stereo
