---
title: "RGB IR calibration is depth dependent"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# RGB IR calibration is depth dependent

# RGB IR calibration is depth dependent

Timestamp: 2026-05-07T19:39:56-07:00

Tags: #tron

Question / hypothesis: Is one checkerboard-derived 2D RGB↔IR calibration enough
for reliable alignment across hand depths?

Observed result: User observed that the apparent RGB↔IR calibration changes
with object distance. A checkerboard pose at one depth can align that plane, but
objects closer/farther shift differently because the RGB and IR cameras have
separate viewpoints/FOVs. This makes a single 2D affine or homography
insufficient as a general calibration model.

Follow-up tasks / instrumentation gaps: Treat checkerboard calibration as a
plane-specific alignment. For hand tracking, model RGB↔IR mapping as
depth-dependent: collect calibration samples at multiple distances, use
estimated depth/proximity to choose/interpolate transforms, or move toward true
stereo calibration with intrinsics/extrinsics and a depth-aware projection.
