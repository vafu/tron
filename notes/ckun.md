---
title: "Task test IR masked RGB tracking"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Task test IR masked RGB tracking

Timestamp: 2026-05-02T16:50:58-07:00

Tags: #tron #task

Question / hypothesis:

MediaPipe tracking regressed in the current indoor setup. Reintroducing the IR
foreground mask as an auxiliary RGB refiner may improve ROI acquisition and
landmark stability by dimming background pixels while preserving the generic
RGB/IR `FrameContext` API.

Observed result:

Wired `RgbMaskingRefiner` back into the active refiner chain after flashlight
detection and temporal IR subtraction. ROI and landmark stages now consume the
masked RGB image, while the camera layer still captures both RGB and IR frames
into `FrameContext`.

Follow-up tasks / instrumentation gaps:

- Run live comparison indoors with and without masking if tracking is still
  unstable.
- Watch whether the masked debug pane keeps the hand bright during IR strobe
  off frames.
- If masking over-darkens the hand, tune `RgbMaskingRefiner` mask floor and
  foreground gain.

2026-05-02T16:56:41-07:00

- Added shared pipeline controls for runtime experiments.
- Pressing `I` toggles `RgbMaskingRefiner` without restarting the app.
- Window title now shows `ir-mask:on` or `ir-mask:off` for quick confirmation.
