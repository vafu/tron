---
title: "Per camera affine calibration profiles"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Per camera affine calibration profiles

# Per camera affine calibration profiles

Timestamp: 2026-05-07T18:54:45-07:00

Tags: #tron #task

Question / hypothesis: RGB and IR streams have different aspect/FOV per camera
set, so a single hardcoded affine calibration should be replaced with
camera-specific calibration profiles.

Observed result: Added calibration initialization from the selected camera label
and stream sizes. Startup now loads `calib/<camera-slug>.calib` when present, or
uses an aspect-aware centered default. Existing keyboard controls still tune
offset/scale live; `O` saves the current affine profile and `R` resets to the
current profile default. The HTTP diagnostics page now publishes the current
calibration values while tuning. `cargo check --bin tron` passes.

Follow-up tasks / instrumentation gaps: Add visual calibration overlay and
publish calibration values in HTTP diagnostics. The affine profile is only a
first-order model; measured RGB->IR point pairs or a homography will likely be
needed for reliable tracking across the full image.
