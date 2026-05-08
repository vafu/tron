---
title: "Camera set selection by name"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Camera set selection by name

2026-05-07T17:55:00-07:00

#tron #task

Question/hypothesis: Make it easy to choose a connected RGB/IR camera set by
name, e.g. `--camera Lenovo` or `--camera Nexigo`, rather than hard-coding
`/dev/video*` nodes.

Observed result: Added `camera::select` discovery. It enumerates V4L capture
nodes, filters by card/bus substring, picks YUYV as RGB and GREY as IR, and
prefers current demo sizes. Local `--list-cameras` output shows NexiGo RGB/IR
as `/dev/video49` + `/dev/video51`, and Lenovo RGB/IR as `/dev/video53` +
`/dev/video55`.

Observed result: `RuntimeConfig` now accepts `--camera NAME`, `--camera=NAME`,
and `--list-cameras`. The app logs the selected camera label, paths, and
dimensions before spawning streams. Without `--camera`, it keeps the old default
`/dev/video0` RGB + `/dev/video2` IR behavior.

Follow-up tasks or instrumentation gaps: IR/RGB calibration is still global.
Lenovo RGB/IR are both 640x480 while NexiGo IR is 640x360, so per-camera
calibration presets may be needed for reliable masking/depth metrics.

