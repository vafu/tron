---
title: Photon to action latency is a core research metric
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Photon to action latency is a core research metric

Timestamp: 2026-05-07T23:37:40-07:00

Question / hypothesis: What should be optimized after MJPG decode/render costs are reduced?

Observed result: At 640x480 MJPG, decode is under 1 ms, but visible lag can still remain. This implies the camera hardware/firmware/USB/UVC path may dominate latency before frames reach the application. MJPG encoding happens on the camera side, so V4L can receive a frame that is already old due to exposure/readout, ISP processing, encode buffering, USB transfer, and driver queueing.

Research goal: Optimize photon-to-action latency, not just application-side FPS or decode time. Photon-to-action means the full path from real-world motion/light hitting the sensor, through capture/decode/tracking/filtering/gesture interpretation, to emitted pointer/action update.

Follow-up tasks: Add instrumentation and test rigs for end-to-end latency. Track frame age, camera queue depth, decode time, inference time, filter delay, action emission time, and display/compositor delay separately. Compare camera modes, resolutions, formats, buffer counts, and devices by photon-to-action latency rather than FPS alone.

Tags: #codex #camera #performance #latency #research
