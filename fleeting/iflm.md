---
title: Rust stream negotiates MJPG for RGB latency
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Rust stream negotiates MJPG for RGB latency

Timestamp: 2026-05-07T23:04:27-07:00

Question / hypothesis: Can tron-stream match the low-latency Firefox/ffplay path by negotiating RGB as MJPG instead of YUYV?

Observed result: tron-stream now supports --format mjpg|yuyv|grey and defaults RGB to MJPG while IR remains GREY. Lenovo RGB test with --camera Lenovo --sensor rgb --format mjpg --size 1280x720 --fps 30 negotiated /dev/video53 as 1280x720 MJPG and accepted interval 1/30. Runtime delivered about 30 fps. OpenCV imgcodecs decode plus BGR-to-RGBA conversion cost about 4-5 ms/frame. GPU upload was about 0.1 ms/frame and render about 0.08-0.11 ms/frame. Frame age was generally low, roughly single-digit to low-20 ms depending on vsync phase.

Interpretation: The RGB lag was not inherent to Rust rendering. The usable RGB path is compressed MJPG at 30 fps; YUYV transport at useful RGB resolutions is too slow on this camera path. For the research pipeline, RGB should publish a decoded current frame/texture from MJPG while IR should remain raw GREY. Avoid pane-level clones, but allow a required camera-backend decode stage for compressed RGB.

Follow-up tasks: Promote capture format into shared camera selection/config, add MJPG support to the main camera backend rather than only tron-stream, and include capture format/resolution/fps in calibration profile identity.

Tags: #codex #camera #performance #latency #mjpg
