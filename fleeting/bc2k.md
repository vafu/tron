---
title: Firefox likely avoids RGB lag by using compressed camera modes
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Firefox likely avoids RGB lag by using compressed camera modes

Timestamp: 2026-05-07T22:53:31-07:00

Question / hypothesis: Why does ffplay/Rust RGB stream lag while webcam meetings in Firefox feel fine?

Observed result: The laggy tests have been using YUYV RGB modes. Lenovo advertises high-resolution MJPG modes at 30 fps, while YUYV modes drop sharply with resolution: 1024x576 is advertised around 15 fps and observed around 12.5 fps; 800x448 accepts a 1/30 interval request but observed delivery stayed around 20.6 fps. Firefox/WebRTC likely negotiates MJPG or another compressed browser-friendly mode, possibly at a different resolution, and keeps its capture queue shallow. That would explain why Firefox feels fine while YUYV ffplay/tron-stream lags.

Interpretation: The problem is probably not Rust rendering. It is mode negotiation and camera transport format. For RGB, YUYV is predictable and shader-friendly but may be too slow on this camera/USB path. MJPG likely gives the browser path its usable latency but requires decode before GPU/render/model input.

Follow-up tasks: Test ffplay with explicit MJPG low-latency options; add MJPG mode support or a separate compressed RGB backend if measurements confirm it; keep IR GREY path raw; expose selected fourcc in camera selection instead of assuming RGB means YUYV.

Tags: #codex #camera #performance #latency
