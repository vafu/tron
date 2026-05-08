---
title: OpenCV RGB decode still needs RGBA upload
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# OpenCV RGB decode still needs RGBA upload

Timestamp: 2026-05-07T23:09:51-07:00

Question / hypothesis: Can we avoid OpenCV BGR-to-RGBA by decoding MJPG directly as RGB, and does wgpu support a plain RGB upload texture?

Observed result: wgpu 22 does not expose a plain sampled Rgb8Unorm/Rgb8 texture format. It has Rgba8/Bgra8, packed RGB formats like Rgb9e5/Rgb10a2, and compressed RGB texture formats, but no ordinary 24-bit RGB sampled texture. OpenCV exposes IMREAD_COLOR_RGB, and tron-stream was tested with MJPG decoded directly to RGB followed by manual RGB-to-RGBA alpha padding. Lenovo 1280x720 MJPG still delivered 30 fps, but decode/padding cost rose to about 6.0-6.3 ms/frame. The previous OpenCV IMREAD_COLOR BGR decode plus OpenCV cvtColor(BGR2RGBA) measured about 4-5 ms/frame, so the debug stream was restored to the faster BGR-to-RGBA path.

Interpretation: OpenCV can decode MJPG, and direct RGB output works, but the renderer still needs a 4-channel upload format unless we introduce a custom packed/compute path. For now, OpenCV's optimized color conversion is faster than manual RGB padding.

Follow-up tasks: For production, evaluate libjpeg-turbo output directly to RGBA/BGRA to avoid a second OpenCV color pass. If we keep OpenCV, consider preallocating/reusing Mats and benchmarking cvtColor separately from imdecode.

Tags: #codex #camera #performance #mjpg #wgpu
