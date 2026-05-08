---
title: TurboJPEG reduces MJPG decode latency
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# TurboJPEG reduces MJPG decode latency

Timestamp: 2026-05-07T23:26:57-07:00

Question / hypothesis: Does libjpeg-turbo decode Lenovo MJPG camera frames faster than OpenCV imdecode + BGR2BGRA?

Observed result: tron-stream now supports --decoder opencv|turbojpeg for MJPG. TurboJPEG uses a reusable Decompressor and decodes directly to BGRA via turbojpeg::PixelFormat::BGRA. Lenovo RGB /dev/video53 at 1280x720 MJPG 30 fps delivered 30 fps. TurboJPEG measured mjpg_decode about 2.0-2.5 ms/frame after warmup with color=0, publish about 0.23-0.31 ms, upload commonly about 0.12-0.15 ms. Some decode/publish/upload spikes occurred later in the run. OpenCV on the same mode measured mjpg_decode about 2.7-3.1 ms/frame plus color about 0.42-0.54 ms/frame.

Interpretation: TurboJPEG is a clear improvement for the RGB path because it removes OpenCV's extra color conversion and usually reduces decode time. Approximate warm-path saving is around 1.0-1.5 ms/frame versus OpenCV. Remaining CPU decode cost is still meaningful, but this is the best current portable path.

Follow-up tasks: Keep TurboJPEG as default for tron-stream MJPG; promote decoder selection into the main camera backend; investigate jitter sources and preallocated publish buffers; compare lower MJPG modes for HID latency/landmark quality tradeoff.

Tags: #codex #camera #performance #mjpg #turbojpeg
