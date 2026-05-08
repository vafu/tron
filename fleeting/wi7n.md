---
title: MJPG timing split shows decode dominates color conversion
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# MJPG timing split shows decode dominates color conversion

Timestamp: 2026-05-07T23:17:19-07:00

Question / hypothesis: In the MJPG RGB path, how much time is JPEG decode versus BGR-to-BGRA conversion?

Observed result: tron-stream now logs wait, mjpg_decode, color, passthrough, and publish timings separately. Lenovo RGB at 1280x720 MJPG 30 fps measured wait about 28.7-28.9 ms/frame, mjpg_decode about 2.75-2.96 ms/frame, BGR-to-BGRA color conversion about 0.40-0.48 ms/frame after warmup, publish about 0.27-0.29 ms/frame, upload about 0.11-0.13 ms/frame, and render about 0.09-0.10 ms/frame.

Interpretation: JPEG decode dominates the CPU conversion cost. BGR-to-BGRA is not free, but it is under 0.5 ms/frame and much smaller than MJPG decode. The remaining frame time is camera wait for 30 fps.

Follow-up tasks: Keep split timing instrumentation while integrating MJPG into the shared backend. If optimizing further, target JPEG decode first, e.g. libjpeg-turbo direct BGRA decode or decoder reuse/preallocation.

Tags: #codex #camera #performance #mjpg #timing
