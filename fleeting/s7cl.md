---
title: Single V4L buffer improves photon-to-frame latency
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Single V4L buffer improves photon-to-frame latency

Timestamp: 2026-05-07T23:51:28-07:00

Question / hypothesis: Firefox webcam preview feels near-instant while direct V4L/Rust tests lag. Test whether V4L mmap buffer count affects photon-to-frame latency even when Rust-side publish/render timestamps do not show obvious backlog.

Method: Added --buffers to src/bin/tron-stream.rs and passed it to MmapStream::with_buffers. Logged V4L timestamp deltas, sequence number, sequence gaps, and metadata.bytesused-based payload sizes. Tested Lenovo RGB /dev/video53 at 640x480 MJPG 30 fps with TurboJPEG decoder using buffer counts 1, 2, and 4.

Observed result: Internal metrics showed all buffer counts stabilizing near 30 fps, V4L timestamp delta around 33.3 ms, no steady sequence gaps, and MJPG decode around 0.8-1.5 ms. However, user observation from the live scene was that a single V4L buffer has much better photon-to-frame latency than larger buffer counts.

Interpretation: The Rust-side frame age metric only measures time since publish into the renderer, not true sensor photon-to-display latency. V4L sequence metadata did not expose a multi-frame backlog, but the live visual result indicates mmap queue depth still affects end-to-end freshness. For latency-sensitive HID experiments, default the debug stream to one mmap buffer and keep buffer count configurable.

Follow-up tasks / instrumentation gaps: Build a real photon-to-display measurement path, for example LED or high-speed external-camera test, because internal timestamps are insufficient. Consider testing whether the main capture backend should also request one buffer or expose a low-latency capture policy.

Tags: #codex #latency #camera #v4l #streaming
