---
title: PipeWire wrapper does not expose separate camera path yet
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# PipeWire wrapper does not expose separate camera path yet

Timestamp: 2026-05-08T00:07:10-07:00

Question / hypothesis: Firefox webcam preview feels lower latency, so investigate whether PipeWire is providing a different camera path than direct V4L and whether pw-v4l2 improves tron-stream latency.

Method: Inspected PipeWire with wpctl status, pw-cli ls Node, and pw-cli ls Device. Also ran tron-stream through pw-v4l2 using Lenovo RGB /dev/video53 at 640x480 MJPG 30 fps, TurboJPEG decoder, one V4L buffer.

Observed result: PipeWire 1.4.10 is running and Firefox appears as a PipeWire client, but the current graph shows no Video devices, no Video sources, and no Video streams. Only ALSA audio devices for the camera hardware are visible as PipeWire devices. pw-v4l2 v4l2-ctl lists the same V4L device nodes. Running pw-v4l2 cargo run --bin tron-stream ... negotiated the same /dev/video53 640x480 MJPG path and showed similar camera cadence: about 30 fps, V4L timestamp delta around 33.3 ms, TurboJPEG decode around 0.8-1.0 ms. Render-side age sampled around 9-14 ms in that run, but this metric is phase-dependent and not proof of photon-to-display latency.

Interpretation: On the current system state, PipeWire is not obviously exposing the cameras as native Video/Source nodes. The pw-v4l2 wrapper did not reveal a clearly different capture path for tron-stream. Firefox may still be using direct V4L/WebRTC for webcam capture, or PipeWire video nodes may only appear while an active portal camera session is open.

Follow-up tasks / instrumentation gaps: Re-run wpctl status and pw-cli ls Node while Firefox is actively previewing the camera. If Video/Source nodes appear then, inspect their formats/latency params. If we want a true PipeWire backend, verify Gentoo PipeWire/WirePlumber camera monitor support first, then add a separate camera backend instead of relying on pw-v4l2.

Tags: #codex #latency #camera #pipewire #v4l
