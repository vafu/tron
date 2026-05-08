---
title: Use Firefox-matching Lenovo stream defaults
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Use Firefox-matching Lenovo stream defaults

Timestamp: 2026-05-08T00:18:59-07:00

Question / hypothesis: Which tron-stream defaults should we keep now that latency is acceptable for the current research phase?

Observed result: Firefox's acceptable preview path uses direct V4L on Lenovo /dev/video53 with 1280x720 MJPG at 30 fps and a four-buffer mmap ring. tron-stream already defaults RGB MJPG without explicit size to 1280x720 at 30 fps. Updated tron-stream's default --buffers value back to 4 to match Firefox, while keeping --buffers and --drain-latest available for future latency experiments.

Follow-up tasks / instrumentation gaps: Move on from latency tuning for now. Later, use a physical photon-to-display measurement rig before changing latency policy in the main camera backend.

Tags: #codex #latency #camera #v4l #streaming
