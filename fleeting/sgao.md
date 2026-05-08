---
title: Firefox uses four V4L mmap buffers for Lenovo MJPG preview
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Firefox uses four V4L mmap buffers for Lenovo MJPG preview

Timestamp: 2026-05-08T00:16:55-07:00

Question / hypothesis: Determine Firefox's V4L buffer strategy for the low-latency Lenovo camera preview.

Observed result: strace of active Firefox camera preview on /dev/video53 showed VIDIOC_DQBUF/VIDIOC_QBUF on V4L2_BUF_TYPE_VIDEO_CAPTURE with V4L2_MEMORY_MMAP. Buffer indices rotate 0, 1, 2, 3, which proves Firefox has at least four mmap capture buffers active. Buffer length is 921600 bytes, bytesused is roughly 155k-161k for MJPG frames, and timestamps are V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC with V4L2_BUF_FLAG_TSTAMP_SRC_SOE. Timestamp deltas are about 32-36 ms, matching 30 fps cadence. The active mode previously observed was /dev/video53 1280x720 MJPG 30 fps.

Interpretation: Firefox's apparently good latency is not due to a single-buffer V4L queue. It uses a four-buffer mmap ring and immediately requeues each dequeued buffer. If Firefox feels faster than tron-stream, likely causes are render/display path, scheduling/thread priority, compositor behavior, WebRTC frame dropping/preview pipeline, or subjective phase differences rather than V4L buffer count alone.

Follow-up tasks / instrumentation gaps: Compare tron-stream at the exact Firefox mode with --buffers 4 as the baseline, then test --buffers 1 and --drain-latest subjectively and with future photon-to-display rig. If we need complete setup details, trace from camera start to capture VIDIOC_REQBUFS and STREAMON.

Tags: #codex #latency #camera #firefox #v4l
