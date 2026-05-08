---
title: Streaming architecture should separate compressed capture from decoded processin...
tags:
- project/tron
- kind/fleeting
- status/inbox
remarked_pipeline: captured
remarked_processor_state: none
---
# Streaming architecture should separate compressed capture from decoded processin...

Timestamp: 2026-05-08T00:22:52-07:00

Question / hypothesis: For a proper tron streaming pipeline, can processing stages such as convex hull run directly on MJPG/JPEG camera payloads, or should capture/decode/processing be separate re-pluggable traits?

Observed result: OpenCV can decode JPEG/MJPG frames via imgcodecs/imdecode and can decode directly to grayscale or color Mats, but normal image processing operations such as thresholding, contours, convex hull, morphology, ROI masking, landmark pre-processing, and rendering upload operate on decoded pixel buffers/Mats, not compressed JPEG entropy data. JPEG-domain processing exists in specialized cases but is not appropriate for tron's hand/IR/RGB pipeline.

Architecture direction: Treat camera capture as producing encoded or raw frame packets with metadata. Treat decoding/color conversion as its own pluggable stage that produces decoded frame views/Mats/textures. Downstream processing traits should consume decoded image views, while stages that only need metadata can consume frame metadata without forcing decode. Keep the playground tron-stream binary as an experiment harness, but move reusable pieces into modules for main tron.

Follow-up tasks / instrumentation gaps: Define traits for FrameSource, FrameDecoder, FrameProcessor/Analyzer, FramePublisher, and RenderSink. Ensure ownership/borrowing allows zero-copy or minimal-copy paths where practical, especially for raw YUYV/GREY and decoded MJPG buffers.

Tags: #codex #architecture #streaming #camera #opencv
