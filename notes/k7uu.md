# Task investigate Windows Hello exposure controls

Timestamp: 2026-05-02T13:22:25-07:00

Tags: #tron #task

Task:

Investigate whether exposure, gain, emitter power, or strobe controls are available through standard V4L controls, UVC extension units, libcamera/IPU6 paths, or linux-enable-ir-emitter internals.

Why:

Distance estimation from IR intensity is weak without exposure control or at least exposure observability. Windows Hello-style devices may hide these controls behind proprietary interfaces.

Experiment:

- Enumerate standard V4L controls for the RGB and IR devices.
- Inspect UVC extension units exposed by the camera.
- Review linux-enable-ir-emitter control probing paths for reusable control discovery.
- Record which controls affect IR brightness, cadence, or exposure.

Output:

A device capability matrix and a recommendation: use IR for depth, use IR only for segmentation, or pursue deeper device-control hacking.

