#!/usr/bin/env python3
"""
Probe V4L2/UVC cameras for exposure data.

This script tries three increasingly indirect paths:

1. Read visible V4L2 controls such as auto_exposure and exposure_time_absolute.
2. Locate likely companion UVC metadata nodes and attempt to capture UVCH data.
3. Parse any captured UVCH records for standard UVC payload-header fields.

Standard UVC payload headers do not carry actual exposure/gain values. They can
carry frame id, end-of-frame, PTS, and source clock reference. If this script
finds no metadata buffers, or only normal UVCH timing fields, hidden exposure
data likely requires vendor extension-unit probing.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DEVICES = ["/dev/video51", "/dev/video55"]
INTERESTING_CTRLS = [
    "auto_exposure",
    "exposure_time_absolute",
    "exposure_dynamic_framerate",
    "gain",
    "brightness",
    "contrast",
    "gamma",
    "backlight_compensation",
]


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def run(cmd: list[str], timeout: float = 5.0) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return CmdResult(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return CmdResult(124, e.stdout or "", e.stderr or f"timeout after {timeout}s")


def print_block(title: str) -> None:
    print(f"\n== {title} ==")


def v4l2_all(dev: str) -> str:
    r = run(["v4l2-ctl", "-d", dev, "--all"])
    if r.returncode != 0:
        print(f"{dev}: --all failed: {(r.stderr or r.stdout).strip()}")
        return ""
    return r.stdout


def is_metadata_node(dev: str) -> bool:
    text = v4l2_all(dev)
    return "Metadata Capture" in text and "Sample Format" in text


def read_controls(dev: str) -> dict[str, str]:
    out: dict[str, str] = {}
    listed = run(["v4l2-ctl", "-d", dev, "--list-ctrls-menus"]).stdout
    available = []
    for line in listed.splitlines():
        m = re.match(r"\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+", line)
        if m:
            available.append(m.group(1))

    for name in INTERESTING_CTRLS:
        if name not in available:
            continue
        r = run(["v4l2-ctl", "-d", dev, f"--get-ctrl={name}"])
        value = (r.stdout or r.stderr).strip()
        out[name] = value if value else f"<read failed rc={r.returncode}>"
    return out


def likely_metadata_nodes(dev: str) -> list[str]:
    m = re.match(r"^/dev/video(\d+)$", dev)
    candidates: list[str] = []
    if m:
        n = int(m.group(1))
        # UVC metadata nodes are commonly the adjacent node for the same
        # streaming interface: video55 -> video56, video51 -> video52.
        candidates.extend([f"/dev/video{n + 1}", f"/dev/video{n - 1}"])

    found: list[str] = []
    for cand in candidates:
        if cand in found or not Path(cand).exists():
            continue
        if is_metadata_node(cand):
            found.append(cand)
    return found


def capture_with_optional_video(
    video_dev: str,
    meta_dev: str,
    frames: int,
    timeout: float,
    keep_raw: Path | None,
) -> bytes:
    if keep_raw:
        meta_path = keep_raw
    else:
        fd, name = tempfile.mkstemp(prefix="uvch-", suffix=".raw")
        os.close(fd)
        meta_path = Path(name)

    meta_cmd = [
        "v4l2-ctl",
        "-d",
        meta_dev,
        "--stream-mmap",
        f"--stream-count={frames}",
        f"--stream-to={meta_path}",
    ]
    video_cmd = [
        "v4l2-ctl",
        "-d",
        video_dev,
        "--stream-mmap",
        f"--stream-count={frames}",
        "--stream-to=/dev/null",
    ]

    meta = subprocess.Popen(meta_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Give metadata stream a chance to queue buffers first, then start paired video.
    time.sleep(0.2)
    video = subprocess.Popen(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    deadline = time.monotonic() + timeout
    for proc in (video, meta):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    data = meta_path.read_bytes() if meta_path.exists() else b""
    if not keep_raw:
        meta_path.unlink(missing_ok=True)
    return data


def parse_uvch_linux_metadata(data: bytes) -> list[dict[str, object]]:
    """
    Parse Linux's V4L2_META_FMT_UVC layout best-effort.

    Kernel metadata records are:
      u64 ns; u16 sof; u8 length; u8 flags; u8 buf[];

    Here `length` and `flags` are the first two bytes of the UVC payload header
    (bHeaderLength and bmHeaderInfo). `buf[]` contains the remaining
    length - 2 bytes.
    """
    records: list[dict[str, object]] = []
    off = 0
    while off + 12 <= len(data):
        ns, sof, length, uvc_flags = struct.unpack_from("<QHBB", data, off)
        off += 12
        if length == 0:
            # Some drivers pad the rest of the capture buffer with zeros.
            break
        if length < 2 or length > 64 or off + (length - 2) > len(data):
            break
        header = bytes([length, uvc_flags]) + data[off : off + length - 2]
        off += length - 2

        rec: dict[str, object] = {
            "ns": ns,
            "sof": sof,
            "raw_header": header.hex(" "),
        }
        if len(header) >= 2:
            hlen = header[0]
            flags = header[1]
            if hlen != len(header):
                break
            rec["uvc_header_len"] = hlen
            rec["uvc_flags"] = flags
            rec["fid"] = bool(flags & 0x01)
            rec["eof"] = bool(flags & 0x02)
            rec["pts_present"] = bool(flags & 0x04)
            rec["scr_present"] = bool(flags & 0x08)
            p = 2
            if flags & 0x04 and len(header) >= p + 4:
                rec["pts"] = struct.unpack_from("<I", header, p)[0]
                p += 4
            if flags & 0x08 and len(header) >= p + 6:
                rec["scr_stc"] = struct.unpack_from("<I", header, p)[0]
                rec["scr_sof"] = struct.unpack_from("<H", header, p + 4)[0]
        records.append(rec)
    return records


def summarize_metadata(dev: str, meta_dev: str, frames: int, timeout: float, keep_dir: Path | None) -> None:
    raw_path = None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)
        raw_path = keep_dir / f"{Path(meta_dev).name}_uvch.raw"

    print(f"Capturing {frames} metadata buffers from {meta_dev} while {dev} streams...")
    data = capture_with_optional_video(dev, meta_dev, frames, timeout, raw_path)
    print(f"{meta_dev}: captured {len(data)} metadata bytes")
    if raw_path:
        print(f"{meta_dev}: raw metadata path {raw_path}")
    if not data:
        print(f"{meta_dev}: no metadata buffers dequeued")
        return

    records = parse_uvch_linux_metadata(data)
    print(f"{meta_dev}: parsed {len(records)} UVCH records")
    if not records:
        print(f"{meta_dev}: data was not recognized as Linux UVCH metadata")
        print(f"{meta_dev}: first bytes {data[:64].hex(' ')}")
        return

    for i, rec in enumerate(records[:10]):
        fields = [
            f"ns={rec.get('ns')}",
            f"sof={rec.get('sof')}",
            f"flags=0x{rec.get('uvc_flags', 0):02x}",
            f"fid={rec.get('fid')}",
            f"eof={rec.get('eof')}",
        ]
        if "pts" in rec:
            fields.append(f"pts={rec['pts']}")
        if "scr_stc" in rec:
            fields.append(f"scr={rec['scr_stc']}/{rec['scr_sof']}")
        fields.append(f"raw=[{rec.get('raw_header')}]")
        print(f"  record {i}: " + " ".join(fields))

    exposure_keys = [k for r in records for k in r.keys() if "exposure" in k.lower() or "gain" in k.lower()]
    if exposure_keys:
        print(f"{meta_dev}: exposure-like keys found: {sorted(set(exposure_keys))}")
    else:
        print(f"{meta_dev}: no standard exposure/gain values found in parsed UVCH records")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", nargs="*", default=DEFAULT_DEVICES, help="video nodes to probe")
    parser.add_argument("--frames", type=int, default=20, help="metadata/video frames to request")
    parser.add_argument("--timeout", type=float, default=6.0, help="seconds to wait for captures")
    parser.add_argument("--metadata", action="append", default=[], help="explicit video:metadata pair")
    parser.add_argument("--keep-raw", type=Path, help="directory to keep captured raw metadata")
    args = parser.parse_args()

    if run(["which", "v4l2-ctl"]).returncode != 0:
        print("v4l2-ctl not found in PATH", file=sys.stderr)
        return 2

    explicit_pairs: dict[str, list[str]] = {}
    for pair in args.metadata:
        if ":" not in pair:
            print(f"bad --metadata pair {pair!r}; expected /dev/videoX:/dev/videoY", file=sys.stderr)
            return 2
        video, meta = pair.split(":", 1)
        explicit_pairs.setdefault(video, []).append(meta)

    for dev in args.devices:
        print_block(dev)
        if not Path(dev).exists():
            print(f"{dev}: does not exist")
            continue

        controls = read_controls(dev)
        if controls:
            print("visible exposure/ISP controls:")
            for name, value in controls.items():
                print(f"  {value}")
        else:
            print("no visible exposure/ISP controls from the known control list")

        metas = explicit_pairs.get(dev) or likely_metadata_nodes(dev)
        if metas:
            print("candidate metadata nodes: " + ", ".join(metas))
        else:
            print("no adjacent metadata node found")
        for meta in metas:
            summarize_metadata(dev, meta, args.frames, args.timeout, args.keep_raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
