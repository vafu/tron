#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/camera-probe.sh --rgb /dev/videoX --rgb-meta /dev/videoY --ir /dev/videoZ --ir-meta /dev/videoW [options]
  tools/camera-probe.sh --list

Options:
  --name NAME          Label used in the report directory.
  --rgb DEV           RGB capture node.
  --rgb-meta DEV      RGB metadata node.
  --ir DEV            IR capture node.
  --ir-meta DEV       IR metadata node.
  --out DIR           Output directory. Default: /tmp/tron-camera-probe-<timestamp>.
  --stream-count N    Number of video/metadata buffers for short samples. Default: 30.
  --list              List V4L2 devices and exit.
  -h, --help          Show this help.

The script writes a text report plus short metadata dumps. It is intended for
Windows Hello camera comparison: node layout, formats, controls, metadata
formats, frame illumination flags, and basic streaming cadence.
EOF
}

die() {
  echo "camera-probe: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

timestamp() {
  date --iso-8601=seconds
}

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_'
}

run_report() {
  local title="$1"
  shift
  {
    echo
    echo "## $title"
    echo "\$ $*"
  } >>"$report"
  "$@" >>"$report" 2>&1 || {
    local status=$?
    echo "command failed with status $status" >>"$report"
    return 0
  }
}

probe_node() {
  local label="$1"
  local dev="$2"

  [[ -n "$dev" ]] || return 0
  [[ -e "$dev" ]] || {
    echo "## $label" >>"$report"
    echo "$dev does not exist" >>"$report"
    return 0
  }

  run_report "$label info" v4l2-ctl -d "$dev" --info
  run_report "$label all" v4l2-ctl -d "$dev" --all
  run_report "$label video formats" v4l2-ctl -d "$dev" --list-formats-ext
  run_report "$label metadata formats" v4l2-ctl -d "$dev" --list-formats-meta --get-fmt-meta
  run_report "$label controls" v4l2-ctl -d "$dev" --list-ctrls-menus
}

sample_video() {
  local label="$1"
  local dev="$2"
  local log="$out_dir/${label}-video-stream.log"

  [[ -n "$dev" && -e "$dev" ]] || return 0

  {
    echo
    echo "## $label short video stream"
    echo "\$ v4l2-ctl -d $dev --stream-mmap=2 --stream-count=$stream_count --stream-to=/dev/null --verbose"
  } >>"$report"

  v4l2-ctl -d "$dev" \
    --stream-mmap=2 \
    --stream-count="$stream_count" \
    --stream-to=/dev/null \
    --verbose >"$log" 2>&1 || true

  sed -n '1,120p' "$log" >>"$report"
}

sample_metadata_pair() {
  local label="$1"
  local video_dev="$2"
  local meta_dev="$3"
  local format="$4"

  [[ -n "$video_dev" && -n "$meta_dev" ]] || return 0
  [[ -e "$video_dev" && -e "$meta_dev" ]] || return 0

  local meta_file="$out_dir/${label}-${format}.bin"
  local meta_log="$out_dir/${label}-${format}.log"
  local video_log="$out_dir/${label}-${format}-paired-video.log"

  {
    echo
    echo "## $label paired metadata stream ($format)"
    echo "\$ v4l2-ctl -d $meta_dev --set-fmt-meta=$format --get-fmt-meta"
  } >>"$report"

  if ! v4l2-ctl -d "$meta_dev" --set-fmt-meta="$format" --get-fmt-meta >>"$report" 2>&1; then
    echo "metadata format $format is not available on $meta_dev" >>"$report"
    return 0
  fi

  v4l2-ctl -d "$video_dev" \
    --stream-mmap=2 \
    --stream-count="$stream_count" \
    --stream-to=/dev/null >"$video_log" 2>&1 &
  local video_pid=$!

  sleep 0.3

  v4l2-ctl -d "$meta_dev" \
    --stream-mmap=4 \
    --stream-count=10 \
    --stream-to="$meta_file" \
    --verbose >"$meta_log" 2>&1 || true

  wait "$video_pid" || true

  {
    echo
    echo "### metadata stream log"
    sed -n '1,160p' "$meta_log"
    echo
    echo "### paired video stream log"
    sed -n '1,80p' "$video_log"
    echo
    echo "### metadata dump"
    ls -l "$meta_file" 2>/dev/null || true
    od -Ax -tx1 -N 512 "$meta_file" 2>/dev/null || true
  } >>"$report"

  if [[ "$format" == "UVCM" && -s "$meta_file" ]]; then
    parse_uvcm "$meta_file" >>"$report" 2>&1 || true
  fi
}

parse_uvcm() {
  local file="$1"
  python3 - "$file" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()

print()
print("### parsed UVCM metadata")
print(f"file={path} bytes={len(data)}")

off = 0
blocks = 0
items = 0
illum = []
while off + 12 <= len(data):
    ts, sof = struct.unpack_from("<QH", data, off)
    length = data[off + 10]
    flags = data[off + 11]
    if length < 2 or off + 10 + length > len(data):
        off += 1
        continue

    blocks += 1
    if length > 12:
        extra = data[off + 22 : off + 10 + length]
        e = 0
        while e + 8 <= len(extra):
            metadata_id, size = struct.unpack_from("<II", extra, e)
            if size < 8 or e + size > len(extra):
                break
            payload = extra[e + 8 : e + size]
            items += 1
            if metadata_id == 6 and len(payload) >= 8:
                frame_flags, reserved = struct.unpack_from("<II", payload, 0)
                illum.append(frame_flags & 1)
                print(
                    "frame_illumination "
                    f"block={blocks - 1} offset={off} ts={ts} sof={sof} "
                    f"uvc_flags=0x{flags:02x} on={frame_flags & 1} "
                    f"raw_flags=0x{frame_flags:08x} reserved=0x{reserved:08x}"
                )
            else:
                print(
                    "metadata_item "
                    f"block={blocks - 1} offset={off} ts={ts} sof={sof} "
                    f"id={metadata_id} size={size} payload={payload.hex()}"
                )
            e += size

    off += 10 + length

print(f"blocks={blocks} metadata_items={items}")
if illum:
    print(f"illumination_sequence={','.join(str(v) for v in illum[:64])}")
    print(f"illumination_on_count={sum(illum)} illumination_total={len(illum)}")
else:
    print("illumination_sequence=<none>")
PY
}

name=""
rgb=""
rgb_meta=""
ir=""
ir_meta=""
out_dir=""
stream_count=30
list_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      name="${2:-}"
      shift 2
      ;;
    --rgb)
      rgb="${2:-}"
      shift 2
      ;;
    --rgb-meta)
      rgb_meta="${2:-}"
      shift 2
      ;;
    --ir)
      ir="${2:-}"
      shift 2
      ;;
    --ir-meta)
      ir_meta="${2:-}"
      shift 2
      ;;
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --stream-count)
      stream_count="${2:-}"
      shift 2
      ;;
    --list)
      list_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

require_cmd v4l2-ctl
require_cmd date
require_cmd od
require_cmd python3

if [[ "$list_only" == 1 ]]; then
  v4l2-ctl --list-devices
  exit 0
fi

[[ -n "$rgb$rgb_meta$ir$ir_meta" ]] || die "provide at least one device node, or use --list"

if [[ -z "$name" ]]; then
  name="camera"
fi

if [[ -z "$out_dir" ]]; then
  out_dir="/tmp/tron-camera-probe-$(safe_name "$name")-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$out_dir"
report="$out_dir/report.txt"

{
  echo "# tron camera probe"
  echo "timestamp: $(timestamp)"
  echo "name: $name"
  echo "out_dir: $out_dir"
  echo "rgb: ${rgb:-<none>}"
  echo "rgb_meta: ${rgb_meta:-<none>}"
  echo "ir: ${ir:-<none>}"
  echo "ir_meta: ${ir_meta:-<none>}"
  echo "stream_count: $stream_count"
} >"$report"

run_report "v4l2 devices" v4l2-ctl --list-devices
probe_node "RGB capture" "$rgb"
probe_node "RGB metadata" "$rgb_meta"
probe_node "IR capture" "$ir"
probe_node "IR metadata" "$ir_meta"

sample_video "rgb" "$rgb"
sample_video "ir" "$ir"
sample_metadata_pair "rgb-meta" "$rgb" "$rgb_meta" "UVCH"
sample_metadata_pair "ir-meta-uvch" "$ir" "$ir_meta" "UVCH"
sample_metadata_pair "ir-meta-uvcm" "$ir" "$ir_meta" "UVCM"

echo "camera-probe: wrote $report"
