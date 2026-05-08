pub const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tron diagnostics</title>
  <style>
    :root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    body { margin: 0; background: #0b0d10; color: #d8f3ff; }
    main { max-width: 1160px; margin: 0 auto; padding: 18px; }
    h1 { margin: 0 0 14px; font-size: 18px; font-weight: 650; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
    section { border: 1px solid #1c3440; background: #10171d; border-radius: 6px; padding: 12px; }
    h2 { margin: 0 0 10px; font-size: 13px; color: #77dfff; }
    dl { display: grid; grid-template-columns: minmax(110px, 1fr) minmax(90px, 1.2fr); gap: 5px 12px; margin: 0; }
    dt { color: #7793a0; }
    dd { margin: 0; text-align: right; color: #f1fbff; }
    canvas { width: 100%; height: 180px; background: #080b0e; border: 1px solid #1c3440; border-radius: 4px; }
    .stale { color: #ffb86b; }
  </style>
</head>
<body>
<main>
  <h1>tron diagnostics <span id="age" class="stale"></span></h1>
  <div class="grid">
    <section><h2>Pipeline</h2><dl id="pipeline"></dl></section>
    <section><h2>Calibration</h2><dl id="calibration"></dl></section>
    <section><h2>Hand</h2><dl id="hand"></dl></section>
    <section><h2>IR Depth</h2><dl id="depth"></dl></section>
    <section><h2>Pointer</h2><dl id="pointer"></dl></section>
  </div>
  <section style="margin-top:12px"><h2>Corrected Signal</h2><canvas id="chart" width="900" height="180"></canvas></section>
</main>
<script>
const history = [];
function fmt(v, n = 3) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(n);
  return String(v);
}
function rows(id, pairs) {
  const el = document.getElementById(id);
  el.innerHTML = pairs.map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("");
}
function draw(value) {
  if (value !== null && value !== undefined) {
    history.push(value);
    if (history.length > 240) history.shift();
  }
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#203a46";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = (i + 1) * c.height / 5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
  }
  if (history.length < 2) return;
  const max = Math.max(1, ...history);
  ctx.strokeStyle = "#76f7ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = i * c.width / Math.max(1, history.length - 1);
    const y = c.height - (v / max) * (c.height - 12) - 6;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
async function tick() {
  try {
    const s = await fetch("/state.json", { cache: "no-store" }).then(r => r.json());
    const age = Date.now() - s.updated_unix_ms;
    document.getElementById("age").textContent = age > 500 ? `stale ${age}ms` : "";
    rows("pipeline", [
      ["frame", s.frame],
      ["rgb seq", s.rgb_seq],
      ["ir seq", s.ir_seq],
      ["ir diff seq", s.ir_diff_seq],
      ["interval ms", s.interval_ms],
      ["input age ms", s.input_age_ms],
      ["proximity", s.proximity],
    ]);
    rows("hand", s.hand ? [
      ["gesture", s.hand.gesture],
      ["presence", s.hand.presence],
      ["handedness", s.hand.handedness],
      ["roi", `${fmt(s.hand.roi.x,2)},${fmt(s.hand.roi.y,2)} ${fmt(s.hand.roi.w,2)}x${fmt(s.hand.roi.h,2)}`],
      ["features", s.hand.gesture_features],
    ] : [["state", "none"]]);
    rows("calibration", s.calibration ? [
      ["scale x", s.calibration.scale_x],
      ["scale y", s.calibration.scale_y],
      ["offset x", s.calibration.offset_x],
      ["offset y", s.calibration.offset_y],
      ["binary", s.calibration.use_binary],
    ] : [["state", "none"]]);
    rows("depth", s.ir_depth ? [
      ["corrected", s.ir_depth.corrected_signal],
      ["delta", s.ir_depth.delta],
      ["confidence", s.ir_depth.confidence],
      ["clip", s.ir_depth.clip_fraction],
      ["hand mean", s.ir_depth.hand_diff_mean],
      ["hand median", s.ir_depth.hand_diff_median],
      ["bg mean", s.ir_depth.background_diff_mean],
      ["bg median", s.ir_depth.background_diff_median],
      ["raw hand", s.ir_depth.raw_hand_mean],
    ] : [["state", "none"]]);
    rows("pointer", s.pointer ? [
      ["x", s.pointer.x],
      ["y", s.pointer.y],
      ["z", s.pointer.z],
      ["grabbed", s.pointer.grabbed],
      ["confidence", s.pointer.confidence],
    ] : [["state", "none"]]);
    draw(s.ir_depth && s.ir_depth.corrected_signal);
  } catch (e) {
    document.getElementById("age").textContent = "disconnected";
  }
}
setInterval(tick, 100);
tick();
</script>
</body>
</html>
"##;
