"""Build the report charts as standalone SVG files from ``data/measurements.json``.

Standard library only. Each SVG carries its own light and dark palette as plain class rules
(dark switched by ``prefers-color-scheme``; a page with a theme toggle overrides the same classes
under its own selector), native hover tooltips on every mark, and a text summary for screen
readers. Plain colours instead of CSS custom properties keep the files renderable by non-browser
tools such as librsvg.

Palette: validated categorical slots 1 (blue) and 2 (orange) of the reference palette, in both
modes; ink, grid and surfaces from the same reference.

Usage: ``python docs/report/make_charts.py`` → writes ``docs/report/charts/*.svg``.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "measurements.json").read_text(encoding="utf-8"))
OUT = HERE / "charts"

W, H = 720, 360
PAD_L, PAD_R, PAD_T, PAD_B = 64, 28, 64, 56

STYLE = """
svg.viz { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.bg { fill: #fcfcfb; }
.title { fill: #0b0b0b; font-size: 16px; font-weight: 600; }
.sub { fill: #52514e; font-size: 12.5px; }
.tick { fill: #898781; font-size: 11.5px; font-variant-numeric: tabular-nums; }
.label { fill: #0b0b0b; font-size: 12.5px; font-variant-numeric: tabular-nums; }
.label-2 { fill: #52514e; font-size: 12px; }
.note { fill: #898781; font-size: 11.5px; }
.grid { stroke: #e1e0d9; stroke-width: 1; }
.axis { stroke: #c3c2b7; stroke-width: 1; }
.ref { stroke: #898781; stroke-width: 1.5; stroke-dasharray: 4 4; }
.s1 { fill: #2a78d6; }
.s2 { fill: #eb6834; }
.line-s1 { stroke: #2a78d6; stroke-width: 2; fill: none; }
.line-est { stroke: #2a78d6; stroke-width: 2; fill: none; stroke-dasharray: 6 5; }
.dot-est { fill: #fcfcfb; stroke: #2a78d6; stroke-width: 2; }
.mark:hover { opacity: 0.85; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; }
  .title, .label { fill: #ffffff; }
  .sub, .label-2 { fill: #c3c2b7; }
  .grid { stroke: #2c2c2a; }
  .axis { stroke: #383835; }
  .s1 { fill: #3987e5; }
  .s2 { fill: #d95926; }
  .line-s1, .line-est { stroke: #3987e5; }
  .dot-est { fill: #1a1a19; stroke: #3987e5; }
}
"""


def _svg(
    name: str, title: str, subtitle: str, summary: str, body: list[str], height: int = H
) -> None:
    doc = [
        f'<svg class="viz" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'role="img" aria-labelledby="{name}-t {name}-d" font-size="12">',
        f'<title id="{name}-t">{escape(title)}</title>',
        f'<desc id="{name}-d">{escape(summary)}</desc>',
        f"<style>{STYLE}</style>",
        f'<rect class="bg" x="0" y="0" width="{W}" height="{height}" rx="12"/>',
        f'<text class="title" x="{PAD_L - 40}" y="28">{escape(title)}</text>',
        f'<text class="sub" x="{PAD_L - 40}" y="47">{escape(subtitle)}</text>',
        *body,
        "</svg>",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text("\n".join(doc) + "\n", encoding="utf-8")


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _bar(x: float, y: float, w: float, h: float, cls: str, tip: str) -> str:
    """Bar with a 4px rounded data end, square at the baseline."""
    r = min(4.0, w / 2, h)
    if h <= 0:
        return ""
    path = (
        f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y + h:.1f} Z"
    )
    return f'<path class="mark {cls}" d="{path}"><title>{escape(tip)}</title></path>'


def _hbar(x: float, y: float, w: float, h: float, cls: str, tip: str) -> str:
    """Horizontal bar anchored at the left baseline, rounded at the data end."""
    r = min(4.0, h / 2, w)
    if w <= 0:
        return ""
    path = (
        f"M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} "
        f"L{x:.1f},{y + h:.1f} Z"
    )
    return f'<path class="mark {cls}" d="{path}"><title>{escape(tip)}</title></path>'


# --- 1. GPU memory versus frames ---------------------------------------------------------


def chart_memory() -> None:
    d = DATA["gpu_memory"]
    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = H - PAD_B, PAD_T + 10
    xmax, ymax = 88, 24

    def X(f: float) -> float:
        return x0 + (x1 - x0) * f / xmax

    def Y(g: float) -> float:
        return y0 - (y0 - y1) * g / ymax

    body = []
    for g in (0, 6, 12, 18, 24):
        body.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(g):.1f}" y2="{Y(g):.1f}"/>')
        body.append(
            f'<text class="tick" x="{x0 - 8}" y="{Y(g) + 4:.1f}" text-anchor="end">{g}</text>'
        )
    for f in (0, 24, 48, 80):
        body.append(
            f'<text class="tick" x="{X(f):.1f}" y="{y0 + 18}" text-anchor="middle">{f}</text>'
        )
    body.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    body.append(f'<text class="note" x="{x1}" y="{y0 + 38}" text-anchor="end">frames</text>')
    body.append(f'<text class="note" x="{x0 - 8}" y="{y1 - 14}" text-anchor="end">GiB</text>')

    avail = d["available_during_runs_gib"]
    body.append(f'<line class="ref" x1="{x0}" x2="{x1}" y1="{Y(avail):.1f}" y2="{Y(avail):.1f}"/>')
    body.append(
        f'<text class="label-2" x="{x0 + 6}" y="{Y(avail) - 7:.1f}">'
        f"available during the runs ≈ {avail:.0f} GiB</text>"
    )

    m = d["measured"]
    e = d["extrapolated"][0]
    body.append(
        f'<polyline class="line-s1" points="{X(m[0]["frames"]):.1f},{Y(m[0]["gib"]):.1f} '
        f'{X(m[1]["frames"]):.1f},{Y(m[1]["gib"]):.1f}"/>'
    )
    body.append(
        f'<polyline class="line-est" points="{X(m[1]["frames"]):.1f},{Y(m[1]["gib"]):.1f} '
        f'{X(e["frames"]):.1f},{Y(e["gib"]):.1f}"/>'
    )
    for p in m:
        tip = f"{p['frames']} frames: {p['gib']} GiB (measured)"
        body.append(
            f'<circle class="mark s1" cx="{X(p["frames"]):.1f}" cy="{Y(p["gib"]):.1f}" r="5">'
            f"<title>{tip}</title></circle>"
        )
        body.append(
            f'<text class="label" x="{X(p["frames"]) + 10:.1f}" y="{Y(p["gib"]) + 16:.1f}">'
            f"{p['gib']} GiB</text>"
        )
    tip = f"{e['frames']} frames: ≈ {e['gib']} GiB (extrapolated, not run)"
    body.append(
        f'<circle class="mark dot-est" cx="{X(e["frames"]):.1f}" cy="{Y(e["gib"]):.1f}" r="5">'
        f"<title>{tip}</title></circle>"
    )
    body.append(
        f'<text class="label" x="{X(e["frames"]) - 10:.1f}" y="{Y(e["gib"]) - 12:.1f}" '
        f'text-anchor="end">≈ {e["gib"]} GiB · estimate, not run</text>'
    )
    fit = d["linear_fit"]
    body.append(
        f'<text class="note" x="{x0}" y="{H - 10}">≈ {fit["intercept_gib"]:.0f} GiB + '
        f"{fit['per_frame_gib']:.2f} GiB per frame · {escape(d['gpu'])} · "
        "solid = measured, dashed = extrapolated</text>"
    )
    _svg(
        "gpu-memory",
        "GPU memory of the reconstructor grows linearly with frames",
        "VGGT-1B, measured on two frame budgets; the 80-frame point is an estimate",
        "Measured 7.4 GiB at 24 frames and 13.8 GiB at 48 frames. A linear fit, about 1 GiB "
        "plus 0.27 GiB per frame, predicts about 22 GiB at 80 frames, above the roughly 19 GiB "
        "that were available, so 80 frames were not run.",
        body,
    )


# --- 2. Confidence threshold versus points -----------------------------------------------


def chart_threshold() -> None:
    d = DATA["confidence_threshold"]
    cap = d["upstream_cap"]
    pts = d["points"]
    x0, x1 = PAD_L + 10, W - PAD_R
    y0, y1 = H - PAD_B, PAD_T + 18
    ymax = 110_000

    def Y(v: float) -> float:
        return y0 - (y0 - y1) * v / ymax

    body = []
    for v in (0, 25_000, 50_000, 75_000, 100_000):
        body.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        body.append(
            f'<text class="tick" x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">'
            f"{v // 1000}k</text>"
        )
    body.append(f'<line class="ref" x1="{x0}" x2="{x1}" y1="{Y(cap):.1f}" y2="{Y(cap):.1f}"/>')
    body.append(
        f'<text class="label-2" x="{x1}" y="{Y(cap) - 8:.1f}" text-anchor="end">'
        f"upstream cap · {_fmt(cap)} points</text>"
    )
    slot = (x1 - x0) / len(pts)
    bw = min(96.0, slot * 0.5)
    for i, p in enumerate(pts):
        cx = x0 + slot * (i + 0.5)
        h = y0 - Y(p["points"])
        value = f"≥ {_fmt(p['points'])}" if p["capped"] else _fmt(p["points"])
        tip = f"threshold {p['threshold']}: {value} points" + (
            " (at the cap: more passed)" if p["capped"] else ""
        )
        body.append(_bar(cx - bw / 2, Y(p["points"]), bw, h, "s1", tip))
        ly = Y(p["points"]) - (26 if p["capped"] else 10)
        body.append(
            f'<text class="label" x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle">{value}</text>'
        )
        body.append(
            f'<text class="tick" x="{cx:.1f}" y="{y0 + 18}" text-anchor="middle">'
            f"{p['threshold']}</text>"
        )
    body.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    body.append(
        f'<text class="note" x="{x1}" y="{y0 + 38}" text-anchor="end">'
        "confidence threshold (--conf_thres_value)</text>"
    )
    body.append(f'<text class="note" x="{PAD_L - 40}" y="{H - 10}">{escape(d["footnote"])}</text>')
    _svg(
        "confidence-threshold",
        "A small change of the threshold removes most points",
        f"{d['frames']} uniformly selected frames; bars at the cap mean more points passed",
        "At thresholds 1.3 and 1.6 the reconstruction reached the upstream cap of 100,000 points; "
        "at 1.8 only 9,346 points remained. The upstream default 5.0 gave no points on 24 frames.",
        body,
    )


# --- 3. P2: frame selection comparison ---------------------------------------------------


def chart_p2() -> None:
    d = DATA["p2"]
    cap = d["upstream_cap"]
    br = d["branches"]
    x0, x1 = PAD_L + 110, W - PAD_R - 70
    top = PAD_T + 26
    row = 62
    bh = 30
    xmax = 110_000

    def X(v: float) -> float:
        return x0 + (x1 - x0) * v / xmax

    body = []
    bottom = top + row * len(br) - (row - bh)
    for v in (0, 25_000, 50_000, 75_000, 100_000):
        body.append(
            f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 10}" '
            f'y2="{bottom + 8}"/>'
        )
        body.append(
            f'<text class="tick" x="{X(v):.1f}" y="{bottom + 26}" text-anchor="middle">'
            f"{v // 1000}k</text>"
        )
    body.append(
        f'<line class="ref" x1="{X(cap):.1f}" x2="{X(cap):.1f}" y1="{top - 16}" y2="{bottom + 8}"/>'
    )
    body.append(
        f'<text class="label-2" x="{X(cap):.1f}" y="{top - 22}" text-anchor="middle">'
        "upstream cap</text>"
    )
    for i, b in enumerate(br):
        y = top + row * i
        cls = "s2" if b["name"] == "quality" else "s1"
        value = f"≥ {_fmt(b['points'])}" if b["capped"] else _fmt(b["points"])
        tip = f"{b['label']}: {value} points at threshold {d['threshold']}"
        body.append(
            f'<text class="label" x="{x0 - 12}" y="{y + bh / 2 + 4:.1f}" text-anchor="end">'
            f"{escape(b['label'])}</text>"
        )
        body.append(_hbar(x0, y, X(b["points"]) - x0, bh, cls, tip))
        body.append(
            f'<text class="label" x="{X(b["points"]) + 8:.1f}" '
            f'y="{y + bh / 2 + 4:.1f}">{value}</text>'
        )
    body.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 10}" y2="{bottom + 8}"/>')
    height = bottom + 64
    body.append(
        f'<text class="note" x="{PAD_L - 40}" y="{height - 12}">'
        f"Same video, {d['frames']} frames, threshold {d['threshold']} (a stress test). "
        "Verdict by the rule fixed before the runs: improvement.</text>"
    )
    _svg(
        "p2-frame-selection",
        "Sharper, non-redundant frames keep the whole object",
        "Points that pass a strict confidence threshold, per frame-selection method",
        "Uniform selection kept 9,346 points and its repeat 9,297, a 0.5 percent difference. "
        "Quality, non-redundant selection reached the upstream cap of 100,000 points, so its "
        "true count is higher. The difference is far larger than run-to-run variation.",
        body,
        height=height,
    )


if __name__ == "__main__":
    chart_memory()
    chart_threshold()
    chart_p2()
    print("written:", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
