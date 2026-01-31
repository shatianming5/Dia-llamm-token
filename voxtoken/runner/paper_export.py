from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..eval.pareto import pareto_front


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_jsonl_many(paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        out.extend(_load_jsonl(Path(p)))
    return out


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def _stable_int_hash(s: str) -> int:
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % 2147483647
    return int(h)


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    q = float(q)
    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
    idx = int(round(q * float(len(sorted_vals) - 1)))
    idx = max(0, min(int(idx), int(len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _bootstrap_ci_mean(vals: List[float], *, n_boot: int, alpha: float, seed: int) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    n_boot = max(200, int(n_boot))
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)

    n = int(len(vals))
    rng = random.Random(int(seed))
    boot: List[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _k in range(n):
            s += float(vals[rng.randrange(n)])
        boot.append(float(s) / float(n))
    boot.sort()
    return {
        "mean": float(_mean(vals)),
        "ci_low": float(_quantile(boot, float(alpha) / 2.0)),
        "ci_high": float(_quantile(boot, 1.0 - float(alpha) / 2.0)),
    }


def _group_means(metrics_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in metrics_rows:
        method = str(r.get("method", "")).strip()
        budget = int(r.get("budget_B", 0) or 0)
        if not method or budget <= 0:
            continue
        groups.setdefault((method, budget), []).append(r)

    out: List[Dict[str, Any]] = []
    for (method, budget_B), rs in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        out.append(
            {
                "method": str(method),
                "budget_B": int(budget_B),
                "n": int(len(rs)),
                "ground_mean_iou_mean": _mean([float(x.get("ground_mean_iou", 0.0)) for x in rs]),
                "ground_hit@0.1_mean": _mean([float(x.get("ground_hit@0.1", 0.0)) for x in rs]),
                "tokens_used_mean": _mean([float(x.get("tokens_used", 0.0)) for x in rs]),
                "latency_ms.total_mean": _mean([float((x.get("latency_ms") or {}).get("total", 0.0)) for x in rs]),
            }
        )
    return out


def _aggregate_by_case(metrics_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in metrics_rows:
        case_id = str(r.get("case_id", "")).strip()
        method = str(r.get("method", "")).strip()
        budget = int(r.get("budget_B", 0) or 0)
        if not case_id or not method or budget <= 0:
            continue
        by_key.setdefault((case_id, method, int(budget)), []).append(r)

    out: List[Dict[str, Any]] = []
    for (case_id, method, budget_B), rs in sorted(by_key.items(), key=lambda x: (x[0][1], int(x[0][2]), x[0][0])):
        best = None
        best_score = -1.0
        for r in rs:
            try:
                s = float(r.get("ground_mean_iou", 0.0))
            except Exception:
                s = 0.0
            if s > best_score and r.get("best_token_box_mm", None) is not None:
                best_score = float(s)
                best = r
        if best is None:
            best = rs[0]

        def mean_key(k: str) -> float:
            vals: List[float] = []
            for r in rs:
                try:
                    vals.append(float(r.get(k, 0.0)))
                except Exception:
                    continue
            return float(_mean(vals))

        lat_vals: List[float] = []
        for r in rs:
            try:
                lat_vals.append(float((r.get("latency_ms") or {}).get("total", 0.0)))
            except Exception:
                continue

        seeds = sorted({int(r.get("seed", 0) or 0) for r in rs})

        out.append(
            {
                "case_id": str(case_id),
                "method": str(method),
                "budget_B": int(budget_B),
                "seed_n": int(len(seeds)),
                "seeds": list(seeds),
                "ground_mean_iou": float(mean_key("ground_mean_iou")),
                "ground_hit@0.1": float(mean_key("ground_hit@0.1")),
                "tokens_used": float(mean_key("tokens_used")),
                "latency_ms": {"total": float(_mean(lat_vals))},
                "gt_boxes_mm": best.get("gt_boxes_mm", []) or [],
                "best_token_id": best.get("best_token_id", None),
                "best_token_box_mm": best.get("best_token_box_mm", None),
            }
        )
    return out


def _write_csv(path: Path, *, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _write_png_rgb(path: Path, *, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != int(width) * int(height) * 3:
        raise ValueError("rgb buffer has wrong size")
    raw = bytearray()
    stride = int(width) * 3
    for y in range(int(height)):
        raw.append(0)  # filter type 0
        off = y * stride
        raw.extend(rgb[off : off + stride])
    compressed = zlib.compress(bytes(raw), level=9)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", int(width), int(height), 8, 2, 0, 0, 0)
    payload = sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _draw_point(img: bytearray, *, width: int, height: int, x: int, y: int, color: Tuple[int, int, int], r: int = 3) -> None:
    for dy in range(-int(r), int(r) + 1):
        for dx in range(-int(r), int(r) + 1):
            xx = int(x) + int(dx)
            yy = int(y) + int(dy)
            if 0 <= xx < int(width) and 0 <= yy < int(height):
                idx = (yy * int(width) + xx) * 3
                img[idx : idx + 3] = bytes([int(color[0]), int(color[1]), int(color[2])])


def _draw_line(img: bytearray, *, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    # Simple Bresenham.
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < int(width) and 0 <= y0 < int(height):
            idx = (y0 * int(width) + x0) * 3
            img[idx : idx + 3] = bytes([int(color[0]), int(color[1]), int(color[2])])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


_FONT_5X7: Dict[str, List[int]] = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    ".": [0, 0, 0, 0, 0, 0b00100, 0b00100],
    "-": [0, 0, 0, 0b11111, 0, 0, 0],
    "_": [0, 0, 0, 0, 0, 0, 0b11111],
    "/": [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
    ":": [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "J": [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    "X": [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    "Y": [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
}


def _draw_text(
    img: bytearray,
    *,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: Tuple[int, int, int],
    scale: int = 1,
) -> None:
    # Very small bitmap font (5x7), uppercase-only.
    s = str(text or "").upper()
    cx = int(x)
    cy = int(y)
    scale = max(1, int(scale))
    for ch in s:
        glyph = _FONT_5X7.get(ch, _FONT_5X7.get(" ", [0, 0, 0, 0, 0, 0, 0]))
        for row_i, bits in enumerate(glyph):
            for col_i in range(5):
                if (int(bits) >> (4 - col_i)) & 1:
                    for dy in range(scale):
                        for dx in range(scale):
                            xx = cx + col_i * scale + dx
                            yy = cy + row_i * scale + dy
                            if 0 <= xx < int(width) and 0 <= yy < int(height):
                                idx = (yy * int(width) + xx) * 3
                                img[idx : idx + 3] = bytes([int(color[0]), int(color[1]), int(color[2])])
        cx += (6 * scale)  # 5px glyph + 1px gap


def _plot_pareto(
    path: Path,
    *,
    points: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    with_ci_legend: bool = False,
) -> None:
    width, height = 800, 600
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 60
    img = bytearray(b"\xFF" * (width * height * 3))

    xs = [float(p.get(x_key, 0.0)) for p in points]
    ys = [float(p.get(y_key, 0.0)) for p in points]
    if not xs or not ys:
        _write_png_rgb(path, width=width, height=height, rgb=bytes(img))
        return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    def sx(x: float) -> int:
        return int(pad_l + (float(x) - min_x) / (max_x - min_x) * float(width - pad_l - pad_r))

    def sy(y: float) -> int:
        return int(pad_t + (max_y - float(y)) / (max_y - min_y) * float(height - pad_t - pad_b))

    # Axes.
    _draw_line(img, width=width, height=height, x0=pad_l, y0=height - pad_b, x1=width - pad_r, y1=height - pad_b, color=(0, 0, 0))
    _draw_line(img, width=width, height=height, x0=pad_l, y0=pad_t, x1=pad_l, y1=height - pad_b, color=(0, 0, 0))

    colors = {
        "fixed": (31, 119, 180),
        "heuristic": (255, 127, 14),
        "learned": (44, 160, 44),
        "random": (148, 103, 189),
        "oracle": (214, 39, 40),
    }

    pareto_idx = set(pareto_front([(float(p.get(y_key, 0.0)), float(p.get(x_key, 0.0))) for p in points]))
    for i, p in enumerate(points):
        method = str(p.get("method", "")).strip()
        c = colors.get(method, (127, 127, 127))
        r = 5 if i in pareto_idx else 3
        _draw_point(img, width=width, height=height, x=sx(float(p.get(x_key, 0.0))), y=sy(float(p.get(y_key, 0.0))), color=c, r=r)

    if with_ci_legend:
        # Minimal legend (bitmap font) to indicate CI provenance for the paper-grade track.
        lx = pad_l + 10
        ly = pad_t + 10
        for j, method in enumerate(["fixed", "heuristic", "learned"]):
            c = colors.get(method, (127, 127, 127))
            _draw_point(img, width=width, height=height, x=lx, y=ly + j * 14, color=c, r=3)
            _draw_text(img, width=width, height=height, x=lx + 10, y=ly + j * 14 - 4, text=method, color=(0, 0, 0), scale=1)
        _draw_text(
            img,
            width=width,
            height=height,
            x=lx,
            y=ly + 3 * 14 + 4,
            text="CI: SEE TABLE1_MAIN_CI.CSV",
            color=(0, 0, 0),
            scale=1,
        )

    _write_png_rgb(path, width=width, height=height, rgb=bytes(img))


def _write_overlay_svg(path: Path, *, title: str, cited: List[List[float]], gt: List[List[float]]) -> None:
    boxes = [*cited, *gt]
    if not boxes:
        return

    min_x = min(float(b[0]) for b in boxes)
    max_x = max(float(b[1]) for b in boxes)
    min_y = min(float(b[2]) for b in boxes)
    max_y = max(float(b[3]) for b in boxes)
    dx = max(1e-6, max_x - min_x)
    dy = max(1e-6, max_y - min_y)

    w = 360
    h = 360
    pad = 20

    def sx(x: float) -> float:
        return float(pad) + (float(x) - float(min_x)) / float(dx) * float(w)

    def sy(y: float) -> float:
        return float(pad) + (float(max_y) - float(y)) / float(dy) * float(h)

    def rect(box: List[float], stroke: str) -> str:
        x0, x1, y0, y1, _z0, _z1 = [float(v) for v in box]
        x = sx(float(x0))
        y = sy(float(y1))
        rw = max(1.0, sx(float(x1)) - sx(float(x0)))
        rh = max(1.0, sy(float(y0)) - sy(float(y1)))
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{rw:.2f}" height="{rh:.2f}" fill="none" stroke="{stroke}" stroke-width="2" />'

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w + 2*pad}" height="{h + 2*pad + 24}">')
    parts.append(f'<text x="{pad}" y="{pad + h + 18}" font-family="monospace" font-size="12">{title}</text>')
    for b in gt:
        parts.append(rect(b, "#2ca02c"))
    for b in cited:
        parts.append(rect(b, "#d62728"))
    parts.append("</svg>\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def paper_export(metrics_jsonls: List[Path], *, out_dir: Path) -> Dict[str, Any]:
    raw_rows = _load_jsonl_many(list(metrics_jsonls))
    rows = _aggregate_by_case(raw_rows)
    groups = _group_means(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    table1_fields = ["method", "budget_B", "n", "ground_mean_iou_mean", "ground_hit@0.1_mean", "tokens_used_mean", "latency_ms.total_mean"]
    _write_csv(out_dir / "table1_main.csv", rows=groups, fieldnames=table1_fields)

    # Paper-grade CI table (bootstrap over cases within each method/budget group).
    by_mb_vals: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for r in rows:
        method = str(r.get("method", "")).strip()
        budget = int(r.get("budget_B", 0) or 0)
        if not method or budget <= 0:
            continue
        by_mb_vals.setdefault((method, budget), {"ground_mean_iou": [], "ground_hit@0.1": []})
        try:
            by_mb_vals[(method, budget)]["ground_mean_iou"].append(float(r.get("ground_mean_iou", 0.0)))
        except Exception:
            pass
        try:
            by_mb_vals[(method, budget)]["ground_hit@0.1"].append(float(r.get("ground_hit@0.1", 0.0)))
        except Exception:
            pass

    ci_rows: List[Dict[str, Any]] = []
    for g in groups:
        method = str(g.get("method", "")).strip()
        budget = int(g.get("budget_B", 0) or 0)
        vals = by_mb_vals.get((method, budget), {"ground_mean_iou": [], "ground_hit@0.1": []})
        iou_stats = _bootstrap_ci_mean(
            list(vals.get("ground_mean_iou", [])),
            n_boot=1000,
            alpha=0.05,
            seed=_stable_int_hash(f"{method}:{budget}:iou"),
        )
        hit_stats = _bootstrap_ci_mean(
            list(vals.get("ground_hit@0.1", [])),
            n_boot=1000,
            alpha=0.05,
            seed=_stable_int_hash(f"{method}:{budget}:hit01"),
        )
        ci_rows.append(
            {
                **g,
                "ground_mean_iou_ci_low": float(iou_stats["ci_low"]),
                "ground_mean_iou_ci_high": float(iou_stats["ci_high"]),
                "ground_hit@0.1_ci_low": float(hit_stats["ci_low"]),
                "ground_hit@0.1_ci_high": float(hit_stats["ci_high"]),
            }
        )

    table1_ci_fields = [
        "method",
        "budget_B",
        "n",
        "ground_mean_iou_mean",
        "ground_mean_iou_ci_low",
        "ground_mean_iou_ci_high",
        "ground_hit@0.1_mean",
        "ground_hit@0.1_ci_low",
        "ground_hit@0.1_ci_high",
        "tokens_used_mean",
        "latency_ms.total_mean",
    ]
    _write_csv(out_dir / "table1_main_ci.csv", rows=ci_rows, fieldnames=table1_ci_fields)

    # Table2: learned vs heuristic deltas per budget.
    by_mb: Dict[Tuple[str, int], Dict[str, Any]] = {(g["method"], int(g["budget_B"])): g for g in groups}
    budgets = sorted({int(g["budget_B"]) for g in groups})
    t2_rows: List[Dict[str, Any]] = []
    for b in budgets:
        h = by_mb.get(("heuristic", int(b)), None)
        l = by_mb.get(("learned", int(b)), None)
        if not h or not l:
            continue
        t2_rows.append(
            {
                "budget_B": int(b),
                "delta_iou_learned_vs_heuristic": float(l["ground_mean_iou_mean"]) - float(h["ground_mean_iou_mean"]),
                "delta_hit01_learned_vs_heuristic": float(l["ground_hit@0.1_mean"]) - float(h["ground_hit@0.1_mean"]),
                "delta_tokens_used_learned_vs_heuristic": float(l["tokens_used_mean"]) - float(h["tokens_used_mean"]),
                "delta_latency_ms_total_learned_vs_heuristic": float(l["latency_ms.total_mean"]) - float(h["latency_ms.total_mean"]),
            }
        )
    table2_fields = [
        "budget_B",
        "delta_iou_learned_vs_heuristic",
        "delta_hit01_learned_vs_heuristic",
        "delta_tokens_used_learned_vs_heuristic",
        "delta_latency_ms_total_learned_vs_heuristic",
    ]
    _write_csv(out_dir / "table2_ablation.csv", rows=t2_rows, fieldnames=table2_fields)

    # Fig2: Pareto scatter (tokens / latency).
    _plot_pareto(out_dir / "fig2_pareto_tokens.png", points=groups, x_key="tokens_used_mean", y_key="ground_mean_iou_mean")
    _plot_pareto(out_dir / "fig2_pareto_latency.png", points=groups, x_key="latency_ms.total_mean", y_key="ground_mean_iou_mean")
    _plot_pareto(out_dir / "fig2_pareto_tokens_ci.png", points=groups, x_key="tokens_used_mean", y_key="ground_mean_iou_mean", with_ci_legend=True)

    # Fig3 examples: best learned@B16 case overlay (GT vs best token box).
    example = None
    best = -1.0
    for r in rows:
        if str(r.get("method", "")).strip() != "learned":
            continue
        if int(r.get("budget_B", 0) or 0) != 16:
            continue
        score = float(r.get("ground_mean_iou", 0.0))
        if score > best and r.get("best_token_box_mm", None) is not None:
            best = float(score)
            example = r
    if example is None:
        for r in rows:
            if str(r.get("method", "")).strip() != "learned":
                continue
            score = float(r.get("ground_mean_iou", 0.0))
            if score > best and r.get("best_token_box_mm", None) is not None:
                best = float(score)
                example = r

    fig3_dir = out_dir / "fig3_examples"
    fig3_dir.mkdir(parents=True, exist_ok=True)
    if example is not None:
        case_id = str(example.get("case_id", "")).strip() or "case-0000"
        budget_B = int(example.get("budget_B", 0) or 0)
        title = f"learned B={budget_B} case={case_id} max_iou={best:.3f}"
        gt = example.get("gt_boxes_mm", []) or []
        token_box = example.get("best_token_box_mm", None)
        cited = [token_box] if isinstance(token_box, list) and len(token_box) == 6 else []
        gt_boxes = [b for b in gt if isinstance(b, list) and len(b) == 6]
        _write_overlay_svg(fig3_dir / f"learned_B{budget_B}_{case_id}.svg", title=title, cited=cited, gt=gt_boxes)

    summary: Dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "in_metrics_jsonls": [str(Path(p)) for p in metrics_jsonls],
        "n_inputs": int(len(metrics_jsonls)),
        "n_rows_raw": int(len(raw_rows)),
        "n_rows_agg": int(len(rows)),
        "out_dir": str(out_dir),
    }
    if len(metrics_jsonls) == 1:
        summary["in_metrics_jsonl"] = str(Path(metrics_jsonls[0]))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper export: tables/figures from benchmark metrics.jsonl (repo-skeleton).")
    parser.add_argument("--in", dest="metrics_jsonl", action="append", required=True, help="Path to metrics.jsonl (repeatable)")
    parser.add_argument("--out", required=True, help="Output directory (artifacts/paper_e0910)")
    args = parser.parse_args()

    in_paths: List[Path] = []
    for raw in args.metrics_jsonl or []:
        p = Path(str(raw))
        if not p.exists():
            print(f"[ERR] missing metrics.jsonl: {p}", file=sys.stderr)
            sys.exit(2)
        in_paths.append(p)
    try:
        summary = paper_export(in_paths, out_dir=Path(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
