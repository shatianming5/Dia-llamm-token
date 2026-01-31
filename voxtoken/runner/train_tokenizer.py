from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _make_synth_volume(shape_cdhw: list[int], *, seed: int, pattern: str) -> list[list[list[list[float]]]]:
    c, d, h, w = [int(x) for x in shape_cdhw]
    pattern = str(pattern or "zeros").strip().lower()

    if pattern == "noise":
        rng = random.Random(int(seed))
        return [[[[float(rng.random()) for _ in range(w)] for _ in range(h)] for _ in range(d)] for _ in range(c)]

    if pattern == "gradient":
        out: list[list[list[list[float]]]] = []
        for _cc in range(c):
            ch: list[list[list[float]]] = []
            for zz in range(d):
                slab: list[list[float]] = []
                for yy in range(h):
                    row: list[float] = []
                    for xx in range(w):
                        v = 0.0
                        if w > 1:
                            v += float(xx) / float(w - 1)
                        if h > 1:
                            v += float(yy) / float(h - 1)
                        if d > 1:
                            v += float(zz) / float(d - 1)
                        row.append(float(v))
                    slab.append(row)
                ch.append(slab)
            out.append(ch)
        return out

    return [[[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(d)] for _ in range(c)]


def _iter_patch_means(volume: list[list[list[list[float]]]], *, patch: int) -> list[float]:
    c = len(volume)
    d = len(volume[0])
    h = len(volume[0][0])
    w = len(volume[0][0][0])
    patch = max(1, int(patch))

    means: list[float] = []
    for z0 in range(0, d, patch):
        z1 = min(d, z0 + patch)
        for y0 in range(0, h, patch):
            y1 = min(h, y0 + patch)
            for x0 in range(0, w, patch):
                x1 = min(w, x0 + patch)
                n = 0
                mean = 0.0
                for cc in range(c):
                    for zz in range(z0, z1):
                        for yy in range(y0, y1):
                            row = volume[cc][zz][yy]
                            for xx in range(x0, x1):
                                v = float(row[xx])
                                n += 1
                                mean += (v - mean) / float(n)
                if n > 0:
                    means.append(float(mean))
    return means


def _kmeans_1d(values: list[float], *, k: int, iters: int, seed: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(int(k))]
    k = max(1, int(k))
    iters = max(1, int(iters))

    v_sorted = sorted(float(v) for v in values)

    # Quantile init.
    centers: list[float] = []
    for i in range(k):
        q = (i + 0.5) / float(k)
        idx = int(q * float(len(v_sorted) - 1))
        centers.append(float(v_sorted[max(0, min(len(v_sorted) - 1, idx))]))

    rng = random.Random(int(seed))
    for _ in range(iters):
        buckets: list[list[float]] = [[] for _ in range(k)]
        for v in values:
            best = 0
            best_d = abs(float(v) - float(centers[0]))
            for j in range(1, k):
                d = abs(float(v) - float(centers[j]))
                if d < best_d:
                    best = j
                    best_d = d
            buckets[best].append(float(v))
        new_centers: list[float] = []
        for j in range(k):
            if buckets[j]:
                new_centers.append(sum(buckets[j]) / float(len(buckets[j])))
            else:
                new_centers.append(float(rng.choice(v_sorted)))
        centers = new_centers

    return [float(x) for x in centers]


def train_tokenizer(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage T: tokenizer training (skeleton: stdlib-only synthetic codebook fit)."""
    cfg = dict(cfg or {})
    train_cfg = cfg.get("train", {})
    if "out_dir" not in cfg and isinstance(train_cfg, dict) and train_cfg.get("save_dir"):
        cfg["out_dir"] = train_cfg.get("save_dir")

    out_root = Path(str(cfg.get("out_dir", "outputs/train_tokenizer")))
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = str(cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    synth = dict(cfg.get("synth", {}))
    shape = [int(x) for x in synth.get("shape_cdhw", [1, 8, 8, 8])]
    n_samples = int(synth.get("n_samples", 16))
    seed = int(synth.get("seed", 0))
    pattern = str(synth.get("pattern", "gradient"))

    tk_cfg = dict(cfg.get("tokenizer", {}))
    grid = dict(tk_cfg.get("grid", {}))
    patch = int(grid.get("patch", 4))

    codebook = dict(cfg.get("codebook", {}))
    k = int(codebook.get("k", 4))
    iters = int(codebook.get("iters", 8))

    means: list[float] = []
    for i in range(max(1, n_samples)):
        vol = _make_synth_volume(shape, seed=seed + i, pattern=pattern)
        means.extend(_iter_patch_means(vol, patch=patch))

    centers = _kmeans_1d(means, k=k, iters=iters, seed=seed)

    checkpoint = {
        "stage": "T",
        "run_id": run_id,
        "codebook": {"k": int(k), "centers": [float(x) for x in centers]},
        "tokenizer": {"grid": {"patch": int(patch)}},
    }

    # Simple quantization / usage diagnostics on training samples.
    err = 0.0
    counts = [0 for _ in range(int(len(centers)))]
    if means and centers:
        for v in means:
            best = 0
            best_d = abs(float(v) - float(centers[0]))
            for j in range(1, len(centers)):
                d = abs(float(v) - float(centers[j]))
                if d < best_d:
                    best = int(j)
                    best_d = float(d)
            counts[int(best)] += 1
            err += float(best_d)
        err /= float(len(means))

    used_codes = int(sum(1 for c in counts if int(c) > 0))
    perplexity = 0.0
    if means and centers and sum(counts) > 0:
        total = float(sum(counts))
        entropy = 0.0
        for c in counts:
            if int(c) <= 0:
                continue
            p = float(c) / total
            entropy -= p * math.log(p)
        perplexity = float(math.exp(entropy))

    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "step": 0,
        "quant_err": float(err),
        "n_values": int(len(means)),
        "codebook_size": int(len(centers)),
        "codebook_used": int(used_codes),
        "perplexity": float(perplexity),
        "counts": [int(x) for x in counts],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"status": "ok", "out_dir": str(out_root), "run_dir": str(run_dir), "checkpoint_path": str(run_dir / "checkpoint.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage T: train tokenizer (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    result = train_tokenizer(cfg)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
