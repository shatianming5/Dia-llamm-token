from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for small dense systems (stdlib-only)."""
    n = len(b)
    # Augment A with b.
    m = [list(map(float, row)) + [float(bi)] for row, bi in zip(a, b)]

    for col in range(n):
        # Find pivot.
        pivot = col
        for r in range(col, n):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        # Normalize pivot row.
        denom = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= denom

        # Eliminate.
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [float(m[i][n]) for i in range(n)]


def _fit_policy_weights_synth(cfg: Dict[str, Any]) -> Dict[str, float]:
    synth = dict(cfg.get("synth", {}))
    n = int(synth.get("n_samples", 256))
    seed = int(synth.get("seed", 0))
    noise_std = float(synth.get("noise_std", 0.0))
    ridge = float(dict(cfg.get("train", {})).get("ridge", 1e-6))

    true_w = dict(synth.get("true_weights", {}))
    w_true = [
        float(true_w.get("recon_error", 1.0)),
        float(true_w.get("evidence_entropy", 1.0)),
        float(true_w.get("citation_pressure", 1.0)),
        float(true_w.get("history_splits", -1.0)),
    ]

    rng = random.Random(int(seed))

    def gen_x() -> list[float]:
        # Keep a bounded range; history is integer-like to mimic "already split" counts.
        return [rng.random(), rng.random(), rng.random(), float(rng.randint(0, 3))]

    xs: list[list[float]] = []
    ys: list[float] = []
    for _ in range(int(n)):
        x = gen_x()
        y = w_true[0] * x[0] + w_true[1] * x[1] + w_true[2] * x[2] + w_true[3] * x[3]
        if noise_std > 0.0:
            y += rng.gauss(0.0, float(noise_std))
        xs.append(x)
        ys.append(float(y))

    # Normal equations: (X^T X + ridge*I) w = X^T y
    dim = 4
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for x, y in zip(xs, ys):
        for i in range(dim):
            xty[i] += float(x[i]) * float(y)
            for j in range(dim):
                xtx[i][j] += float(x[i]) * float(x[j])
    for i in range(dim):
        xtx[i][i] += float(ridge)

    w = _solve_linear_system(xtx, xty)
    return {
        "recon_error": float(w[0]),
        "evidence_entropy": float(w[1]),
        "citation_pressure": float(w[2]),
        "history_splits": float(w[3]),
    }

def _dataset_path(cfg: Dict[str, Any]) -> Path | None:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    raw = cfg.get("dataset_jsonl") or train_cfg.get("dataset_jsonl") or data_cfg.get("dataset_jsonl")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    return p


def _fit_policy_weights_dataset(cfg: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
    p = _dataset_path(cfg)
    if p is None:
        raise ValueError("dataset_jsonl not configured")
    if not p.exists():
        raise FileNotFoundError(f"dataset_jsonl not found: {p}")

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    ridge = float(train_cfg.get("ridge", 1e-6))

    xs: list[list[float]] = []
    ys: list[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        x = [
            float(obj.get("recon_error", 0.0)),
            float(obj.get("evidence_entropy", 0.0)),
            float(obj.get("citation_pressure", 0.0)),
            float(obj.get("history_splits", 0.0)),
        ]
        y = float(obj.get("reward", 0.0))
        xs.append(x)
        ys.append(y)

    if len(xs) < 4:
        raise ValueError(f"dataset_jsonl too small for 4D fit: n={len(xs)}")

    # Normal equations: (X^T X + ridge*I) w = X^T y
    dim = 4
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for x, y in zip(xs, ys):
        for i in range(dim):
            xty[i] += float(x[i]) * float(y)
            for j in range(dim):
                xtx[i][j] += float(x[i]) * float(x[j])
    for i in range(dim):
        xtx[i][i] += float(ridge)

    w = _solve_linear_system(xtx, xty)
    weights = {
        "recon_error": float(w[0]),
        "evidence_entropy": float(w[1]),
        "citation_pressure": float(w[2]),
        "history_splits": float(w[3]),
    }

    def pred(x: list[float]) -> float:
        return float(w[0]) * float(x[0]) + float(w[1]) * float(x[1]) + float(w[2]) * float(x[2]) + float(w[3]) * float(x[3])

    mse = 0.0
    for x, y in zip(xs, ys):
        mse += (float(y) - float(pred(x))) ** 2
    mse = float(mse) / float(len(xs)) if xs else 0.0

    meta = {"dataset_jsonl": str(p), "n_samples": int(len(xs)), "mse": float(mse)}
    return weights, meta


def train_policy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage P: offline contextual bandit / policy training (skeleton: stdlib-only linear fit)."""
    cfg = dict(cfg or {})
    train_cfg = cfg.get("train", {})
    if "out_dir" not in cfg and isinstance(train_cfg, dict) and train_cfg.get("save_dir"):
        cfg["out_dir"] = train_cfg.get("save_dir")

    out_root = Path(str(cfg.get("out_dir", "outputs/train_policy")))
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = str(cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = _dataset_path(cfg)
    fit_meta: Dict[str, Any] = {"fit": "synth"}
    if dataset is not None:
        weights, meta = _fit_policy_weights_dataset(cfg)
        fit_meta = {"fit": "dataset", **meta}
    else:
        weights = _fit_policy_weights_synth(cfg)

    checkpoint = {"stage": "P", "run_id": run_id, "weights": weights, "fit_meta": dict(fit_meta)}
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    # Minimal metrics: report fit quality on the synthetic dataset definition (loss is proxy only).
    synth = dict(cfg.get("synth", {}))
    n = int(synth.get("n_samples", 256))
    noise_std = float(synth.get("noise_std", 0.0))
    if fit_meta.get("fit") == "dataset":
        n = int(fit_meta.get("n_samples", n))
        noise_std = 0.0
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"loss": float(max(0.0, noise_std) ** 2), "n_samples": int(n), "step": 0, **fit_meta}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"status": "ok", "out_dir": str(out_root), "run_dir": str(run_dir), "checkpoint_path": str(run_dir / "checkpoint.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage P: train policy (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    result = train_policy(cfg)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
