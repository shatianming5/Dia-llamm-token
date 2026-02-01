from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..models.tokenizer import TokenPyramid, Tokenizer3D
from ..schemas import Token
from .infer_refine import _load_volume_for_manifest_row  # repo-skeleton reuse


Box = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1 (mm)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError("manifest rows must be JSON objects")
        rows.append(obj)
    return rows


def _parse_gt_boxes_sent0(row: Dict[str, Any]) -> List[Box]:
    raw = row.get("grounding_boxes_by_sent_mm", {}) or row.get("gt_boxes_by_sent_mm", {})
    if not isinstance(raw, dict):
        return []
    v = raw.get("0", None) or raw.get(0, None)
    if not isinstance(v, list):
        return []
    out: List[Box] = []
    for b in v:
        if not isinstance(b, (list, tuple)) or len(b) != 6:
            continue
        try:
            out.append(tuple(float(x) for x in b))  # type: ignore[assignment]
        except Exception:
            continue
    return out


def _box_iou_3d(a: Box, b: Box) -> float:
    ax0, ax1, ay0, ay1, az0, az1 = [float(x) for x in a]
    bx0, bx1, by0, by1, bz0, bz1 = [float(x) for x in b]

    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)
    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    iz0 = max(az0, bz0)
    iz1 = min(az1, bz1)

    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) * max(0.0, iz1 - iz0)
    if inter <= 0.0:
        return 0.0

    va = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) * max(0.0, az1 - az0)
    vb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0) * max(0.0, bz1 - bz0)
    union = va + vb - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _best_iou(token_boxes: List[Box], gt: List[Box]) -> float:
    best = 0.0
    for tb in token_boxes:
        for gb in gt:
            best = max(best, float(_box_iou_3d(tb, gb)))
    return float(best)


def _token_feature_vector(
    token: Token,
    *,
    volume: Any,
    voxel_spacing_mm: List[float],
) -> Dict[str, Any]:
    sx, sy, sz = [float(x) for x in (voxel_spacing_mm or [1.0, 1.0, 1.0])]

    def _shape_cdhw(vol: Any) -> Tuple[int, int, int, int]:
        try:
            c = len(vol)
            d = len(vol[0])
            h = len(vol[0][0])
            w = len(vol[0][0][0])
            return int(c), int(d), int(h), int(w)
        except Exception:
            return 1, 1, 1, 1

    def _box_mm_to_zyx_slices(box_mm: Box) -> Tuple[int, int, int, int, int, int]:
        x0, x1, y0, y1, z0, z1 = box_mm
        ix0 = int(round(float(x0) / sx))
        ix1 = int(round(float(x1) / sx))
        iy0 = int(round(float(y0) / sy))
        iy1 = int(round(float(y1) / sy))
        iz0 = int(round(float(z0) / sz))
        iz1 = int(round(float(z1) / sz))
        return iz0, iz1, iy0, iy1, ix0, ix1

    def _patch_stats() -> Tuple[float, float, float]:
        # Welford mean/variance + max over the token patch.
        c, d, h, w = _shape_cdhw(volume)
        z0, z1, y0, y1, x0, x1 = _box_mm_to_zyx_slices(token.omega_box_mm)
        z0 = max(0, min(int(z0), int(d)))
        z1 = max(0, min(int(z1), int(d)))
        y0 = max(0, min(int(y0), int(h)))
        y1 = max(0, min(int(y1), int(h)))
        x0 = max(0, min(int(x0), int(w)))
        x1 = max(0, min(int(x1), int(w)))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return 0.0, 0.0, 0.0

        n = 0
        mean = 0.0
        m2 = 0.0
        vmax = float("-inf")
        for cc in range(int(c)):
            for zz in range(int(z0), int(z1)):
                for yy in range(int(y0), int(y1)):
                    row = volume[cc][zz][yy]
                    for xx in range(int(x0), int(x1)):
                        v = float(row[xx])
                        n += 1
                        delta = v - mean
                        mean += delta / float(n)
                        delta2 = v - mean
                        m2 += delta * delta2
                        if v > vmax:
                            vmax = float(v)
        if n <= 0:
            return 0.0, 0.0, 0.0
        var = float(m2 / float(n))
        if vmax == float("-inf"):
            vmax = 0.0
        return float(mean), float(var), float(vmax)

    mean_int, var, vmax = _patch_stats()
    x0, x1, y0, y1, z0, z1 = [float(x) for x in token.omega_box_mm]
    return {
        "recon_error": float(var),
        "evidence_entropy": float(math.log1p(max(0.0, float(var)))),
        "citation_pressure": 0.0,
        # NOTE: repo-skeleton proxy; downstream can interpret as "splittability/level".
        "history_splits": int(len(token.children_ids)),
        "center_x_mm": float((x0 + x1) / 2.0),
        "center_y_mm": float((y0 + y1) / 2.0),
        "center_z_mm": float((z0 + z1) / 2.0),
        "mean_intensity": float(mean_int),
        "max_intensity": float(vmax),
        "level": int(token.level),
    }


def _refine_select(pyramid: TokenPyramid, tokens: List[Token], split_ids: List[int], budget_B: int) -> Tuple[List[Token], List[int]]:
    # Replace selected parent tokens with their children (1 level deeper), respecting the budget.
    out: List[Token] = list(tokens)
    active_ids = {int(t.token_id) for t in out}
    executed: List[int] = []

    for pid in split_ids:
        pid = int(pid)
        if pid not in active_ids:
            continue
        child_ids = list(pyramid.children_map.get(pid, []))
        if not child_ids:
            continue
        new_count = int(len(out) - 1 + len(child_ids))
        if new_count > int(budget_B):
            continue

        out = [t for t in out if int(t.token_id) != pid]
        active_ids.remove(pid)
        for cid in child_ids:
            if cid not in pyramid.token_by_id:
                continue
            ct = pyramid.token_by_id[cid]
            if int(ct.token_id) in active_ids:
                continue
            out.append(ct)
            active_ids.add(int(ct.token_id))
        executed.append(pid)

    out.sort(key=lambda t: int(t.token_id))
    return out, executed


def build_policy_dataset_oracle(
    *,
    manifest_jsonl: Path,
    out_jsonl: Path,
    cfg: Dict[str, Any],
    split: str,
    budgets: List[int],
    max_cases: int,
    seed: int,
) -> Dict[str, Any]:
    rows = _load_jsonl(manifest_jsonl)
    rows.sort(key=lambda r: str(r.get("case_id", "")).strip())

    split_norm = str(split or "").strip()
    if split_norm not in {"train", "val", "test"}:
        raise ValueError(f"invalid --split: {split_norm} (expected train|val|test)")

    budgets_sorted = sorted({int(b) for b in budgets if int(b) > 0})
    if not budgets_sorted:
        raise ValueError("no positive budgets provided")

    refine_cfg = cfg.get("refine", {})
    if not isinstance(refine_cfg, dict):
        refine_cfg = {}
    max_rounds = int(refine_cfg.get("max_rounds", 0))
    max_splits_per_round = int(refine_cfg.get("max_splits_per_round", 0)) or 0
    max_splits_total = int(max(0, max_rounds) * max(0, max_splits_per_round))

    tokenizer = Tokenizer3D(dict(cfg.get("tokenizer", {})))
    spacing = tokenizer.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    if not isinstance(spacing, list) or len(spacing) != 3:
        spacing = [1.0, 1.0, 1.0]
    spacing = [float(x) for x in spacing]

    out_lines: List[str] = []
    loader_counts: Counter[str] = Counter()
    counts_by_budget: Counter[str] = Counter()
    n_cases = 0
    n_skipped_split = 0
    n_skipped_missing_gt = 0
    n_steps = 0
    n_pos = 0
    n_neg = 0

    for row in rows:
        if max_cases > 0 and n_cases >= int(max_cases):
            break
        if str(row.get("split", "")).strip() != split_norm:
            n_skipped_split += 1
            continue

        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue

        gt_boxes = _parse_gt_boxes_sent0(row)
        if not gt_boxes:
            n_skipped_missing_gt += 1
            continue

        volume, input_meta = _load_volume_for_manifest_row(row, cfg)
        loader_counts[str((input_meta or {}).get("volume_loader", "unknown"))] += 1

        pyramid: TokenPyramid = tokenizer.build_pyramid(volume)
        base_tokens = tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=8)

        for budget_B in budgets_sorted:
            tk: List[Token] = list(base_tokens)
            step_idx = 0

            for _ in range(max(0, int(max_splits_total))):
                token_boxes = [t.omega_box_mm for t in tk]
                base_iou = _best_iou(token_boxes, gt_boxes)

                token_rewards: Dict[int, float] = {}
                best_pid: int | None = None
                best_impr = 0.0

                for t in tk:
                    pid = int(t.token_id)
                    child_ids = list(pyramid.children_map.get(pid, []))
                    new_count = int(len(tk) - 1 + len(child_ids))
                    if not child_ids or new_count > int(budget_B):
                        impr = 0.0
                    else:
                        tk_candidate = [x for x in tk if int(x.token_id) != pid] + [
                            pyramid.token_by_id[cid] for cid in child_ids if cid in pyramid.token_by_id
                        ]
                        cand_iou = _best_iou([x.omega_box_mm for x in tk_candidate], gt_boxes)
                        impr = float(cand_iou) - float(base_iou)
                    token_rewards[pid] = float(impr)

                    if float(impr) > float(best_impr) + 1e-12:
                        best_impr = float(impr)
                        best_pid = int(pid)
                    elif abs(float(impr) - float(best_impr)) <= 1e-12 and best_pid is not None and int(pid) < int(best_pid):
                        best_pid = int(pid)

                if best_pid is None or float(best_impr) <= 0.0:
                    break

                # Emit one-vs-rest labels for this step (single positive best_pid).
                for t in tk:
                    pid = int(t.token_id)
                    label = int(pid == int(best_pid))
                    if label:
                        n_pos += 1
                    else:
                        n_neg += 1
                    feats = _token_feature_vector(t, volume=volume, voxel_spacing_mm=spacing)
                    out_lines.append(
                        json.dumps(
                            {
                                "case_id": str(case_id),
                                "budget_B": int(budget_B),
                                "step_idx": int(step_idx),
                                "token_id": int(pid),
                                **feats,
                                "reward": float(token_rewards.get(pid, 0.0)),
                                "label": int(label),
                            },
                            ensure_ascii=False,
                        )
                    )

                tk2, executed = _refine_select(pyramid, tk, [int(best_pid)], int(budget_B))
                if not executed or len(tk2) == len(tk):
                    break
                tk = tk2
                step_idx += 1
                n_steps += 1
                counts_by_budget[str(int(budget_B))] += 1

        n_cases += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("".join(s + "\n" for s in out_lines), encoding="utf-8")

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "manifest": str(manifest_jsonl),
        "out_jsonl": str(out_jsonl),
        "split": str(split_norm),
        "budgets": [int(b) for b in budgets_sorted],
        "max_cases": int(max_cases),
        "seed": int(seed),
        "n_cases": int(n_cases),
        "n_rows": int(len(out_lines)),
        "n_steps": int(n_steps),
        "label_counts": {"pos": int(n_pos), "neg": int(n_neg)},
        "steps_by_budget": {k: int(v) for k, v in counts_by_budget.items()},
        "n_skipped_split": int(n_skipped_split),
        "n_skipped_missing_gt": int(n_skipped_missing_gt),
        "volume_loader_counts": dict(loader_counts),
    }
    (out_jsonl.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an oracle-trajectory policy dataset from a GT-box manifest (repo-skeleton).")
    parser.add_argument("--manifest", required=True, help="Input GT manifest.jsonl (must include volume_path + gt boxes)")
    parser.add_argument("--out", required=True, help="Output dataset.jsonl path")
    parser.add_argument("--config", required=True, help="YAML config (reuses tokenizer/refine settings)")
    parser.add_argument("--split", default="train", help="Split to use (train|val|test)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32], help="Budgets to generate trajectories for")
    parser.add_argument("--max-cases", type=int, default=0, help="Max cases to process (0 = no limit)")
    parser.add_argument("--seed", type=int, default=0, help="Seed (logged for reproducibility; does not randomize selection)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] missing config: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = _load_yaml(cfg_path)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[ERR] missing manifest: {manifest}", file=sys.stderr)
        sys.exit(2)

    out_jsonl = Path(args.out)
    try:
        summary = build_policy_dataset_oracle(
            manifest_jsonl=manifest,
            out_jsonl=out_jsonl,
            cfg=cfg,
            split=str(args.split),
            budgets=[int(b) for b in list(args.budgets or [])],
            max_cases=int(args.max_cases),
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

