from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .infer_refine import _write_sidecar_artifacts, run_infer_refine
from .unified_eval import unified_eval
from .validate_run import validate_run


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _load_manifest_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _case_id(row: Dict[str, Any]) -> str:
    return str(row.get("case_id", "")).strip()


def _path_exists(p: str) -> bool:
    s = str(p or "").strip()
    return bool(s) and Path(s).exists()


def _select_rows(
    rows: List[Dict[str, Any]],
    *,
    case_ids: List[str] | None,
    max_cases: int,
    require_volume_loader: str | None,
) -> List[Dict[str, Any]]:
    if case_ids:
        idx = {_case_id(r): r for r in rows if _case_id(r)}
        out = []
        for cid in case_ids:
            cid = str(cid).strip()
            if cid not in idx:
                raise KeyError(f"case_id not found in manifest: {cid}")
            out.append(idx[cid])
        return out

    want = (str(require_volume_loader).strip().lower() if require_volume_loader else "")
    out: List[Dict[str, Any]] = []
    for row in rows:
        cid = _case_id(row)
        if not cid:
            continue
        vol_path = str(row.get("volume_path", "")).strip()
        has_vol = _path_exists(vol_path)

        if want == "nifti" and not has_vol:
            continue
        if want == "dummy" and has_vol:
            continue

        out.append(row)
        if max_cases > 0 and len(out) >= int(max_cases):
            break
    return out


def batch_infer_eval(
    *,
    manifest: str,
    out_dir: str,
    budget_B: int,
    config: str,
    max_cases: int,
    case_ids: List[str] | None,
    require_volume_loader: str | None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config)
    base_cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}

    rows = _load_manifest_jsonl(manifest_path)
    selected = _select_rows(
        rows,
        case_ids=case_ids,
        max_cases=int(max_cases),
        require_volume_loader=require_volume_loader,
    )
    if not selected:
        raise ValueError("no manifest rows selected (check --case-ids/--max-cases/--require-volume-loader)")

    runs_root = out_path / "runs"
    eval_root = out_path / "eval"
    runs_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row in selected:
        cid = _case_id(row) or "case-0000"
        case_out = runs_root / cid
        case_out.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["_manifest_jsonl"] = str(manifest_path)
        cfg["_manifest_case_id"] = str(cid)

        try:
            run = run_infer_refine(cfg, budget_B=int(budget_B))
            run_path = case_out / "run.json"
            run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary = {
                "timestamp_utc": _utc_now_iso(),
                "python": sys.version,
                "platform": platform.platform(),
                "config_path": str(cfg_path),
                "manifest": str(manifest_path),
                "case_id": str(cid),
                "budget_B": int(budget_B),
            }
            (case_out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            # Ensure per-case "paper-facing" sidecars exist for batch runs too.
            _write_sidecar_artifacts(case_out, run)

            inp = {}
            meta = run.get("meta", {}) if isinstance(run, dict) else {}
            if isinstance(meta, dict) and isinstance(meta.get("input", {}), dict):
                inp = dict(meta.get("input", {}))

            if require_volume_loader is not None:
                want = str(require_volume_loader).strip().lower()
                got = str(inp.get("volume_loader", "")).strip().lower()
                if want and got != want:
                    raise RuntimeError(f"case {cid}: volume_loader '{got}' does not match required '{want}'")

            # Validate basic invariants + meta.input presence.
            v_errors = validate_run(
                run if isinstance(run, dict) else {},
                require_meta_input=True,
                require_meta_input_case_id=str(cid),
                require_meta_input_volume_loader=str(require_volume_loader) if require_volume_loader else None,
                require_meta_input_volume_path_exists=bool(require_volume_loader and want == "nifti"),
                require_meta_input_report_path_exists=True,
                run_json_path=run_path,
                require_final_report_txt=True,
                require_evidence_graph_json=True,
                require_trace_jsonl=True,
            )
            if v_errors:
                raise RuntimeError(f"case {cid}: validate_run failed: {v_errors}")

            eval_dir = eval_root / cid
            eval_dir.mkdir(parents=True, exist_ok=True)
            ev = unified_eval(str(run_path), str(eval_dir), case_id=str(cid), budget_B=0)
            metrics_path = Path(str(ev.get("metrics_path", eval_dir / "metrics.json")))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics, dict):
                metrics_rows.append(metrics)
            else:
                raise RuntimeError(f"case {cid}: metrics.json is not an object")

        except Exception as exc:  # noqa: BLE001
            failures.append({"case_id": str(cid), "error": str(exc)})
            continue

    # Always write aggregate outputs for audit.
    metrics_jsonl = out_path / "metrics.jsonl"
    metrics_jsonl.write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metrics_rows),
        encoding="utf-8",
    )
    agg = {
        "timestamp_utc": _utc_now_iso(),
        "manifest": str(manifest_path),
        "n_selected": int(len(selected)),
        "n_ok": int(len(metrics_rows)),
        "n_failed": int(len(failures)),
        "failures": failures,
        "metrics_jsonl": str(metrics_jsonl),
    }
    (out_path / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        raise RuntimeError(f"{len(failures)} case(s) failed: {failures[:3]}")

    return {"out_dir": str(out_path), "metrics_jsonl": str(metrics_jsonl), "n": int(len(metrics_rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run infer_refine + unified_eval over a JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--out", required=True, help="Output directory (writes runs/ eval/ metrics.jsonl)")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--config", default="voxtoken/configs/inference.yaml")
    parser.add_argument("--max-cases", type=int, default=0, help="Max cases to run (0 means all)")
    parser.add_argument("--case-ids", nargs="+", default=None, help="Explicit case_id list (overrides max-cases)")
    parser.add_argument(
        "--require-volume-loader",
        default=None,
        choices=["nifti", "dummy"],
        help="Filter selection and validate run.meta.input.volume_loader",
    )
    args = parser.parse_args()

    result = batch_infer_eval(
        manifest=str(args.manifest),
        out_dir=str(args.out),
        budget_B=int(args.budget),
        config=str(args.config),
        max_cases=int(args.max_cases),
        case_ids=list(args.case_ids) if args.case_ids else None,
        require_volume_loader=(str(args.require_volume_loader) if args.require_volume_loader else None),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
