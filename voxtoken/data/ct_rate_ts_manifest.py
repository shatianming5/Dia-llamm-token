from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _strip_nii_suffix(name: str) -> str:
    s = str(name or "").strip()
    if s.endswith(".nii.gz"):
        return s[: -len(".nii.gz")]
    if s.endswith(".nii"):
        return s[: -len(".nii")]
    return s


def _safe_int_list(x: Any, *, n: int) -> List[int] | None:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        return None
    out: List[int] = []
    for v in x:
        try:
            out.append(int(v))
        except Exception:
            return None
    return out


def _safe_float_list(x: Any, *, n: int) -> List[float] | None:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        return None
    out: List[float] = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def _target_shape_cdhw(cfg: Dict[str, Any]) -> List[int]:
    vol_cfg = cfg.get("volume", {})
    if not isinstance(vol_cfg, dict):
        vol_cfg = {}
    target = vol_cfg.get("target_shape_cdhw") or vol_cfg.get("shape_cdhw") or [1, 8, 8, 8]
    try:
        c, d, h, w = [int(x) for x in target]
    except Exception:
        return [1, 8, 8, 8]
    return [max(1, c), max(1, d), max(1, h), max(1, w)]


def _token_voxel_spacing_xyz_mm(cfg: Dict[str, Any]) -> List[float]:
    tok_cfg = cfg.get("tokenizer", {})
    if not isinstance(tok_cfg, dict):
        tok_cfg = {}
    spacing = tok_cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
        try:
            return [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        except Exception:
            return [1.0, 1.0, 1.0]
    return [1.0, 1.0, 1.0]


def _nifti_shape_xyz(path: Path) -> List[int]:
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nibabel is required to read NIfTI shapes") from exc
    img = nib.load(str(path))
    shape = list(getattr(img, "shape", ())[:3])
    if len(shape) != 3:
        raise ValueError(f"unsupported NIfTI shape: {getattr(img, 'shape', None)}")
    return [int(shape[0]), int(shape[1]), int(shape[2])]


def _downsample_steps_xyz(volume_shape_xyz: List[int], target_shape_cdhw: List[int]) -> List[int]:
    sx, sy, sz = [int(x) for x in volume_shape_xyz]
    _c, d_t, h_t, w_t = [int(x) for x in target_shape_cdhw]
    step_x = max(1, int(sx) // max(1, int(w_t)))
    step_y = max(1, int(sy) // max(1, int(h_t)))
    step_z = max(1, int(sz) // max(1, int(d_t)))
    return [int(step_x), int(step_y), int(step_z)]


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b) if int(b) != 0 else 0


def _bbox_xyz_vox_to_box_mm(
    bbox_xyz: List[int],
    *,
    volume_shape_xyz: List[int],
    target_shape_cdhw: List[int],
    token_spacing_xyz_mm: List[float],
) -> Box | None:
    # bbox_xyz is half-open [x0,y0,z0,x1,y1,z1) in the *original* voxel index space.
    if len(bbox_xyz) != 6:
        return None
    x0, y0, z0, x1, y1, z1 = [int(v) for v in bbox_xyz]

    # Clamp to original bounds first to avoid negative/overflow indices.
    sx, sy, sz = [int(x) for x in volume_shape_xyz]
    x0 = max(0, min(int(x0), int(sx)))
    x1 = max(0, min(int(x1), int(sx)))
    y0 = max(0, min(int(y0), int(sy)))
    y1 = max(0, min(int(y1), int(sy)))
    z0 = max(0, min(int(z0), int(sz)))
    z1 = max(0, min(int(z1), int(sz)))
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        return None

    step_x, step_y, step_z = _downsample_steps_xyz(volume_shape_xyz, target_shape_cdhw)
    _c, d_t, h_t, w_t = [int(x) for x in target_shape_cdhw]

    # Map original voxel bounds into the downsampled index space used by `_load_nifti_volume_small`.
    # Use floor/ceil so tiny objects remain representable after coarse downsampling.
    x0_ds = int(x0) // int(step_x)
    x1_ds = _ceil_div(int(x1), int(step_x))
    y0_ds = int(y0) // int(step_y)
    y1_ds = _ceil_div(int(y1), int(step_y))
    z0_ds = int(z0) // int(step_z)
    z1_ds = _ceil_div(int(z1), int(step_z))

    # Clamp to downsampled bounds (token boxes use [0, W/H/D] boundaries).
    x0_ds = max(0, min(int(x0_ds), int(w_t)))
    x1_ds = max(0, min(int(x1_ds), int(w_t)))
    y0_ds = max(0, min(int(y0_ds), int(h_t)))
    y1_ds = max(0, min(int(y1_ds), int(h_t)))
    z0_ds = max(0, min(int(z0_ds), int(d_t)))
    z1_ds = max(0, min(int(z1_ds), int(d_t)))

    if x1_ds <= x0_ds:
        if x0_ds >= int(w_t):
            x0_ds = max(0, int(w_t) - 1)
            x1_ds = int(w_t)
        else:
            x1_ds = int(x0_ds) + 1
    if y1_ds <= y0_ds:
        if y0_ds >= int(h_t):
            y0_ds = max(0, int(h_t) - 1)
            y1_ds = int(h_t)
        else:
            y1_ds = int(y0_ds) + 1
    if z1_ds <= z0_ds:
        if z0_ds >= int(d_t):
            z0_ds = max(0, int(d_t) - 1)
            z1_ds = int(d_t)
        else:
            z1_ds = int(z0_ds) + 1

    sx_mm, sy_mm, sz_mm = [float(x) for x in token_spacing_xyz_mm]
    x0_mm = float(x0_ds) * float(sx_mm)
    x1_mm = float(x1_ds) * float(sx_mm)
    y0_mm = float(y0_ds) * float(sy_mm)
    y1_mm = float(y1_ds) * float(sy_mm)
    z0_mm = float(z0_ds) * float(sz_mm)
    z1_mm = float(z1_ds) * float(sz_mm)
    if not (x1_mm > x0_mm and y1_mm > y0_mm and z1_mm > z0_mm):
        return None
    return (float(x0_mm), float(x1_mm), float(y0_mm), float(y1_mm), float(z0_mm), float(z1_mm))


def _parse_bracketed_numbers(s: str) -> List[float] | None:
    raw = str(s or "").strip()
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    raw = raw.replace(",", " ")
    parts = [p for p in raw.split() if p.strip()]
    if not parts:
        return None
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            return None
    return out


def _load_ts_nodule_bboxes_xyz(csv_path: Path) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            mask_file = str(row.get("mask_file", "")).strip()
            if not mask_file:
                continue
            bbox_raw = _parse_bracketed_numbers(str(row.get("bbox", "")).strip())
            if not bbox_raw or len(bbox_raw) != 6:
                continue
            bbox_int = [int(round(float(x))) for x in bbox_raw]
            out.setdefault(mask_file, []).append(bbox_int)
    return out


def _load_effusion_candidates(csv_path: Path) -> Dict[tuple[str, str], List[List[int]]]:
    """
    Load TotalSegmentator pleural/pericard effusion candidates CSV.

    Expected columns: mask_file, finding_type, bbox
    bbox format: [z0, z1, y0, y1, x0, x1] in voxel indices (half-open).
    """
    out: Dict[tuple[str, str], List[List[int]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            mask_file = str(row.get("mask_file", "")).strip()
            finding_type = str(row.get("finding_type", "")).strip()
            if not mask_file or not finding_type:
                continue
            bbox_raw = _parse_bracketed_numbers(str(row.get("bbox", "")).strip())
            if not bbox_raw or len(bbox_raw) != 6:
                continue
            bbox_int = [int(round(float(x))) for x in bbox_raw]
            out.setdefault((mask_file, finding_type), []).append(bbox_int)
    return out


def _index_masks(root: Path) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for p in sorted(root.rglob("*.nii.gz")):
        name = p.name
        if name not in idx:
            idx[name] = str(p)
    return idx


def _extract_task_structures(cfg: Dict[str, Any], *, task: str) -> List[str]:
    ts = cfg.get("totalseg", {})
    if not isinstance(ts, dict):
        ts = {}
    defs = ts.get("task_defs", {})
    if not isinstance(defs, dict):
        defs = {}
    task_cfg = defs.get(str(task), {})
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    structures = task_cfg.get("structures", [])
    if not isinstance(structures, list) or not structures:
        return []
    out = [str(x).strip() for x in structures if str(x).strip()]
    return out


def _candidates_dir(cfg: Dict[str, Any]) -> Path:
    ts = cfg.get("totalseg", {})
    if not isinstance(ts, dict):
        ts = {}
    raw = ts.get("candidates_total_dir") or ts.get("candidates_dir") or ""
    p = Path(str(raw)).expanduser()
    return p


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError("candidates.json must be a JSON list")
    out: List[Dict[str, Any]] = []
    for x in obj:
        if isinstance(x, dict):
            out.append(x)
    return out


def build_ct_rate_ts_manifest(
    *,
    in_manifest_jsonl: Path,
    in_mode: str,
    out_dir: Path,
    cfg: Dict[str, Any],
    task: str,
    max_cases: int,
) -> Dict[str, Any]:
    task = str(task).strip()
    if not task:
        raise ValueError("--task must be non-empty")
    in_mode = str(in_mode or "labeled_manifest").strip()
    if in_mode not in {"labeled_manifest", "ts_index"}:
        raise ValueError(f"unsupported --in-mode: {in_mode}")

    target_shape = _target_shape_cdhw(cfg)
    token_spacing_xyz_mm = _token_voxel_spacing_xyz_mm(cfg)

    candidates_dir = _candidates_dir(cfg)
    candidates_available = candidates_dir.exists()
    structures = _extract_task_structures(cfg, task=str(task))
    structure_set = {str(x) for x in structures} if structures else set()

    eff_cfg = cfg.get("effusions", {})
    if not isinstance(eff_cfg, dict):
        eff_cfg = {}
    eff_csv_raw = str(eff_cfg.get("candidates_csv") or "").strip()
    eff_root_raw = str(eff_cfg.get("masks_root") or "").strip()
    eff_csv = Path(eff_csv_raw).expanduser() if eff_csv_raw else None
    eff_root = Path(eff_root_raw).expanduser() if eff_root_raw else None
    use_eff = bool(eff_csv is not None and eff_root is not None and eff_csv.is_file() and eff_root.exists())
    eff_bboxes_by_mask_task: Dict[tuple[str, str], List[List[int]]] = {}
    eff_tasks_available: set[str] = set()
    if use_eff:
        eff_bboxes_by_mask_task = _load_effusion_candidates(eff_csv)  # type: ignore[arg-type]
        eff_tasks_available = {str(k[1]) for k in eff_bboxes_by_mask_task.keys()}

    ts_cfg = cfg.get("ts_seg", {})
    if not isinstance(ts_cfg, dict):
        ts_cfg = {}
    ts_task = ts_cfg.get(str(task), {})
    if not isinstance(ts_task, dict):
        ts_task = {}

    ts_csv_raw = str(ts_task.get("metadata_csv") or "").strip()
    ts_masks_root_raw = str(ts_task.get("masks_root") or "").strip()
    ts_csv = Path(ts_csv_raw).expanduser() if ts_csv_raw else None
    ts_masks_root = Path(ts_masks_root_raw).expanduser() if ts_masks_root_raw else None
    use_ts = bool(ts_csv is not None and ts_masks_root is not None and ts_csv.is_file() and ts_masks_root.exists())

    ts_bboxes_by_mask: Dict[str, List[List[int]]] = {}
    ts_mask_index: Dict[str, str] = {}
    if use_ts:
        ts_bboxes_by_mask = _load_ts_nodule_bboxes_xyz(ts_csv)  # type: ignore[arg-type]
        ts_mask_index = _index_masks(ts_masks_root)  # type: ignore[arg-type]

    if not (use_eff or use_ts or structure_set):
        raise ValueError(f"missing GT config for task '{task}': need effusions, ts_seg.{task}, or totalseg.task_defs.{task}.structures")
    if structure_set and not candidates_available:
        raise FileNotFoundError(f"totalseg candidates_total_dir not found: {candidates_dir}")

    in_rows = _load_jsonl(in_manifest_jsonl)
    # Deterministic ordering.
    in_rows.sort(key=lambda r: str(r.get("case_id", "")).strip())

    out_rows: List[Dict[str, Any]] = []
    skipped_missing_effusions_meta = 0
    skipped_missing_effusions_mask = 0
    skipped_missing_ts_meta = 0
    skipped_missing_ts_mask = 0
    skipped_missing_candidates = 0
    skipped_missing_boxes = 0
    skipped_missing_label = 0

    for row in in_rows:
        if max_cases > 0 and len(out_rows) >= int(max_cases):
            break

        if task == "lung_nodules" and in_mode == "labeled_manifest":
            labels_pos = row.get("labels_pos", [])
            if not (isinstance(labels_pos, list) and any(str(x).strip() == "Lung nodule" for x in labels_pos)):
                skipped_missing_label += 1
                continue

        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue

        volume_path = str(row.get("volume_path", "")).strip()
        if not volume_path or not Path(volume_path).exists():
            continue

        volume_name = str(row.get("volume_name", "")).strip() or Path(volume_path).name
        volume_shape_xyz = _nifti_shape_xyz(Path(volume_path))
        downsample_steps_xyz = _downsample_steps_xyz(volume_shape_xyz, target_shape)

        gt_boxes_mm: List[List[float]] = []
        gt_mask_path = ""
        gt_source = ""
        totalseg_cand_path = ""
        totalseg_source = ""
        gt_label_ids: List[int] = []
        gt_structure_names: List[str] = []

        # Preferred explicit-mask source for effusion tasks.
        if use_eff and task in eff_tasks_available:
            bboxes_zyx = eff_bboxes_by_mask_task.get((volume_name, task), [])
            if not bboxes_zyx:
                skipped_missing_effusions_meta += 1
            else:
                stem = _strip_nii_suffix(volume_name) or _strip_nii_suffix(case_id)
                mp = eff_root / stem / f"{task}.nii.gz"
                if not mp.exists():
                    skipped_missing_effusions_mask += 1
                else:
                    for bbox_zyx in bboxes_zyx:
                        if len(bbox_zyx) != 6:
                            continue
                        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox_zyx]
                        box = _bbox_xyz_vox_to_box_mm(
                            [int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)],
                            volume_shape_xyz=volume_shape_xyz,
                            target_shape_cdhw=target_shape,
                            token_spacing_xyz_mm=token_spacing_xyz_mm,
                        )
                        if box is None:
                            continue
                        gt_boxes_mm.append([float(x) for x in box])
                    if gt_boxes_mm:
                        gt_mask_path = str(mp)
                        gt_source = "totalseg.pleural_pericard_effusion"
                        gt_structure_names = [str(task)]

        # Preferred: TotalSegmentator lung_nodules explicit masks + bbox metadata CSV.
        if (not gt_boxes_mm or not gt_mask_path) and use_ts:
            bboxes_xyz = ts_bboxes_by_mask.get(volume_name, [])
            if not bboxes_xyz:
                skipped_missing_ts_meta += 1
            else:
                mp = ts_mask_index.get(volume_name, "")
                if not mp:
                    skipped_missing_ts_mask += 1
                elif not Path(mp).exists():
                    skipped_missing_ts_mask += 1
                else:
                    for bbox_xyz in bboxes_xyz:
                        box = _bbox_xyz_vox_to_box_mm(
                            bbox_xyz,
                            volume_shape_xyz=volume_shape_xyz,
                            target_shape_cdhw=target_shape,
                            token_spacing_xyz_mm=token_spacing_xyz_mm,
                        )
                        if box is None:
                            continue
                        gt_boxes_mm.append([float(x) for x in box])
                    gt_mask_path = str(mp)
                    gt_source = "ts_seg.lung_nodules"
                    gt_structure_names = [str(task)]

        # Fallback: totalseg candidates_total_dir + proxy structures (e.g., lung lobes).
        if (not gt_boxes_mm or not gt_mask_path) and structure_set:
            stem = _strip_nii_suffix(volume_name) or _strip_nii_suffix(case_id)
            if stem and candidates_available:
                cand_path = candidates_dir / f"{stem}.candidates.json"
                if not cand_path.exists():
                    skipped_missing_candidates += 1
                else:
                    totalseg_cand_path = str(cand_path)
                    candidates = _load_candidates(cand_path)
                    selected: List[Dict[str, Any]] = []
                    for c in candidates:
                        meta = c.get("meta", {})
                        if not isinstance(meta, dict):
                            continue
                        if str(meta.get("structure_name", "")).strip() in structure_set:
                            selected.append(c)
                    selected.sort(key=lambda c: str((c.get("meta") or {}).get("structure_name", "")))

                    seg_paths: List[str] = []
                    sources: List[str] = []
                    for c in selected:
                        meta = c.get("meta", {})
                        if not isinstance(meta, dict):
                            continue
                        bbox_zyx = _safe_int_list(meta.get("bbox_zyx", None), n=6)
                        if bbox_zyx is None:
                            continue
                        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox_zyx]
                        box = _bbox_xyz_vox_to_box_mm(
                            [int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)],
                            volume_shape_xyz=volume_shape_xyz,
                            target_shape_cdhw=target_shape,
                            token_spacing_xyz_mm=token_spacing_xyz_mm,
                        )
                        if box is None:
                            continue
                        gt_boxes_mm.append([float(x) for x in box])

                        structure_name = str(meta.get("structure_name", "")).strip()
                        if structure_name:
                            gt_structure_names.append(structure_name)
                        try:
                            gt_label_ids.append(int(meta.get("label_id", -1)))
                        except Exception:
                            gt_label_ids.append(-1)

                        seg_path = str(meta.get("seg_path", "")).strip() or str(meta.get("mask_path", "")).strip()
                        if seg_path:
                            seg_paths.append(seg_path)
                        src = str(meta.get("source", "")).strip()
                        if src:
                            sources.append(src)

                    seg_paths = sorted({str(p) for p in seg_paths if str(p).strip()})
                    gt_mask_path = seg_paths[0] if seg_paths else ""
                    if gt_mask_path and Path(gt_mask_path).exists():
                        gt_source = "totalseg.candidates_total_dir"
                        totalseg_source = ",".join(sorted({str(x) for x in sources if str(x).strip()}))

        # Final validation for this row.
        gt_boxes_mm.sort(key=lambda b: (float(b[0]), float(b[2]), float(b[4]), float(b[1]), float(b[3]), float(b[5])))
        gt_structure_names = sorted({str(x) for x in gt_structure_names if str(x).strip()})
        gt_label_ids = sorted({int(x) for x in gt_label_ids if int(x) >= 0})

        if not gt_boxes_mm:
            skipped_missing_boxes += 1
            continue
        if not gt_mask_path or not Path(gt_mask_path).exists():
            skipped_missing_boxes += 1
            continue

        out_rows.append(
            {
                "case_id": case_id,
                "volume_path": volume_path,
                "report_path": str(row.get("report_path", "")).strip(),
                "split": str(row.get("split", "")).strip(),
                "gt_mask_path": gt_mask_path,
                "gt_label_ids": gt_label_ids,
                "gt_structure_names": gt_structure_names,
                "grounding_boxes_by_sent_mm": {"0": gt_boxes_mm},
                "totalseg_candidates_path": str(totalseg_cand_path),
                "totalseg_source": str(totalseg_source),
                "gt_source": str(gt_source),
                "gt_is_pseudo": True,
                "coord_system": "token_space_mm",
                "volume_shape_xyz": list(volume_shape_xyz),
                "target_shape_cdhw": list(target_shape),
                "downsample_steps_xyz": list(downsample_steps_xyz),
                "token_spacing_xyz_mm": list(token_spacing_xyz_mm),
            }
        )

    # Deterministic final ordering.
    out_rows.sort(key=lambda r: str(r.get("case_id", "")).strip())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "manifest.jsonl"
    out_manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
        encoding="utf-8",
    )
    summary = {
        "timestamp_utc": _utc_now_iso(),
        "task": str(task),
        "in_mode": str(in_mode),
        "in_manifest": str(in_manifest_jsonl),
        "out_manifest": str(out_manifest),
        "n_rows": int(len(out_rows)),
        "max_cases": int(max_cases),
        "skipped_missing_label": int(skipped_missing_label),
        "skipped_missing_effusions_meta": int(skipped_missing_effusions_meta),
        "skipped_missing_effusions_mask": int(skipped_missing_effusions_mask),
        "skipped_missing_ts_meta": int(skipped_missing_ts_meta),
        "skipped_missing_ts_mask": int(skipped_missing_ts_mask),
        "skipped_missing_candidates": int(skipped_missing_candidates),
        "skipped_missing_boxes": int(skipped_missing_boxes),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CT-RATE TotalSegmentator GT-box manifest (repo-skeleton).")
    parser.add_argument("--in", dest="in_manifest", required=True, help="Input manifest.jsonl (CT-RATE labeled subset)")
    parser.add_argument("--task", required=True, help="Task name (e.g., lung_nodules, pleural_effusion, pericardial_effusion)")
    parser.add_argument(
        "--in-mode",
        default="labeled_manifest",
        choices=["labeled_manifest", "ts_index"],
        help="Input mode: labeled_manifest (filter by labels_pos for lung_nodules) or ts_index (cases.jsonl from build_ts_nodule_case_index).",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="YAML config (ct_rate_ts_grounding_e0907.yaml)")
    parser.add_argument("--max-cases", type=int, default=0, help="Max number of cases to write (0 = no limit)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] missing config: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = _load_yaml(cfg_path)

    in_manifest = Path(args.in_manifest)
    if not in_manifest.exists():
        print(f"[ERR] missing input manifest: {in_manifest}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    try:
        summary = build_ct_rate_ts_manifest(
            in_manifest_jsonl=in_manifest,
            in_mode=str(args.in_mode),
            out_dir=out_dir,
            cfg=cfg,
            task=str(args.task),
            max_cases=int(args.max_cases or 0),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
