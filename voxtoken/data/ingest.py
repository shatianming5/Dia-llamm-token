from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            yield {str(k): ("" if v is None else str(v)) for k, v in row.items()}


def _ct_rate_csv(root: Path, split: str) -> Path:
    s = str(split or "").strip().lower()
    if s in {"val", "valid", "validation"}:
        name = "validation_reports.csv"
    else:
        name = "train_reports.csv"
    return root / "dataset" / "radiology_text_reports" / name


def _ct_rate_predicted_labels_csv(root: Path, split: str) -> Path:
    s = str(split or "").strip().lower()
    if s in {"val", "valid", "validation"}:
        name = "valid_predicted_labels.csv"
    else:
        name = "train_predicted_labels.csv"
    return root / "dataset" / "multi_abnormality_labels" / name


def _load_ct_rate_predicted_labels(path: Path) -> Dict[str, List[str]]:
    """
    Returns:
        {VolumeName -> [positive_label_name, ...]}
    """
    if not path.exists():
        raise FileNotFoundError(f"CT-RATE predicted labels CSV not found: {path}")

    out: Dict[str, List[str]] = {}
    for row in _iter_csv_rows(path):
        vol = str(row.get("VolumeName", "")).strip()
        if not vol:
            continue
        pos: List[str] = []
        for k, v in row.items():
            if k == "VolumeName":
                continue
            s = str(v).strip()
            if not s:
                continue
            try:
                val = float(s)
            except Exception:
                continue
            if val > 0.0:
                pos.append(str(k))
        out[vol] = pos
    return out


def _build_volume_index(search_roots: List[Path], *, max_files: int = 20000) -> Dict[str, str]:
    """
    Build a small basename->path index for resolving VolumeName into a concrete file path.

    Safety: do not scan arbitrarily large trees unless the caller opts in via config.
    """
    index: Dict[str, str] = {}
    seen = 0
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.nii.gz"):
            index[p.name] = str(p)
            seen += 1
            if seen >= int(max_files):
                return index
    return index


def _normalize_ct_rate_root(path: str) -> Path:
    # User sometimes refers to "/data/ct_rate"; on this machine the mount is "/data/CT-RATE".
    p = Path(str(path or "")).expanduser()
    if p.exists():
        return p
    fallback = Path("/data/CT-RATE")
    if str(p) == "/data/ct_rate" and fallback.exists():
        return fallback
    return p


def _ingest_synthetic(cfg: Dict[str, Any]) -> None:
    out_dir = Path(str(cfg.get("out_dir", "data/raw")))
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cases = int(cfg.get("num_cases", 1))
    manifest_path = Path(str(cfg.get("manifest_path", out_dir / "manifest.jsonl")))

    lines: List[str] = []
    for i in range(num_cases):
        case_id = f"case-{i:04d}"
        vol_path = out_dir / f"{case_id}.vol.json"
        rpt_path = out_dir / f"{case_id}.txt"

        vol = {"shape_cdhw": [1, 8, 8, 8], "fill": 0.0}
        vol_path.write_text(json.dumps(vol, ensure_ascii=False, indent=2), encoding="utf-8")
        rpt_path.write_text("No findings.\n", encoding="utf-8")

        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "volume_path": str(vol_path),
                    "report_path": str(rpt_path),
                    "source": "synthetic",
                },
                ensure_ascii=False,
            )
        )

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _ingest_ct_rate(cfg: Dict[str, Any]) -> None:
    root = _normalize_ct_rate_root(str(cfg.get("ct_rate_root", "/data/CT-RATE")))
    if not root.exists():
        raise FileNotFoundError(f"ct_rate_root not found: {root}")

    out_dir = Path(str(cfg.get("out_dir", "data/raw")))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split = str(cfg.get("split", "validation"))
    num_cases = int(cfg.get("num_cases", 20))
    seed = int(cfg.get("seed", 0))
    manifest_path = Path(str(cfg.get("manifest_path", out_dir / "manifest.jsonl")))

    report_fields = cfg.get("report_fields") or ["Findings_EN", "Impressions_EN"]
    if not isinstance(report_fields, list) or not report_fields:
        report_fields = ["Findings_EN", "Impressions_EN"]
    report_fields = [str(x) for x in report_fields]

    include_predicted_labels = bool(cfg.get("include_predicted_labels", False) or cfg.get("include_labels", False))
    labels_pos_by_volume: Dict[str, List[str]] = {}
    if include_predicted_labels:
        labels_csv = cfg.get("predicted_labels_csv")
        labels_path = Path(str(labels_csv)) if labels_csv else _ct_rate_predicted_labels_csv(root, split)
        if not labels_path.is_absolute():
            labels_path = root / labels_path
        labels_pos_by_volume = _load_ct_rate_predicted_labels(labels_path)

    # Optionally resolve volume paths by scanning a restricted set of search roots.
    volume_search_roots_raw = cfg.get("volume_search_roots") or []
    if not isinstance(volume_search_roots_raw, list):
        volume_search_roots_raw = []
    volume_search_roots: List[Path] = []
    for item in volume_search_roots_raw:
        p = Path(str(item))
        if not p.is_absolute():
            p = root / p
        volume_search_roots.append(p)

    volume_index: Dict[str, str] = {}
    if volume_search_roots:
        volume_index = _build_volume_index(volume_search_roots, max_files=int(cfg.get("max_volume_files", 20000)))

    prefer_resolved = bool(cfg.get("prefer_resolved_volume", False))
    # Deterministic reservoir-ish selection: iterate in file order; optionally filter for resolved volumes.
    # If prefer_resolved is enabled, we keep scanning until we collect num_cases or hit a cap.
    max_rows = int(cfg.get("max_rows", 200000))
    taken = 0
    seen = 0

    # Create deterministic but stable ordering when multiple passes are needed: we can offset by seed.
    # For simplicity we just skip the first `seed % 997` matching rows to vary subsets deterministically.
    skip = int(seed) % 997
    skipped = 0

    lines: List[str] = []
    csv_path = _ct_rate_csv(root, split)
    if not csv_path.exists():
        raise FileNotFoundError(f"CT-RATE reports CSV not found: {csv_path}")

    for row in _iter_csv_rows(csv_path):
        seen += 1
        if seen > max_rows:
            break

        vol_name = str(row.get("VolumeName", "")).strip()
        if not vol_name:
            continue

        volume_path = ""
        if volume_index and vol_name in volume_index:
            volume_path = str(volume_index[vol_name])

        if prefer_resolved and not volume_path:
            continue

        if skipped < skip:
            skipped += 1
            continue

        case_id = vol_name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[: -len(".nii.gz")]
        case_id = case_id.strip()
        if not case_id:
            continue

        chunks: List[str] = []
        for key in report_fields:
            val = str(row.get(key, "")).strip()
            if val:
                chunks.append(val)
        report_text = "\n".join(chunks).strip() + ("\n" if chunks else "")

        rpt_path = reports_dir / f"{case_id}.txt"
        rpt_path.write_text(report_text, encoding="utf-8")

        labels_pos: List[str] = []
        if include_predicted_labels:
            labels_pos = list(labels_pos_by_volume.get(vol_name, []))

        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "volume_name": vol_name,
                    "volume_path": volume_path,
                    "report_path": str(rpt_path),
                    "source": "ct-rate",
                    "dataset_split": str(split).strip().lower(),
                    **({"labels_pos": labels_pos, "labels_source": "ct-rate/predicted"} if include_predicted_labels else {}),
                },
                ensure_ascii=False,
            )
        )
        taken += 1
        if taken >= num_cases:
            break

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def _grid_boxes_mm(
    *,
    shape_dhw: List[int],
    voxel_spacing_mm: List[float],
    patch: int,
) -> List[List[float]]:
    d, h, w = [max(1, int(x)) for x in (shape_dhw or [8, 8, 8])]
    sx, sy, sz = [float(x) for x in (voxel_spacing_mm or [1.0, 1.0, 1.0])]
    patch = max(1, int(patch))

    boxes: List[List[float]] = []
    for z0 in range(0, d, patch):
        z1 = min(d, z0 + patch)
        for y0 in range(0, h, patch):
            y1 = min(h, y0 + patch)
            for x0 in range(0, w, patch):
                x1 = min(w, x0 + patch)
                boxes.append(
                    [
                        float(x0) * sx,
                        float(x1) * sx,
                        float(y0) * sy,
                        float(y1) * sy,
                        float(z0) * sz,
                        float(z1) * sz,
                    ]
                )
    return boxes


def _ingest_radgenome(cfg: Dict[str, Any]) -> None:
    """
    Minimal RadGenome-like ingest for grounding experiments.

    If the configured root does not exist, we fall back to a synthetic dataset that
    provides sentence->box grounding targets (in mm) to keep the M2 pipeline runnable.
    """
    root = Path(str(cfg.get("radgenome_root", "/data/radgenome_chestct"))).expanduser()

    out_dir = Path(str(cfg.get("out_dir", "data/raw")))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    num_cases = int(cfg.get("num_cases", 20))
    seed = int(cfg.get("seed", 0))
    manifest_path = Path(str(cfg.get("manifest_path", out_dir / "manifest.jsonl")))

    split = str(cfg.get("split", "train")).strip().lower()
    use_region_report = bool(cfg.get("use_region_report", False))

    shape_dhw = cfg.get("shape_dhw") or cfg.get("shape") or [8, 8, 8]
    try:
        d, h, w = [int(x) for x in shape_dhw]
    except Exception:
        d, h, w = 8, 8, 8
    spacing = cfg.get("voxel_spacing_mm") or [1.0, 1.0, 1.0]
    try:
        sx, sy, sz = [float(x) for x in spacing]
    except Exception:
        sx, sy, sz = 1.0, 1.0, 1.0
    patch = int(cfg.get("patch", 4))
    gt_mode = str(cfg.get("gt_mode", "grid") or "grid").strip().lower()
    gt_shift_vox = cfg.get("gt_shift_vox", None)
    try:
        gt_shift_vox_i = int(gt_shift_vox) if gt_shift_vox is not None else None
    except Exception:
        gt_shift_vox_i = None

    boxes = _grid_boxes_mm(shape_dhw=[int(d), int(h), int(w)], voxel_spacing_mm=[float(sx), float(sy), float(sz)], patch=int(patch))

    lines: List[str] = []

    if root.exists():
        # Best-effort: if a manifest.jsonl exists, pass it through (grounding field optional).
        src_manifest = root / "manifest.jsonl"
        if src_manifest.exists():
            for line in src_manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lines.append(line.strip())
            manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            return

        # Real dataset mode (local path): RadGenome-ChestCT provides preprocessed volumes.
        dataset_dir = root / "dataset"
        volumes_root = Path(str(cfg.get("volumes_root", dataset_dir / "train_preprocessed"))).expanduser()
        if not volumes_root.is_absolute():
            volumes_root = root / volumes_root

        # If the expected structure exists and we can find at least one volume, ingest a small subset.
        if volumes_root.exists():
            max_files = int(cfg.get("max_volume_files", 50000))
            volume_index: Dict[str, str] = _build_volume_index([volumes_root], max_files=max_files)
            # Filter to radgenome-style train volumes by default (still allow others if user wants).
            only_prefix = str(cfg.get("volume_name_prefix", "train_")).strip()
            vol_items = [(name, path) for name, path in volume_index.items() if not only_prefix or str(name).startswith(only_prefix)]
            vol_items.sort(key=lambda x: str(x[0]))

            # Deterministic skip to vary the subset by seed.
            skip = int(seed) % 997
            if skip > 0 and len(vol_items) > 0:
                vol_items = vol_items[skip:] + vol_items[:skip]

            take = int(num_cases)
            if take <= 0:
                take = len(vol_items)
            selected = vol_items[:take]

            # Optional: write a minimally useful report from train_region_report.csv.
            # This can be large; keep disabled by default.
            sentences_by_volume: Dict[str, List[str]] = {}
            if use_region_report:
                csv_override = cfg.get("region_report_csv", None)
                if csv_override:
                    rr_csv = Path(str(csv_override)).expanduser()
                    if not rr_csv.is_absolute():
                        rr_csv = root / rr_csv
                else:
                    rr_csv = dataset_dir / "radgenome_files" / ("validation_region_report.csv" if split in {"val", "valid", "validation"} else "train_region_report.csv")
                want = {str(name) for name, _ in selected}
                if rr_csv.exists() and want:
                    for row in _iter_csv_rows(rr_csv):
                        vn = str(row.get("Volumename", "")).strip()
                        if vn not in want:
                            continue
                        anatomy = str(row.get("Anatomy", "")).strip()
                        sent = str(row.get("Sentence", "")).strip()
                        if not sent:
                            continue
                        line = f"[{anatomy}] {sent}" if anatomy else sent
                        sentences_by_volume.setdefault(vn, []).append(line)

            for vol_name, vol_path in selected:
                case_id = str(vol_name)
                if case_id.endswith(".nii.gz"):
                    case_id = case_id[: -len(".nii.gz")]
                case_id = case_id.strip()
                if not case_id:
                    continue

                rpt_path = reports_dir / f"{case_id}.txt"
                if use_region_report and str(vol_name) in sentences_by_volume:
                    content = "\n".join(sentences_by_volume.get(str(vol_name), [])) + "\n"
                else:
                    content = f"RadGenome-ChestCT placeholder report for {case_id}.\n"
                rpt_path.write_text(content, encoding="utf-8")

                lines.append(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "volume_name": str(vol_name),
                            "volume_path": str(vol_path),
                            "report_path": str(rpt_path),
                            "source": "radgenome-chestct",
                            "dataset_split": str(split),
                        },
                        ensure_ascii=False,
                    )
                )

            if lines:
                manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return

    # Synthetic fallback: write a small manifest where each sentence maps to one box.
    for i in range(max(1, int(num_cases))):
        case_id = f"rg_{i:04d}"
        rpt_path = reports_dir / f"{case_id}.txt"
        # Minimal placeholder report (not used by inference in this repo skeleton).
        rpt_path.write_text("\n".join([f"Finding_{j}." for j in range(len(boxes))]) + "\n", encoding="utf-8")

        grounding: Dict[str, Any] = {}
        child_patch = max(1, int(patch) // 2)
        # Default shift: half-child so the GT box straddles 2 children along x.
        shift_vox = gt_shift_vox_i
        if shift_vox is None:
            shift_vox = max(0, min(int(child_patch) - 1, int(child_patch) // 2))

        for sent_id, b in enumerate(boxes):
            x0, x1, y0, y1, z0, z1 = [float(x) for x in b]

            if gt_mode == "grid":
                gt_box = [x0, x1, y0, y1, z0, z1]
            else:
                dx = float(child_patch) * float(sx)
                dy = float(child_patch) * float(sy)
                dz = float(child_patch) * float(sz)
                gx0 = float(x0)
                if gt_mode == "mixed_child" and (int(sent_id) % 2 == 0):
                    gx0 = float(x0) + float(shift_vox) * float(sx)
                gt_box = [gx0, float(gx0) + float(dx), y0, float(y0) + float(dy), z0, float(z0) + float(dz)]

            grounding[str(int(sent_id))] = [gt_box]

        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "volume_path": "",  # keep empty so inference uses dummy volume without NIfTI deps
                    "report_path": str(rpt_path),
                    "source": "radgenome-synth",
                    "seed": int(seed),
                    "volume_shape_dhw": [int(d), int(h), int(w)],
                    "voxel_spacing_mm": [float(sx), float(sy), float(sz)],
                    "token_patch": int(patch),
                    "gt_mode": str(gt_mode),
                    "grounding_boxes_by_sent_mm": grounding,
                },
                ensure_ascii=False,
            )
        )

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ingest(cfg: Dict[str, Any]) -> None:
    """Download/ingest datasets into a normalized on-disk layout."""
    source = str(cfg.get("source", "synthetic")).strip().lower().replace("_", "-")
    if source in {"ct-rate", "ctrate"}:
        _ingest_ct_rate(cfg)
        return
    if source in {"radgenome", "rad-genome"}:
        _ingest_radgenome(cfg)
        return
    _ingest_synthetic(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset ingest (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    ingest(cfg)

    out_dir = str(cfg.get("out_dir", "data/raw"))
    manifest_path = str(cfg.get("manifest_path", str(Path(out_dir) / "manifest.jsonl")))
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": out_dir,
        "manifest_path": manifest_path,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
