from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _normalize_mask_filename(mask: str) -> str:
    s = str(mask or "").strip()
    if not s:
        return ""
    if s.lower().endswith(".nii.gz"):
        return s
    return f"{s}.nii.gz"

def _is_gzip_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


class _ConcatReader(io.RawIOBase):
    """
    Read multiple file parts sequentially as one byte stream.

    RadGenome-ChestCT ships anatomy masks as a gzip-compressed tar that is split into
    files `train_anatomy_mask_aa`, `train_anatomy_mask_ab`, ... (like `split -b ...`).
    Only the first part has a gzip header; the remaining parts are raw continuation
    bytes and are not valid gzip files on their own.
    """

    def __init__(self, parts: List[Path]):
        super().__init__()
        self._parts = [Path(p) for p in parts]
        self._idx = 0
        self._fh = None

    def readable(self) -> bool:  # noqa: D401
        return True

    def _open_next(self) -> bool:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        while self._idx < len(self._parts):
            p = self._parts[self._idx]
            self._idx += 1
            if not p.exists():
                continue
            self._fh = p.open("rb")
            return True
        return False

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        if self.closed:
            return 0
        if self._fh is None and not self._open_next():
            return 0

        mv = memoryview(b)
        n_total = 0
        while n_total < len(mv):
            if self._fh is None and not self._open_next():
                break
            try:
                n = self._fh.readinto(mv[n_total:])  # type: ignore[union-attr]
            except Exception:
                n = 0
            if not n:
                if not self._open_next():
                    break
                continue
            n_total += int(n)
        return int(n_total)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        finally:
            self._fh = None
            super().close()


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


def _mask_bbox_xyz(mask_path: Path) -> List[int] | None:
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nibabel + numpy are required to compute mask bounding boxes") from exc

    img = nib.load(str(mask_path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim < 3:
        return None
    if arr.ndim > 3:
        arr3 = arr[..., 0]
    else:
        arr3 = arr
    nz = np.argwhere(arr3 > 0)
    if nz.size == 0:
        return None
    mins = nz.min(axis=0)
    maxs = nz.max(axis=0) + 1
    x0, y0, z0 = [int(x) for x in mins[:3]]
    x1, y1, z1 = [int(x) for x in maxs[:3]]
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        return None
    return [int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)]


def _iter_selected_rows(
    rows: List[Dict[str, Any]],
    *,
    split_filter: str,
    max_total: int,
    max_train: int,
    max_val: int,
    max_test: int,
) -> Iterable[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    yielded = 0
    for row in rows:
        split = str(row.get("split", "")).strip()
        if split_filter and split != split_filter:
            continue

        cap = None
        if split == "train" and max_train >= 0:
            cap = int(max_train)
        elif split == "val" and max_val >= 0:
            cap = int(max_val)
        elif split == "test" and max_test >= 0:
            cap = int(max_test)

        if cap is not None and cap > 0 and counts[split] >= cap:
            continue
        if max_total > 0 and yielded >= int(max_total):
            break

        counts[split] += 1
        yielded += 1
        yield row


def _extract_members_from_archives(
    *,
    archives: List[Path],
    want_member_to_out_path: Dict[str, Path],
) -> Dict[str, str]:
    remaining = {k for k in want_member_to_out_path.keys() if str(k).strip()}
    found_in: Dict[str, str] = {}
    if not remaining:
        return found_in

    for archive in archives:
        if not remaining:
            break
        if not archive.exists():
            continue

        # Stream mode: avoids loading the full tar index into memory.
        #
        # Note: Some public dataset shards can be truncated (missing the final tar EOF blocks).
        # In that case, Python's tarfile may raise ReadError while iterating. We treat it as
        # "best-effort done" for this shard and continue scanning other shards.
        try:
            with tarfile.open(str(archive), mode="r|gz") as tf:
                try:
                    for member in tf:
                        name = str(getattr(member, "name", "") or "")
                        if name not in remaining:
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        out_path = want_member_to_out_path[name]
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with out_path.open("wb") as out_f:
                            shutil.copyfileobj(f, out_f, length=1024 * 1024)
                        found_in[name] = str(archive)
                        remaining.remove(name)
                        if not remaining:
                            break
                except tarfile.ReadError:
                    continue
        except tarfile.ReadError:
            continue
    return found_in


def build_radgenome_mask_manifest(
    *,
    in_manifest_jsonl: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    radgenome_root: Path,
    mask: str,
    split: str,
    max_cases: int,
    max_cases_train: int,
    max_cases_val: int,
    max_cases_test: int,
) -> Dict[str, Any]:
    rows = _load_jsonl(in_manifest_jsonl)
    rows.sort(key=lambda r: str(r.get("case_id", "")).strip())

    split_norm = str(split or "").strip()
    if split_norm and split_norm not in {"train", "val", "test"}:
        raise ValueError(f"invalid --split: {split_norm} (expected train|val|test)")

    mask_filename = _normalize_mask_filename(mask)
    if not mask_filename:
        raise ValueError("--mask is required")

    dataset_dir = radgenome_root / "dataset"
    parts = sorted([p for p in dataset_dir.glob("train_anatomy_mask_*") if p.is_file()])
    if not parts:
        raise FileNotFoundError(f"no train_anatomy_mask_* files found under: {dataset_dir}")

    # RadGenome-ChestCT anatomy masks are typically provided as a single tar.gz split across
    # `train_anatomy_mask_aa`, `train_anatomy_mask_ab`, ... where only the first part has
    # a gzip header. Detect and read them as one concatenated stream.
    use_concat = bool(len(parts) > 1 and _is_gzip_file(parts[0]) and all(not _is_gzip_file(p) for p in parts[1:]))
    archive_groups: List[List[Path]] = [list(parts)] if use_concat else [[p] for p in parts]

    target_shape = _target_shape_cdhw(cfg)
    token_spacing_xyz_mm = _token_voxel_spacing_xyz_mm(cfg)

    out_rows: List[Dict[str, Any]] = []
    counts_written: Counter[str] = Counter()
    attempted: set[str] = set()

    # Eligibility: only cases present in the input manifest (and split-filtered, if requested).
    row_by_case_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in row_by_case_id:
            continue
        sp = str(row.get("split", "")).strip()
        if split_norm and sp != split_norm:
            continue
        vp = str(row.get("volume_path", "")).strip()
        if not vp or not Path(vp).exists():
            continue
        row_by_case_id[case_id] = row

    if not row_by_case_id:
        raise ValueError("no eligible rows found in input manifest (missing split/volume_path?)")

    prefix = "train_anatomy_mask/seg_"
    suffix = f"/{mask_filename}"

    n_seen_mask_members = 0
    n_missing_case = 0
    n_empty_mask = 0
    n_bad_box = 0
    n_quota_skip = 0

    def cap_reached(name: str, cap: int) -> bool:
        # cap<=0 means "unlimited / don't require".
        if int(cap) <= 0:
            return True
        return int(counts_written.get(name, 0)) >= int(cap)

    def done() -> bool:
        if int(max_cases) > 0 and int(len(out_rows)) >= int(max_cases):
            return True
        if split_norm:
            cap = 0
            if split_norm == "train":
                cap = int(max_cases_train)
            elif split_norm == "val":
                cap = int(max_cases_val)
            else:
                cap = int(max_cases_test)
            return cap_reached(split_norm, cap)
        return cap_reached("train", int(max_cases_train)) and cap_reached("val", int(max_cases_val)) and cap_reached("test", int(max_cases_test))

    # Scan the tar shards in order and accept the first cases we can resolve+extract,
    # to avoid seeking for specific case_ids (which would require scanning huge shards).
    for group in archive_groups:
        if done():
            break
        archive_label = "+".join([p.name for p in group]) if len(group) > 1 else group[0].name
        try:
            if len(group) == 1:
                tf_ctx = tarfile.open(str(group[0]), mode="r|gz")
            else:
                tf_ctx = tarfile.open(fileobj=_ConcatReader(group), mode="r|gz")
            with tf_ctx as tf:
                try:
                    for member in tf:
                        if done():
                            break
                        name = str(getattr(member, "name", "") or "")
                        if not name.startswith(prefix) or not name.endswith(suffix):
                            continue
                        n_seen_mask_members += 1
                        rest = name[len(prefix) :]
                        if not rest.endswith(suffix):
                            continue
                        case_id = rest[: -len(suffix)]
                        if not case_id or case_id in attempted:
                            continue
                        row = row_by_case_id.get(case_id, None)
                        if row is None:
                            n_missing_case += 1
                            continue

                        sp = str(row.get("split", "")).strip()
                        cap = 0
                        if sp == "train":
                            cap = int(max_cases_train)
                        elif sp == "val":
                            cap = int(max_cases_val)
                        elif sp == "test":
                            cap = int(max_cases_test)
                        else:
                            continue
                        if int(cap) > 0 and int(counts_written.get(sp, 0)) >= int(cap):
                            n_quota_skip += 1
                            continue
                        if int(max_cases) > 0 and int(len(out_rows)) >= int(max_cases):
                            break

                        attempted.add(case_id)

                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        mask_path = out_dir / "masks" / case_id / mask_filename
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        with mask_path.open("wb") as out_f:
                            shutil.copyfileobj(f, out_f, length=1024 * 1024)

                        bbox_xyz = _mask_bbox_xyz(mask_path)
                        if bbox_xyz is None:
                            n_empty_mask += 1
                            continue

                        volume_path = Path(str(row.get("volume_path", "")).strip())
                        if not volume_path.exists():
                            continue

                        volume_shape_xyz = _nifti_shape_xyz(volume_path)
                        box_mm = _bbox_xyz_vox_to_box_mm(
                            bbox_xyz,
                            volume_shape_xyz=volume_shape_xyz,
                            target_shape_cdhw=target_shape,
                            token_spacing_xyz_mm=token_spacing_xyz_mm,
                        )
                        if box_mm is None:
                            n_bad_box += 1
                            continue

                        out_rows.append(
                            {
                                "case_id": case_id,
                                "volume_path": str(volume_path),
                                "report_path": str(row.get("report_path", "")).strip(),
                                "split": str(row.get("split", "")).strip(),
                                "gt_mask_path": str(mask_path),
                                "gt_structure_names": [
                                    str(mask_filename[: -len(".nii.gz")] if mask_filename.lower().endswith(".nii.gz") else mask_filename)
                                ],
                                "grounding_boxes_by_sent_mm": {"0": [[float(x) for x in box_mm]]},
                                "radgenome_mask_member": str(name),
                                "radgenome_mask_archive": str(archive_label),
                                "gt_source": "radgenome.chestct",
                                "gt_is_pseudo": True,
                                "coord_system": "token_space_mm",
                                "volume_shape_xyz": list(volume_shape_xyz),
                                "target_shape_cdhw": list(target_shape),
                                "downsample_steps_xyz": list(_downsample_steps_xyz(volume_shape_xyz, target_shape)),
                                "token_spacing_xyz_mm": list(token_spacing_xyz_mm),
                            }
                        )
                        counts_written[sp] += 1
                except tarfile.ReadError:
                    continue
        except tarfile.ReadError:
            continue

    out_rows.sort(key=lambda r: str(r.get("case_id", "")).strip())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "manifest.jsonl"
    out_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "in_manifest": str(in_manifest_jsonl),
        "out_manifest": str(out_manifest),
        "radgenome_root": str(radgenome_root),
        "mask": str(mask_filename),
        "split_filter": str(split_norm),
        "max_cases": int(max_cases),
        "max_cases_train": int(max_cases_train),
        "max_cases_val": int(max_cases_val),
        "max_cases_test": int(max_cases_test),
        "n_eligible_rows": int(len(row_by_case_id)),
        "n_written": int(len(out_rows)),
        "n_seen_mask_members": int(n_seen_mask_members),
        "n_missing_case": int(n_missing_case),
        "n_empty_mask": int(n_empty_mask),
        "n_bad_box": int(n_bad_box),
        "n_quota_skip": int(n_quota_skip),
        "counts_written": {k: int(v) for k, v in counts_written.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RadGenome-ChestCT GT-box manifest from anatomy masks (repo-skeleton).")
    parser.add_argument("--in", dest="in_manifest", required=True, help="Input manifest.jsonl (processed; must include split + volume_path)")
    parser.add_argument("--out", required=True, help="Output directory (writes manifest.jsonl + summary.json + extracted masks)")
    parser.add_argument("--config", required=True, help="YAML config (reuses volume.target_shape_cdhw + tokenizer.voxel_spacing_mm)")
    parser.add_argument("--radgenome-root", required=True, help="RadGenome-ChestCT root (must contain dataset/train_anatomy_mask_*)")
    parser.add_argument("--mask", required=True, help="Mask name (e.g., 'lung effusion' or 'lung effusion.nii.gz')")
    parser.add_argument("--split", default="", help="Optional split filter (train|val|test)")
    parser.add_argument("--max-cases", type=int, default=0, help="Max total cases to process (0 = no limit)")
    parser.add_argument("--max-cases-train", type=int, default=-1, help="Max train cases (0 = no limit, -1 = use --max-cases)")
    parser.add_argument("--max-cases-val", type=int, default=-1, help="Max val cases (0 = no limit, -1 = use --max-cases)")
    parser.add_argument("--max-cases-test", type=int, default=-1, help="Max test cases (0 = no limit, -1 = use --max-cases)")
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
    radgenome_root = Path(args.radgenome_root).expanduser()
    if not radgenome_root.exists():
        print(f"[ERR] radgenome_root not found: {radgenome_root}", file=sys.stderr)
        sys.exit(2)

    # If per-split caps are unset (-1), default them to the global cap.
    max_cases = int(args.max_cases or 0)
    max_train = int(args.max_cases_train)
    max_val = int(args.max_cases_val)
    max_test = int(args.max_cases_test)
    if max_train < 0:
        max_train = int(max_cases)
    if max_val < 0:
        max_val = int(max_cases)
    if max_test < 0:
        max_test = int(max_cases)

    try:
        summary = build_radgenome_mask_manifest(
            in_manifest_jsonl=in_manifest,
            out_dir=out_dir,
            cfg=cfg,
            radgenome_root=radgenome_root,
            mask=str(args.mask),
            split=str(args.split),
            max_cases=int(max_cases),
            max_cases_train=int(max_train),
            max_cases_val=int(max_val),
            max_cases_test=int(max_test),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
