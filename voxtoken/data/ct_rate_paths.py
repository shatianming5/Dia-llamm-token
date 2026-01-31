from __future__ import annotations

from pathlib import Path
from typing import Optional


def _strip_nii_suffix(name: str) -> str:
    s = str(name or "").strip()
    if s.endswith(".nii.gz"):
        return s[: -len(".nii.gz")]
    if s.endswith(".nii"):
        return s[: -len(".nii")]
    return s


def resolve_ct_rate_volume_path(mask_file: str, *, root: str = "/data/CT-RATE/dataset") -> Optional[str]:
    """
    Resolve a CT-RATE NIfTI path from a volume/mask filename like `valid_1_a_1.nii.gz`.

    Expected canonical layout (observed on this machine):
      /data/CT-RATE/dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz

    If the canonical parse fails, falls back to a filesystem search under `root`.
    """
    name = str(mask_file or "").strip()
    if not name:
        return None

    dataset_root = Path(str(root)).expanduser()
    stem = _strip_nii_suffix(name)
    parts = [p for p in stem.split("_") if p.strip()]
    if len(parts) >= 3:
        split = str(parts[0]).strip()
        pid = f"{split}_{parts[1]}"
        study = f"{split}_{parts[1]}_{parts[2]}"
        candidate = dataset_root / split / pid / study / name
        if candidate.exists():
            return str(candidate)

    # Fallback: try to locate by filename anywhere under the dataset root.
    if dataset_root.exists():
        matches = list(dataset_root.rglob(name))
        matches = [p for p in matches if p.is_file()]
        if len(matches) == 1:
            return str(matches[0])
    return None


__all__ = ["resolve_ct_rate_volume_path"]

