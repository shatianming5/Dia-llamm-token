from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_rates(payload: Dict[str, Any]) -> Tuple[float | None, float | None, float | None]:
    u = payload.get("unsupported_rate", {})
    if not isinstance(u, dict):
        return None, None, None
    base = None
    removed = None
    swapped = None
    if "base" in u:
        try:
            base = float(u.get("base", 0.0))
        except Exception:
            base = None
    if "citation_swap" in u:
        try:
            swapped = float(u.get("citation_swap", 0.0))
        except Exception:
            swapped = None
    if "remove_citations" in u:
        try:
            removed = float(u.get("remove_citations", 0.0))
        except Exception:
            removed = None
    return base, removed, swapped


def _get_ground_hits(payload: Dict[str, Any]) -> Tuple[float | None, float | None, float | None]:
    g = payload.get("ground_hit@0.1", {})
    if not isinstance(g, dict):
        return None, None, None
    base = None
    perm = None
    swapped = None
    if "base" in g:
        try:
            base = float(g.get("base", 0.0))
        except Exception:
            base = None
    if "permute_omega" in g:
        try:
            perm = float(g.get("permute_omega", 0.0))
        except Exception:
            perm = None
    if "swap_citations" in g:
        try:
            swapped = float(g.get("swap_citations", 0.0))
        except Exception:
            swapped = None
    return base, perm, swapped


def validate_counterfactuals(
    payload: Dict[str, Any],
    *,
    require_swap_gt_base: bool = False,
    require_remove_gt_base: bool = False,
    require_remove_ge: float | None = None,
    require_base_le: float | None = None,
    require_ground_hit01_permute_lt_base: bool = False,
    require_ground_hit01_swap_lt_base: bool = False,
    require_ground_hit01_drop_ge: float | None = None,
) -> List[str]:
    errors: List[str] = []

    base, removed, swapped = _get_rates(payload)
    if base is None:
        errors.append("missing or invalid unsupported_rate.base")
    if require_swap_gt_base and swapped is None:
        errors.append("missing or invalid unsupported_rate.citation_swap")
    if removed is None:
        errors.append("missing or invalid unsupported_rate.remove_citations")

    if errors:
        return errors

    assert base is not None
    assert removed is not None

    if require_swap_gt_base:
        assert swapped is not None
        if not (swapped > base):
            errors.append(f"citation_swap({swapped}) is not > base({base})")

    if require_remove_gt_base and not (removed > base):
        errors.append(f"remove_citations({removed}) is not > base({base})")

    if require_remove_ge is not None:
        thr = float(require_remove_ge)
        if not (removed >= thr):
            errors.append(f"remove_citations({removed}) is not >= {thr}")

    if require_base_le is not None:
        thr = float(require_base_le)
        if not (base <= thr):
            errors.append(f"base({base}) is not <= {thr}")

    if require_ground_hit01_permute_lt_base or require_ground_hit01_swap_lt_base or require_ground_hit01_drop_ge is not None:
        gh_base, gh_perm, gh_swap = _get_ground_hits(payload)
        if gh_base is None:
            errors.append("missing or invalid ground_hit@0.1.base")
        if require_ground_hit01_permute_lt_base and gh_perm is None:
            errors.append("missing or invalid ground_hit@0.1.permute_omega")
        if require_ground_hit01_swap_lt_base and gh_swap is None:
            errors.append("missing or invalid ground_hit@0.1.swap_citations")

        if errors:
            return errors

        assert gh_base is not None

        def _drop(x: float) -> float:
            return float(gh_base) - float(x)

        if require_ground_hit01_permute_lt_base:
            assert gh_perm is not None
            if not (gh_perm < gh_base):
                errors.append(f"permute_omega ground_hit@0.1({gh_perm}) is not < base({gh_base})")

        if require_ground_hit01_swap_lt_base:
            assert gh_swap is not None
            if not (gh_swap < gh_base):
                errors.append(f"swap_citations ground_hit@0.1({gh_swap}) is not < base({gh_base})")

        if require_ground_hit01_drop_ge is not None:
            thr = float(require_ground_hit01_drop_ge)
            # Prefer permute_omega if present; otherwise swap.
            if gh_perm is not None:
                if not (_drop(gh_perm) >= thr):
                    errors.append(f"permute_omega ground_hit@0.1 drop({_drop(gh_perm)}) is not >= {thr}")
            elif gh_swap is not None:
                if not (_drop(gh_swap) >= thr):
                    errors.append(f"swap_citations ground_hit@0.1 drop({_drop(gh_swap)}) is not >= {thr}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate counterfactual evaluation outputs.")
    parser.add_argument("--in", dest="counterfactuals_json", required=True, help="Path to counterfactuals.json")
    parser.add_argument("--require-citation-swap-gt-base", action="store_true")
    parser.add_argument("--require-remove-citations-gt-base", action="store_true")
    parser.add_argument("--require-remove-citations-ge", type=float, default=None)
    parser.add_argument("--require-base-le", type=float, default=None)
    parser.add_argument("--require-ground-hit01-permute-lt-base", action="store_true")
    parser.add_argument("--require-ground-hit01-swap-lt-base", action="store_true")
    parser.add_argument("--require-ground-hit01-drop-ge", type=float, default=None)
    args = parser.parse_args()

    payload = _load_json(Path(args.counterfactuals_json))
    errors = validate_counterfactuals(
        payload,
        require_swap_gt_base=bool(args.require_citation_swap_gt_base),
        require_remove_gt_base=bool(args.require_remove_citations_gt_base),
        require_remove_ge=args.require_remove_citations_ge,
        require_base_le=args.require_base_le,
        require_ground_hit01_permute_lt_base=bool(args.require_ground_hit01_permute_lt_base),
        require_ground_hit01_swap_lt_base=bool(args.require_ground_hit01_swap_lt_base),
        require_ground_hit01_drop_ge=args.require_ground_hit01_drop_ge,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] counterfactuals.json validated")


if __name__ == "__main__":
    main()
