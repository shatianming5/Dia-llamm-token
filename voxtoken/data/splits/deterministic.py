from __future__ import annotations


def assign_split(case_id: str, *, seed: int, train_p: float, val_p: float) -> str:
    """
    Deterministic split assignment used by the repo skeleton.

    Notes:
      - Pure function (stdlib-only), intended to keep preprocess reproducible.
      - `train_p` and `val_p` should sum to <= 1.0.
    """

    import hashlib

    s = str(case_id or "")
    seed_s = str(int(seed))

    # Use a stable hash with good mixing to avoid pathological splits for similar prefixes
    # (e.g., many case_ids starting with "valid_").
    h = hashlib.blake2b(f"{seed_s}:{s}".encode("utf-8"), digest_size=8).digest()
    u64 = int.from_bytes(h, "big", signed=False)
    r = float(u64) / float(2**64)

    if r < float(train_p):
        return "train"
    if r < (float(train_p) + float(val_p)):
        return "val"
    return "test"
