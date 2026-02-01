from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..torch_compat import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _dataset_path(cfg: Dict[str, Any]) -> Path:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    raw = cfg.get("dataset_jsonl") or train_cfg.get("dataset_jsonl")
    if not raw:
        raise ValueError("missing dataset_jsonl in config")
    return Path(str(raw)).expanduser()


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [list(map(float, row)) + [float(bi)] for row, bi in zip(a, b)]

    for col in range(n):
        pivot = col
        for r in range(col, n):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        denom = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= denom

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [float(m[i][n]) for i in range(n)]


def _fit_linear_weights(xs: List[List[float]], ys: List[float], *, ridge: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if len(xs) < 4:
        raise ValueError(f"dataset too small for 4D fit: n={len(xs)}")

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

    def pred(x: List[float]) -> float:
        return float(w[0]) * float(x[0]) + float(w[1]) * float(x[1]) + float(w[2]) * float(x[2]) + float(w[3]) * float(x[3])

    mse = 0.0
    for x, y in zip(xs, ys):
        mse += (float(y) - float(pred(x))) ** 2
    mse = float(mse) / float(len(xs)) if xs else 0.0

    weights = {
        "recon_error": float(w[0]),
        "evidence_entropy": float(w[1]),
        "citation_pressure": float(w[2]),
        "history_splits": float(w[3]),
    }
    meta = {"fit": "dataset", "n_samples": int(len(xs)), "mse": float(mse)}
    return weights, meta


def _extract_feature_vector(obj: Dict[str, Any], *, input_dim: int) -> List[float]:
    input_dim = int(input_dim)
    if int(input_dim) < 4:
        raise ValueError(f"model.input_dim must be >=4 (got {input_dim})")
    x = [
        float(obj.get("recon_error", 0.0)),
        float(obj.get("evidence_entropy", 0.0)),
        float(obj.get("citation_pressure", 0.0)),
        float(obj.get("history_splits", 0.0)),
    ]
    if int(input_dim) >= 5:
        x.append(float(obj.get("center_x_mm", 0.0)))
    if int(input_dim) >= 6:
        x.append(float(obj.get("center_y_mm", 0.0)))
    if int(input_dim) >= 7:
        x.append(float(obj.get("center_z_mm", 0.0)))
    if int(input_dim) >= 8:
        x.append(float(obj.get("mean_intensity", 0.0)))
    if int(input_dim) >= 9:
        x.append(float(obj.get("max_intensity", 0.0)))
    if int(input_dim) >= 10:
        x.append(float(obj.get("level", 0.0)))
    if int(input_dim) >= 11:
        x.append(float(obj.get("box_dx_mm", 0.0)))
    if int(input_dim) >= 12:
        x.append(float(obj.get("box_dy_mm", 0.0)))
    if int(input_dim) >= 13:
        x.append(float(obj.get("box_dz_mm", 0.0)))
    if int(input_dim) >= 14:
        x.append(float(obj.get("box_volume_mm3", 0.0)))
    if int(input_dim) >= 15:
        x.append(float(obj.get("step_idx", 0.0)))
    if int(input_dim) >= 16:
        x.append(float(obj.get("budget_B", 0.0)))
    if len(x) < int(input_dim):
        x = x + [0.0 for _ in range(int(input_dim) - len(x))]
    x = x[: int(input_dim)]
    return [float(v) for v in x]


def _zscore_stats(xs: List[List[float]], *, input_dim: int, eps: float) -> Tuple[List[float], List[float]]:
    if not xs:
        raise ValueError("cannot compute feature norm stats on empty xs")
    input_dim = int(input_dim)
    n = int(len(xs))
    mean = [0.0 for _ in range(int(input_dim))]
    for x in xs:
        if len(x) < int(input_dim):
            continue
        for j in range(int(input_dim)):
            mean[j] += float(x[j])
    for j in range(int(input_dim)):
        mean[j] /= float(max(1, int(n)))

    var = [0.0 for _ in range(int(input_dim))]
    for x in xs:
        if len(x) < int(input_dim):
            continue
        for j in range(int(input_dim)):
            d = float(x[j]) - float(mean[j])
            var[j] += d * d
    for j in range(int(input_dim)):
        var[j] /= float(max(1, int(n)))

    std = [(float(v) ** 0.5) for v in var]
    std = [float(s) if float(s) > float(eps) else 1.0 for s in std]
    return [float(m) for m in mean], [float(s) for s in std]


def _load_dataset_jsonl(
    path: Path,
    *,
    max_samples: int,
    input_dim: int,
    loss: str,
    label_key: str,
) -> Tuple[List[List[float]], List[float], List[float]]:
    xs: List[List[float]] = []
    train_ys: List[float] = []
    reward_ys: List[float] = []
    input_dim = int(input_dim)
    if int(input_dim) < 4:
        raise ValueError(f"model.input_dim must be >=4 (got {input_dim})")
    loss_norm = str(loss or "mse").strip().lower() or "mse"
    if loss_norm not in {"mse", "bce", "list_ce", "list_soft_ce"}:
        raise ValueError(f"train.loss must be one of: mse|bce|list_ce|list_soft_ce (got {loss!r})")
    label_key = str(label_key or "label").strip() or "label"
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        try:
            x = _extract_feature_vector(obj, input_dim=int(input_dim))
            reward = float(obj.get("reward", 0.0))
            if loss_norm in {"bce", "list_ce", "list_soft_ce"}:
                y = float(obj.get(label_key, 0.0))
            else:
                y = float(reward)
        except Exception:
            continue
        xs.append(x)
        train_ys.append(float(y))
        reward_ys.append(float(reward))
        n += 1
        if int(max_samples) > 0 and n >= int(max_samples):
            break
    return xs, train_ys, reward_ys


def _load_dataset_groups_jsonl(
    path: Path,
    *,
    max_groups: int,
    input_dim: int,
    label_key: str,
) -> List[Dict[str, Any]]:
    """
    Build listwise groups keyed by (case_id, budget_B, step_idx).

    Each group is a dict:
      - x: List[List[float]]      (n_candidates, input_dim)
      - target: int               (index of the single label==1 candidate)
      - rewards: List[float]      (n_candidates)
      - case_id/budget_B/step_idx (for audit)
    """
    label_key = str(label_key or "label").strip() or "label"

    by_key: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    order: List[Tuple[str, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue

        case_id = str(obj.get("case_id", "")).strip()
        if not case_id:
            continue
        try:
            budget_B = int(obj.get("budget_B", 0) or 0)
        except Exception:
            continue
        if budget_B <= 0:
            continue
        try:
            step_idx = int(obj.get("step_idx", obj.get("step", 0)) or 0)
        except Exception:
            continue
        if step_idx < 0:
            continue

        try:
            x = _extract_feature_vector(obj, input_dim=int(input_dim))
        except Exception:
            continue
        try:
            reward = float(obj.get("reward", 0.0))
        except Exception:
            reward = 0.0
        try:
            label = int(obj.get(label_key, 0) or 0)
        except Exception:
            label = 0
        label = 1 if int(label) == 1 else 0

        key = (case_id, int(budget_B), int(step_idx))
        g = by_key.get(key)
        if g is None:
            if int(max_groups) > 0 and len(order) >= int(max_groups):
                continue
            g = {"x": [], "labels": [], "rewards": [], "case_id": case_id, "budget_B": int(budget_B), "step_idx": int(step_idx)}
            by_key[key] = g
            order.append(key)
        g["x"].append(x)
        g["labels"].append(int(label))
        g["rewards"].append(float(reward))

    groups: List[Dict[str, Any]] = []
    for key in order:
        g = by_key.get(key)
        if not g:
            continue
        xs = list(g.get("x") or [])
        labels = list(g.get("labels") or [])
        rewards = list(g.get("rewards") or [])
        if len(xs) < 2:
            continue
        pos = [i for i, v in enumerate(labels) if int(v) == 1]
        if len(pos) != 1:
            continue
        if len(rewards) != len(xs):
            continue
        groups.append(
            {
                "x": xs,
                "target": int(pos[0]),
                "rewards": [float(v) for v in rewards],
                "case_id": str(g.get("case_id", "")),
                "budget_B": int(g.get("budget_B", 0) or 0),
                "step_idx": int(g.get("step_idx", 0) or 0),
            }
        )
    return groups


def _dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0") or 0)
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    return int(rank), int(world_size), int(local_rank)


def train_policy_torch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for train_policy_torch (install torch + CUDA if needed)")

    cfg = dict(cfg or {})
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    ddp_cfg = cfg.get("ddp", {})
    if not isinstance(ddp_cfg, dict):
        ddp_cfg = {}
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    run_id = str(cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).strip()
    if not run_id:
        raise ValueError("run_id missing")

    out_root = Path(str(cfg.get("out_dir", "outputs/train_policy"))).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_jsonl = _dataset_path(cfg)
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_jsonl}")

    seed = int(cfg.get("seed", 0))
    epochs = int(train_cfg.get("epochs", 3))
    batch_size = int(train_cfg.get("batch_size", 4096))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    ridge = float(train_cfg.get("ridge", 1e-6))
    max_samples = int(train_cfg.get("max_samples", 0))
    loss_name = str(train_cfg.get("loss", "mse")).strip().lower() or "mse"
    label_key = str(train_cfg.get("label_key", "label")).strip() or "label"
    pos_weight = float(train_cfg.get("pos_weight", 1.0))
    feature_norm = str(train_cfg.get("feature_norm", "none")).strip().lower() or "none"
    feature_norm_eps = float(train_cfg.get("feature_norm_eps", 1.0e-6))

    fp16 = bool(ddp_cfg.get("fp16", False))
    backend = str(ddp_cfg.get("backend", "nccl")).strip().lower() or "nccl"

    rank, world_size, local_rank = _dist_info()
    distributed = bool(int(world_size) > 1)

    device = torch.device("cuda", int(local_rank)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(local_rank))

    if distributed:
        import torch.distributed as dist  # type: ignore

        dist.init_process_group(backend=str(backend))

    random.seed(int(seed) + int(rank))
    torch.manual_seed(int(seed) + int(rank))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + int(rank))

    input_dim = int(model_cfg.get("input_dim", 4))
    xs, ys, reward_ys = _load_dataset_jsonl(
        dataset_jsonl,
        max_samples=int(max_samples),
        input_dim=int(input_dim),
        loss=str(loss_name),
        label_key=str(label_key),
    )
    if not xs:
        raise ValueError("dataset_jsonl is empty or unreadable")

    feature_norm = str(feature_norm or "none").strip().lower() or "none"
    if feature_norm not in {"none", "zscore"}:
        raise ValueError(f"train.feature_norm must be one of: none|zscore (got {feature_norm!r})")
    feat_mean: List[float] | None = None
    feat_std: List[float] | None = None
    if feature_norm == "zscore":
        feat_mean, feat_std = _zscore_stats(xs, input_dim=int(input_dim), eps=float(feature_norm_eps))

    def _norm_vec(x: List[float]) -> List[float]:
        if feat_mean is None or feat_std is None:
            return list(x)
        if len(x) < int(input_dim):
            x = list(x) + [0.0 for _ in range(int(input_dim) - len(x))]
        x = x[: int(input_dim)]
        return [(float(x[j]) - float(feat_mean[j])) / float(feat_std[j]) for j in range(int(input_dim))]

    from ..models.policy_mlp import PolicyMLP

    if PolicyMLP is None:
        raise RuntimeError("torch is available but PolicyMLP is not; unexpected environment state")

    hidden_dims_raw = model_cfg.get("hidden_dims", [64, 64])
    if not isinstance(hidden_dims_raw, list):
        hidden_dims_raw = [64, 64]
    hidden_dims = [int(x) for x in hidden_dims_raw if int(x) > 0] or [64, 64]
    activation = str(model_cfg.get("activation", "relu"))

    model = PolicyMLP(input_dim=int(input_dim), hidden_dims=list(hidden_dims), activation=str(activation)).to(device)
    if distributed:
        import torch.distributed as dist  # type: ignore
        from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore

        model = DDP(model, device_ids=[int(local_rank)] if device.type == "cuda" else None)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_norm = str(loss_name).strip().lower() or "mse"
    if loss_norm not in {"mse", "bce", "list_ce", "list_soft_ce"}:
        raise ValueError(f"train.loss must be one of: mse|bce|list_ce|list_soft_ce (got {loss_name!r})")

    if loss_norm in {"list_ce", "list_soft_ce"}:
        # Listwise CE over candidate tokens per (case_id, budget_B, step_idx).
        groups = _load_dataset_groups_jsonl(
            dataset_jsonl,
            max_groups=int(max_samples),
            input_dim=int(input_dim),
            label_key=str(label_key),
        )
        if not groups:
            raise ValueError("dataset_jsonl has no valid listwise groups (need step_idx + exactly-one positive label per step)")

        class _GroupDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
            def __init__(self, gs: List[Dict[str, Any]]):
                self._gs = list(gs)

            def __len__(self) -> int:
                return int(len(self._gs))

            def __getitem__(self, idx: int) -> Tuple[List[List[float]], int, List[float]]:
                g = self._gs[int(idx)]
                xs_g = g.get("x") or []
                tgt = int(g.get("target", 0))
                rewards_g = g.get("rewards") or []
                return list(xs_g), int(tgt), [float(v) for v in rewards_g]

        ds = _GroupDataset(groups)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=int(seed))  # type: ignore[attr-defined]
        else:
            sampler = None

        def _collate(
            batch: List[Tuple[List[List[float]], int, List[float]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            bsz = int(len(batch))
            max_len = 1
            for xs_g, _t, _r in batch:
                max_len = max(int(max_len), int(len(xs_g)))
            dim = int(input_dim)

            x_pad = torch.zeros((bsz, int(max_len), int(dim)), dtype=torch.float32)
            mask = torch.zeros((bsz, int(max_len)), dtype=torch.bool)
            target = torch.zeros((bsz,), dtype=torch.long)
            rewards_pad = torch.zeros((bsz, int(max_len)), dtype=torch.float32)
            for i, (xs_g, t, rewards_g) in enumerate(batch):
                n = int(len(xs_g))
                if n <= 0:
                    continue
                x_pad[i, :n, :] = torch.tensor([_norm_vec(list(v)) for v in xs_g], dtype=torch.float32)
                mask[i, :n] = True
                target[i] = int(t)
                if rewards_g:
                    rewards_pad[i, :n] = torch.tensor([float(v) for v in rewards_g[:n]], dtype=torch.float32)
            return x_pad, mask, target, rewards_pad

        loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            ds,
            batch_size=int(batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=False,
            collate_fn=_collate,
        )
    else:
        # Pointwise training (historical).
        x_tensor = torch.tensor([_norm_vec(list(x)) for x in xs], dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)  # type: ignore[attr-defined]
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=int(seed))  # type: ignore[attr-defined]
        else:
            sampler = None
        loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            ds,
            batch_size=int(batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=False,
        )
        if loss_norm == "bce":
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32, device=device))
        else:
            loss_fn = torch.nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(fp16 and device.type == "cuda"))
    autocast = torch.cuda.amp.autocast

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        if sampler is not None:
            sampler.set_epoch(int(epoch))

        total_loss = 0.0
        total_n = 0
        total_acc = 0.0
        if loss_norm in {"list_ce", "list_soft_ce"}:
            import torch.nn.functional as F  # type: ignore

            for xb, mask, target, rewards in loader:
                xb = xb.to(device)
                mask = mask.to(device)
                target = target.to(device)
                rewards = rewards.to(device)

                bsz, seq_len, dim = xb.shape
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=bool(scaler.is_enabled())):
                    logits = model(xb.view(int(bsz) * int(seq_len), int(dim))).view(int(bsz), int(seq_len))
                    logits = logits.masked_fill(~mask, -1.0e9)
                    if loss_norm == "list_ce":
                        loss = F.cross_entropy(logits, target)
                    else:
                        # Soft targets derived from per-candidate rewards. This keeps the hard argmax signal,
                        # but also provides gradient on near-optimal alternatives.
                        rp = torch.relu(rewards).to(torch.float32) * mask.to(torch.float32)
                        denom = rp.sum(dim=1, keepdim=True)

                        p = torch.zeros_like(rp)
                        nonzero = denom.view(-1) > 0.0
                        if bool(nonzero.any()):
                            p[nonzero] = rp[nonzero] / denom[nonzero]
                        if bool((~nonzero).any()):
                            p[~nonzero].scatter_(1, target[~nonzero].view(-1, 1), 1.0)

                        log_probs = F.log_softmax(logits, dim=1)
                        loss = -(p * log_probs).sum(dim=1).mean()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    pred_idx = torch.argmax(logits, dim=1)
                    acc = (pred_idx == target).to(torch.float32).sum().detach()
                total_loss += float(loss.detach().cpu().item()) * int(bsz)
                total_acc += float(acc.detach().cpu().item())
                total_n += int(bsz)
        else:
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=bool(scaler.is_enabled())):
                    pred = model(xb)
                    loss = loss_fn(pred.view(-1), yb.view(-1))  # type: ignore[name-defined]
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                total_loss += float(loss.detach().cpu().item()) * int(xb.shape[0])
                total_n += int(xb.shape[0])

        if distributed:
            import torch.distributed as dist  # type: ignore

            t = torch.tensor([float(total_loss), float(total_acc), float(total_n)], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss = float(t[0].detach().cpu().item())
            total_acc = float(t[1].detach().cpu().item())
            total_n = int(t[2].detach().cpu().item())

        mean_loss = float(total_loss) / float(max(1, int(total_n)))
        mean_acc = float(total_acc) / float(max(1, int(total_n)))
        if int(rank) == 0:
            row: Dict[str, Any] = {
                "timestamp_utc": _utc_now_iso(),
                "epoch": int(epoch),
                "loss": str(loss_name),
                "n_samples": int(total_n),
                "seed": int(seed),
                "world_size": int(world_size),
                "device": str(device),
            }
            if loss_norm == "list_ce":
                row["loss_list_ce"] = float(mean_loss)
                row["top1_acc"] = float(mean_acc)
            elif loss_norm == "list_soft_ce":
                row["loss_list_soft_ce"] = float(mean_loss)
                row["top1_acc"] = float(mean_acc)
            else:
                row["loss_mse"] = float(mean_loss)
                row["pos_weight"] = float(pos_weight)
            epoch_rows.append(row)

    if distributed:
        import torch.distributed as dist  # type: ignore

        dist.barrier()

    model_path = run_dir / "model.pt"
    checkpoint_path = run_dir / "checkpoint.json"

    if int(rank) == 0:
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()  # type: ignore[union-attr]
        else:
            state_dict = model.state_dict()
        torch.save(dict(state_dict), str(model_path))

        # Keep `weights` on reward for backward-compatible heuristic scoring, even when training uses BCE labels.
        weights, fit_meta = _fit_linear_weights(xs, reward_ys, ridge=float(ridge))
        fit_meta = {"fit": "dataset", "dataset_jsonl": str(dataset_jsonl), **dict(fit_meta)}

        checkpoint = {
            "stage": "P",
            "run_id": str(run_id),
            "weights": dict(weights),
            "fit_meta": dict(fit_meta),
            "model": {
                "type": "mlp",
                "input_dim": int(input_dim),
                "hidden_dims": list(hidden_dims),
                "activation": str(activation),
            },
            "model_path": str(model_path.name),
        }
        if feat_mean is not None and feat_std is not None:
            checkpoint["feature_norm"] = {
                "type": "zscore",
                "mean": list(feat_mean),
                "std": list(feat_std),
                "eps": float(feature_norm_eps),
            }

        (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in epoch_rows), encoding="utf-8")

    if distributed:
        import torch.distributed as dist  # type: ignore

        dist.barrier()
        dist.destroy_process_group()

    return {
        "status": "ok",
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage P: train policy (torch; supports torchrun/DDP).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    try:
        result = train_policy_torch(cfg)
    except Exception as exc:  # noqa: BLE001
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
