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


def _load_dataset_jsonl(path: Path, *, max_samples: int, input_dim: int) -> Tuple[List[List[float]], List[float]]:
    xs: List[List[float]] = []
    ys: List[float] = []
    input_dim = int(input_dim)
    if int(input_dim) < 4:
        raise ValueError(f"model.input_dim must be >=4 (got {input_dim})")
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        try:
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
            if len(x) < int(input_dim):
                x = x + [0.0 for _ in range(int(input_dim) - len(x))]
            x = x[: int(input_dim)]
            y = float(obj.get("reward", 0.0))
        except Exception:
            continue
        xs.append(x)
        ys.append(float(y))
        n += 1
        if int(max_samples) > 0 and n >= int(max_samples):
            break
    return xs, ys


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
    xs, ys = _load_dataset_jsonl(dataset_jsonl, max_samples=int(max_samples), input_dim=int(input_dim))
    if not xs:
        raise ValueError("dataset_jsonl is empty or unreadable")

    x_tensor = torch.tensor(xs, dtype=torch.float32)
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
    loss_fn = torch.nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(fp16 and device.type == "cuda"))
    autocast = torch.cuda.amp.autocast

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        if sampler is not None:
            sampler.set_epoch(int(epoch))

        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(scaler.is_enabled())):
                pred = model(xb)
                loss = loss_fn(pred.view(-1), yb.view(-1))
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

            t = torch.tensor([float(total_loss), float(total_n)], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss = float(t[0].detach().cpu().item())
            total_n = int(t[1].detach().cpu().item())

        mean_loss = float(total_loss) / float(max(1, int(total_n)))
        if int(rank) == 0:
            epoch_rows.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "epoch": int(epoch),
                    "loss_mse": float(mean_loss),
                    "n_samples": int(total_n),
                    "seed": int(seed),
                    "world_size": int(world_size),
                    "device": str(device),
                }
            )

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

        weights, fit_meta = _fit_linear_weights(xs, ys, ridge=float(ridge))
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
