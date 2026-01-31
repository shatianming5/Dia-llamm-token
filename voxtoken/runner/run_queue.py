from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class QueueJob:
    exp_id: str
    stage: str
    name: str
    command: str
    workdir: str


def _load_queue(path: Path) -> List[QueueJob]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("queue JSON must be a list of job objects")

    jobs: List[QueueJob] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"queue[{idx}] must be an object")
        exp_id = str(item.get("id", "")).strip()
        stage = str(item.get("stage", "")).strip()
        name = str(item.get("name", "")).strip()
        command = str(item.get("command", "")).strip()
        workdir = str(item.get("workdir", ".")).strip() or "."

        if not exp_id:
            raise ValueError(f"queue[{idx}].id missing")
        if not stage:
            raise ValueError(f"queue[{idx}].stage missing")
        if not command:
            raise ValueError(f"queue[{idx}].command missing")

        jobs.append(QueueJob(exp_id=exp_id, stage=stage, name=name, command=command, workdir=workdir))
    return jobs


def _result_basename(job: QueueJob) -> str:
    return f"{job.exp_id}-{job.stage}"


def _read_previous_result(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _run_job(
    job: QueueJob,
    *,
    repo_root: Path,
    results_dir: Path,
    logs_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base = _result_basename(job)
    result_path = results_dir / f"{base}.json"
    log_path = logs_dir / f"{base}.log"

    prev = _read_previous_result(result_path)
    if prev is not None and not overwrite:
        prev_exit = prev.get("exit_code", None)
        try:
            prev_exit = int(prev_exit)
        except Exception:
            prev_exit = None
        if prev_exit == 0:
            return {"status": "skipped", "reason": "already_ok", "result_path": str(result_path), "log_path": str(prev.get("log_path", log_path))}

    if dry_run:
        return {"status": "dry_run", "id": job.exp_id, "stage": job.stage, "name": job.name, "command": job.command, "workdir": job.workdir}

    start = _utc_now_iso()
    t0 = time.perf_counter()

    # Jobs in this repo use workdir="." to mean "repo root", not ".rd_queue/".
    cwd = (repo_root / job.workdir).resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    header = {
        "id": job.exp_id,
        "stage": job.stage,
        "name": job.name,
        "command": job.command,
        "workdir": str(cwd),
        "start_time_utc": start,
    }

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        f.write("\n")
        f.flush()
        proc = subprocess.Popen(
            job.command,
            shell=True,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
        exit_code = int(proc.wait())

    end = _utc_now_iso()
    elapsed_sec = float(time.perf_counter() - t0)

    payload = {
        "id": job.exp_id,
        "stage": job.stage,
        "name": job.name,
        "command": job.command,
        "workdir": str(cwd),
        "start_time_utc": start,
        "end_time_utc": end,
        "elapsed_sec": float(elapsed_sec),
        "exit_code": int(exit_code),
        "ok": bool(exit_code == 0),
        "log_path": str(log_path),
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible .rd_queue queue JSON file (sequential, logs+results).")
    parser.add_argument("--queue", required=True, help="Path to a queue JSON file (list of {id,stage,name,command,workdir}).")
    parser.add_argument("--results-dir", default=".rd_queue/results", help="Results directory (JSON per job).")
    parser.add_argument("--logs-dir", default=".rd_queue/logs", help="Logs directory (stdout/stderr per job).")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if a previous ok result exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the jobs that would run and exit.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Run all jobs even if one fails.")
    parser.add_argument("--only-id", action="append", default=[], help="Only run jobs with this experiment id (repeatable).")
    parser.add_argument("--only-stage", action="append", default=[], help="Only run jobs with this stage (repeatable).")
    parser.add_argument("--write-queue-json", default=".rd_queue/queue.json", help="Write the used queue content to this path.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / "voxtoken").exists():
        print("[ERR] run_queue must be executed from the repo root (missing ./voxtoken)", file=sys.stderr)
        sys.exit(2)

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"[ERR] queue not found: {queue_path}", file=sys.stderr)
        sys.exit(2)

    try:
        jobs = _load_queue(queue_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] failed to parse queue: {exc}", file=sys.stderr)
        raise

    only_ids = {str(x).strip() for x in (args.only_id or []) if str(x).strip()}
    only_stages = {str(x).strip() for x in (args.only_stage or []) if str(x).strip()}
    if only_ids:
        jobs = [j for j in jobs if j.exp_id in only_ids]
    if only_stages:
        jobs = [j for j in jobs if j.stage in only_stages]

    if args.dry_run:
        for j in jobs:
            print(json.dumps({"id": j.exp_id, "stage": j.stage, "name": j.name, "workdir": j.workdir, "command": j.command}, ensure_ascii=False))
        return

    queue_out = Path(str(args.write_queue_json)).resolve()
    queue_out.parent.mkdir(parents=True, exist_ok=True)
    queue_out.write_text(queue_path.read_text(encoding="utf-8"), encoding="utf-8")

    results_dir = Path(str(args.results_dir)).resolve()
    logs_dir = Path(str(args.logs_dir)).resolve()

    failures: List[Dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        print(f"[rdq] ({i}/{len(jobs)}) {job.exp_id} {job.stage} :: {job.name}".strip())
        payload = _run_job(
            job,
            repo_root=repo_root,
            results_dir=results_dir,
            logs_dir=logs_dir,
            overwrite=bool(args.overwrite),
            dry_run=False,
        )

        if payload.get("status") == "skipped":
            continue
        if not bool(payload.get("ok", False)):
            failures.append(payload)
            if not bool(args.continue_on_fail):
                break

    if failures:
        print(f"[rdq] FAIL {len(failures)}/{len(jobs)} jobs failed.", file=sys.stderr)
        for f in failures:
            print(f"[rdq] - {f.get('id')} {f.get('stage')} exit_code={f.get('exit_code')} log={f.get('log_path')}", file=sys.stderr)
        sys.exit(1)

    print(f"[rdq] DONE all jobs finished. queue={queue_path} results={results_dir} logs={logs_dir}")


if __name__ == "__main__":
    main()

