# Mohu

本文件用于记录两类阻塞项，并驱动“实现→验证→再修复”的闭环迭代。

收敛条件：当 **Not Implemented** 与 **Ambiguities** 两部分都为空时，进入实验定义与运行阶段（`docs/experiment.md`）。

注意：本清单以 `docs/plan.md` 顶部的 `C####`/`P####`（milestone-scoped commitments）为准；proposal-level 的长线设计不直接作为阻塞项。

2026-01-30 更新：用户要求 **proposal-level 的 M2/M3/M4 也必须补齐**，因此这些内容会被提升为 `docs/plan.md` 顶部的 `C####`/`P####` 并纳入本清单的阻塞项。

Snapshot (2026-01-31):

- Git: 74d2ce1
- Last rdq run: 2026-01-31T07:00:31.998209+00:00 (see `.rd_queue/results/`, queue: `.rd_queue/queue_e0907_0910.json`)

## 1. Not Implemented

> 规则：逐条实现；每条必须“实现→验证通过→才能进入下一条”；验证失败必须回到该条继续修复。

（空表示当前无阻塞未实现项）

## 2. Ambiguities

> 规则：逐条澄清为可执行规格（指标/协议/保存物/脚本/验收标准）；必要时修订 `docs/plan.md`（保留 before/after）；并完成实现与验证。

（空表示当前无阻塞模糊点）

## Resolved (optional)

- [x] M0039: Build a CT-RATE TS grounding manifest (volume+mask -> GT boxes) and a validator.
  - Ref: P0035, C0036, E0907
  - Verification: `python -m voxtoken.runner.reproduce --exp E0907`
  - Resolution: Implemented `voxtoken/data/ct_rate_ts_manifest.py` + `voxtoken/runner/validate_ct_rate_ts_manifest.py` and added `totalseg` config; verified via E0907 (`.rd_queue/results/E0907-*.json`, queue `.rd_queue/queue_e0907_0910.json`).
- [x] M0040: Build a multi-case policy dataset from CT-RATE TS GT boxes and add a policy-checkpoint validator.
  - Ref: P0036, C0037, E0908
  - Verification: `python -m voxtoken.runner.reproduce --exp E0908`
  - Resolution: Implemented `build_policy_dataset_multi` + `validate_policy_checkpoint`; updated `train_policy` checkpoint to include `fit_meta` (dataset vs synth); verified via E0908 (`.rd_queue/results/E0908-*.json`).
- [x] M0041: Add a CT-RATE TS grounding benchmark runner and a validator for aggregate outputs.
  - Ref: P0037, C0038, E0909
  - Verification: `python -m voxtoken.runner.reproduce --exp E0909`
  - Resolution: Implemented `ct_rate_grounding_benchmark` + `validate_grounding_benchmark`; verified via E0909 (`.rd_queue/results/E0909-*.json`).
- [x] M0042: Add a paper export script (tables/figures) + validator for artifacts.
  - Ref: P0038, C0039, E0910
  - Verification: `python -m voxtoken.runner.reproduce --exp E0910`
  - Resolution: Implemented `paper_export` + `validate_paper_artifacts` (Table1/2 + Fig2/3); verified via E0910 (`.rd_queue/results/E0910-*.json`).

- [x] M0034: Add tokenizer perplexity metrics + recon_error separation gate (Stage T acceptance checks).
  - Ref: P0032, C0033, E0904
  - Verification: `python -m voxtoken.runner.reproduce --exp E0904`
  - Resolution: Added `perplexity/codebook_used` to tokenizer train metrics (`metrics.json`) + added validators `validate_tokenizer_train_metrics` and `validate_recon_error_separation`; verified via E0904.
- [x] M0035: Make evidence graph sidecar traceable to token Ω (Stage E acceptance checks).
  - Ref: P0033, C0034, E0905
  - Verification: `python -m voxtoken.runner.reproduce --exp E0905`
  - Resolution: Updated `evidence_graph.json` to include tokens + `token_omega_index`, and strengthened `validate_run --require-evidence-graph-json` to enforce traceability; verified via E0905.
- [x] M0036: Add verifier stability gate (A5 acceptance check).
  - Ref: P0034, C0035, E0906
  - Verification: `python -m voxtoken.runner.reproduce --exp E0906`
  - Resolution: Added `validate_verifier_stability` and localized gate-injected issues; verified determinism on the no-citation run via E0906.

- [x] M0037: Resolve tokenizer layout ambiguity in `docs/plan.md` repo skeleton (tokenizer/ package vs `tokenizer.py`) and keep one canonical structure.
  - Ref: P0004, P0007
  - Context: `docs/plan.md` previously sketched both `voxtoken/models/tokenizer.py` and a `models/tokenizer/` subpackage, which is ambiguous; keeping both in the repo also breaks imports (`voxtoken.models.tokenizer` cannot be both module and package).
  - Acceptance: `docs/plan.md` skeleton tree matches the canonical layout (`voxtoken/models/tokenizer.py`); no conflicting `voxtoken/models/tokenizer/` directory exists; imports and inference still run.
  - Verification: `python -c "from voxtoken.models.tokenizer import Tokenizer3D; print('ok')" && python -m voxtoken.runner.infer_refine --out artifacts/e0907_tokenizer_layout --budget 16 --config voxtoken/configs/inference.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0907_tokenizer_layout/run.json`
  - Resolution: Updated `docs/plan.md` repo skeleton tree to file-based tokenizer layout and removed the conflicting `voxtoken/models/tokenizer/` directory; verified via the command above.

- [x] M0038: Fix `docs/plan.md` section 10 repo skeleton code block indentation and path consistency.
  - Ref: P0003
  - Context: The section-10 tree snippet had tab-indented fences and broken indentation (e.g., `constrained.py` de-indented), making it non-copyable and misleading.
  - Acceptance: The section-10 code block is a normal ```text fence with consistent indentation, and every referenced file path exists in the repo.
  - Verification: `python -c 'from pathlib import Path; import re; t=Path("docs/plan.md").read_text(encoding="utf-8"); assert "\\t```" not in t; assert re.search(r"## 10\\. 可直接照着写的“最小 repo 骨架”[\\s\\S]*?```text\\n", t); req=[\"voxtoken/data/adapters/ct_rate.py\",\"voxtoken/data/adapters/radgenome.py\",\"voxtoken/data/preprocess.py\",\"voxtoken/core/schemas.py\",\"voxtoken/core/geometry.py\",\"voxtoken/models/encoder3d.py\",\"voxtoken/models/vq.py\",\"voxtoken/models/tokenizer.py\",\"voxtoken/models/evidence_head.py\",\"voxtoken/models/policy.py\",\"voxtoken/generation/planner.py\",\"voxtoken/generation/realizer.py\",\"voxtoken/generation/constrained.py\",\"voxtoken/verify/verifier.py\",\"voxtoken/verify/slot_extract.py\",\"voxtoken/runner/train_tokenizer.py\",\"voxtoken/runner/train_evidence.py\",\"voxtoken/runner/train_policy.py\",\"voxtoken/runner/infer_refine.py\",\"voxtoken/eval/metrics.py\",\"voxtoken/eval/pareto.py\",\"voxtoken/eval/counterfactuals.py\",\"voxtoken/eval/visualize_grounding.py\"]; missing=[p for p in req if not Path(p).exists()]; assert not missing, missing; print("ok")'`
  - Resolution: Rewrote the section-10 code block to remove tab-indented fences and fix indentation so the skeleton is copy-pastable and accurate.

- [x] M0001: Make E0200 (heuristic split refinement) runnable: hierarchical token pyramid + real split selection + trace output.
  - Ref: P0004, C0005, E0200
  - Context: `Tokenizer3D` currently only builds level-0 tokens and `infer_refine` refinement ignores `split_ids`; E0200 requires a non-empty split trace under budget.
  - Acceptance: `python -m voxtoken.runner.infer_refine --out artifacts/e0200 --budget 16 --config voxtoken/configs/inference_heuristic.yaml` writes `run.json` with non-empty `trace` and at least one non-empty `split_token_ids`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0200 --budget 16 --config voxtoken/configs/inference_heuristic.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0200/run.json --require-trace --require-split`
  - Resolution: Implemented hierarchical `TokenPyramid` + split replacement in `infer_refine`; added `voxtoken/configs/inference_heuristic.yaml` and `voxtoken/runner/validate_run.py` (verified OK, artifacts at `artifacts/e0200/`).
  - Notes: Also update `docs/experiment.md` to check off E0200 only after verification passes.
- [x] M0002: Make E0300 runnable: deterministic policy training checkpoint + inference loading path.
  - Ref: P0005, C0006, E0300
  - Context: `train_policy` is placeholder and produces timestamped output; `SplitPolicy` does not load checkpoints; E0300 needs a reproducible checkpoint path.
  - Acceptance: `python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy_e0300.yaml` writes `outputs/train_policy/E0300/checkpoint.json` (weights included), and `infer_refine` can load it via `voxtoken/configs/inference_learned_policy.yaml`.
  - Verification: `python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy_e0300.yaml && python -m voxtoken.runner.infer_refine --out artifacts/e0300 --budget 16 --config voxtoken/configs/inference_learned_policy.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0300/run.json --require-trace`
  - Resolution: Implemented deterministic `train_policy` (synthetic linear fit) + checkpoint loading in `SplitPolicy`; inference now records `meta.policy` (verified OK, checkpoint at `outputs/train_policy/E0300/checkpoint.json`).
- [x] M0003: Make E0400/E0401 runnable: ablation toggles (no-citation / no-constrained+overclaim) + verifier rules.
  - Ref: P0006, C0007, C0008, E0400, E0401
  - Context: Realizer always emits citations and verifier does not flag overclaim; need toggles and rules so ablations change `unsupported_rate` / `verifier_score` measurably.
  - Acceptance: E0400 produces missing citations -> `unsupported_rate > 0`; E0401 produces overclaim -> `Issue(type=overclaim)` and lower `verifier_score` vs constrained baseline.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0400 --budget 16 --config voxtoken/configs/inference_no_citation.yaml && python -m voxtoken.runner.unified_eval --in artifacts/e0400/run.json --out artifacts/e0400_eval && python -m voxtoken.runner.validate_metrics --in artifacts/e0400_eval/metrics.json --require-unsupported-gt 0 && python -m voxtoken.runner.infer_refine --out artifacts/e0401 --budget 16 --config voxtoken/configs/inference_no_constrained.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0401/run.json --require-overclaim`
  - Resolution: Added `emit_citations` + overclaim toggles in Realizer, gates in `infer_refine`, overclaim rule in verifier, and `validate_metrics` runner (verified OK; artifacts at `artifacts/e0400*` and `artifacts/e0401*`).
- [x] M0004: Make E0500 runnable: deterministic tokenizer training checkpoint + inference loading (token codes enabled).
  - Ref: P0007, C0009, E0500
  - Context: `train_tokenizer` is placeholder with timestamped output; `Tokenizer3D` does not load checkpoints or assign `Token.code` for downstream modules.
  - Acceptance: `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer_e0500.yaml` writes `outputs/train_tokenizer/E0500/checkpoint.json` and `infer_refine` can load it (records in `run.json.meta.tokenizer`).
  - Verification: `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer_e0500.yaml && python -m voxtoken.runner.infer_refine --out artifacts/e0500 --budget 16 --config voxtoken/configs/inference_tokenizer_ckpt.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0500/run.json --require-tokenizer-codes`
  - Resolution: Implemented stdlib-only tokenizer "training" (synthetic codebook), checkpoint loading + per-token code assignment, and export in `run.json.meta` (verified OK; artifacts at `outputs/train_tokenizer/E0500/` and `artifacts/e0500/`).
- [x] M0005: Make E0600 runnable: deterministic evidence head training checkpoint + inference loading (>=2 finding types).
  - Ref: P0008, C0010, E0600
  - Context: `train_evidence` is placeholder with timestamped output; `EvidenceHead` always emits a constant finding type and ignores token codes.
  - Acceptance: `python -m voxtoken.runner.train_evidence --config voxtoken/configs/train_evidence_e0600.yaml` writes `outputs/train_evidence/E0600/checkpoint.json`; inference with that checkpoint emits >=2 distinct finding types in report.
  - Verification: `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer_e0500.yaml && python -m voxtoken.runner.train_evidence --config voxtoken/configs/train_evidence_e0600.yaml && python -m voxtoken.runner.infer_refine --out artifacts/e0600 --budget 16 --config voxtoken/configs/inference_evidence_ckpt.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0600/run.json --require-tokenizer-codes --require-finding-types-ge 2`
  - Resolution: Implemented evidence checkpoint training + loading; evidence now maps token codes to `finding_type` producing multi-type plans/reports (verified OK; artifacts at `outputs/train_evidence/E0600/` and `artifacts/e0600/`).
- [x] M0006: Implement `voxtoken.runner.reproduce` to re-run `E####` from `docs/experiment.md` (execute `1GPU script`).
  - Ref: P0011, C0013, E0800
  - Context: `voxtoken/runner/reproduce.py` was a placeholder; need an auditable way to re-run a specific experiment from the ledger.
  - Acceptance: `python -m voxtoken.runner.reproduce --exp E0200 --ledger docs/experiment.md` exits 0 (runs the resolved command chain).
  - Verification: `python -m voxtoken.runner.reproduce --exp E0200 --ledger docs/experiment.md`
  - Resolution: Implemented ledger-table parsing + command execution (with recursion guard); verified via E0800 (`.rd_queue/results/E0800-*.json`).
- [x] M0007: Add counterfactual evaluation runner + validator (citation removal sensitivity).
  - Ref: P0012, C0014, E0801
  - Context: `voxtoken/eval/counterfactuals.py` existed but there was no CLI runner and no validation gate for its outputs.
  - Acceptance: E0801 produces `counterfactuals.json` where `unsupported_rate.remove_citations > unsupported_rate.base`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0801 --budget 16 --config voxtoken/configs/inference.yaml && python -m voxtoken.runner.counterfactual_eval --in artifacts/e0801/run.json --out artifacts/e0801_cf && python -m voxtoken.runner.validate_counterfactuals --in artifacts/e0801_cf/counterfactuals.json --require-remove-citations-gt-base`
  - Resolution: Added `voxtoken.runner.counterfactual_eval` + `validate_counterfactuals`; verified via E0801 (`.rd_queue/results/E0801-*.json`).
- [x] M0008: Add ingest+preprocess configs and a manifest validator for deterministic splits.
  - Ref: P0013, C0015, E0802
  - Context: data modules were runnable but lacked a docs-spec experiment with stable configs and acceptance checks.
  - Acceptance: E0802 writes `artifacts/data_proc_e0802/manifest.jsonl` with `n>=10` and a valid `split` per row.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_e0802.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_e0802.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/data_proc_e0802/manifest.jsonl --require-n-ge 10 --valid-splits train val test`
  - Resolution: Added configs + `validate_manifest`; verified via E0802 (`.rd_queue/results/E0802-*.json`).
- [x] M0009: Make ingest support CT-RATE (`/data/ct_rate` / `/data/CT-RATE`) and validate report/volume paths.
  - Ref: P0014, C0016, E0803
  - Context: need to test the data pipeline on a real local dataset rather than synthetic JSONL.
  - Acceptance: E0803 writes a processed `manifest.jsonl` with `n>=20`, valid `split`, and existing `report_path` + `volume_path` per row.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_e0803.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_e0803.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_proc_e0803/manifest.jsonl --require-n-ge 20 --valid-splits train val test --require-report-path-exists --require-volume-path-exists --require-nonempty-volume-paths-ge 20`
  - Resolution: Implemented `source: ct_rate` ingest mode (CSV parsing + optional volume search) and strengthened manifest validation; verified via E0803 (`.rd_queue/results/E0803-*.json`).
- [x] M0010: Add CT-RATE train split ingest/preprocess configs and validate report-path existence.
  - Ref: P0015, C0017, E0804
  - Context: need to use CT-RATE train reports even when volume files are not available locally.
  - Acceptance: E0804 writes `artifacts/ct_rate_train_proc_e0804/manifest.jsonl` with `n>=200`, valid `split`, report paths exist, and `case_id` starts with `train_`.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_train_e0804.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_train_e0804.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_train_proc_e0804/manifest.jsonl --require-n-ge 200 --valid-splits train val test --require-report-path-exists --require-case-id-prefix train_`
  - Resolution: Added configs + `validate_manifest` support for `--require-case-id-prefix`; verified via E0804 (`.rd_queue/results/E0804-*.json`).
- [x] M0011: Extend `infer_refine` to support `--manifest` + `--case-id` and load small NIfTI volumes when present.
  - Ref: P0016, C0018, E0805
  - Context: want to run inference on a real dataset case selected from a manifest rather than a synthetic-only dummy volume.
  - Acceptance: E0805 writes `run.json.meta.input` with `volume_loader=nifti` and passes `validate_run`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0805 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --case-id valid_1_a_1 && python -m voxtoken.runner.validate_run --in artifacts/e0805/run.json --require-meta-input --require-meta-input-case-id valid_1_a_1 --require-meta-input-volume-loader nifti --require-meta-input-volume-path-exists`
  - Resolution: Implemented manifest selection + small NIfTI downsampling loader; updated `validate_run` with meta.input checks; verified via E0805 (`.rd_queue/results/E0805-*.json`).
- [x] M0012: Add a runnable experiment that runs `infer_refine` on CT-RATE train reports manifest (dummy volume fallback).
  - Ref: P0017, C0019, E0806
  - Context: connect "train reports ingest" with "manifest-driven inference" even when volumes are not available.
  - Acceptance: E0806 writes `run.json` with `meta.input.volume_loader=dummy` and `meta.input.report_path` exists.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0806 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_train_proc_e0804/manifest.jsonl --case-id train_1_a_1 && python -m voxtoken.runner.validate_run --in artifacts/e0806/run.json --require-meta-input --require-meta-input-case-id train_1_a_1 --require-meta-input-volume-loader dummy --require-meta-input-report-path-exists`
  - Resolution: Added `validate_run` support for `--require-meta-input-report-path-exists`; verified via E0806 (`.rd_queue/results/E0806-*.json`).
- [x] M0013: Implement batch inference+eval over CT-RATE validation cases (E0810).
  - Ref: P0018, C0020, E0810
  - Context: we can run a single case from a manifest (E0805/E0806), but need a batch runner to process multiple cases and aggregate metrics.
  - Acceptance: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --out artifacts/e0810 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 5 --require-volume-loader nifti` exits 0 and writes `artifacts/e0810/metrics.jsonl` with `n>=5`.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --out artifacts/e0810 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 5 --require-volume-loader nifti && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0810/metrics.jsonl --require-n-ge 5`
  - Resolution: Added `batch_infer_eval` + `validate_metrics_jsonl`; verified via E0810 (`.rd_queue/results/E0810-*.json`).
- [x] M0014: Run the batch runner scale test (E0811) and record results.
  - Ref: P0019, C0021, E0811
  - Context: E0810 validates a small batch; E0811 validates `--max-cases 0` processes all selected rows from the manifest.
  - Acceptance: `python -m voxtoken.runner.batch_infer_eval ... --max-cases 0` writes `artifacts/e0811/metrics.jsonl` with `n>=20` and exits 0.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --out artifacts/e0811 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 0 --require-volume-loader nifti && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0811/metrics.jsonl --require-n-ge 20`
  - Resolution: Verified via E0811 (`.rd_queue/results/E0811-*.json`).
- [x] M0015: Make CT-RATE ingest join predicted labels into the manifest (E0820).
  - Ref: P0020, C0022, E0820
  - Acceptance: processed manifest contains `labels_pos` and at least one row has non-empty labels.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_labeled_e0820.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_labeled_e0820.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --require-n-ge 20 --valid-splits train val test --require-report-path-exists --require-volume-path-exists --require-nonempty-volume-paths-ge 20 --require-case-id-prefix valid_ --require-labels-pos --require-nonempty-labels-pos-ge 1`
  - Resolution: Implemented predicted-label join in CT-RATE ingest and manifest validation; verified via E0820 (`.rd_queue/results/E0820-*.json`).
- [x] M0016: Add CT-RATE per-case label evaluation runner + validator (E0821).
  - Ref: P0021, C0023, E0821
  - Acceptance: `label_metrics.json` exists and validates with `n_gold_pos > 0`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0821 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --case-id valid_1_a_1 && python -m voxtoken.runner.ct_rate_label_eval --run artifacts/e0821/run.json --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0821_label --case-id valid_1_a_1 && python -m voxtoken.runner.validate_label_metrics --in artifacts/e0821_label/label_metrics.json --require-n-gold-pos-gt 0`
  - Resolution: Added `ct_rate_label_eval` + `validate_label_metrics`; verified via E0821 (`.rd_queue/results/E0821-*.json`).
- [x] M0017: Add CT-RATE batch label evaluation runner + validator (E0822).
  - Ref: P0022, C0024, E0822
  - Acceptance: `label_metrics.jsonl` exists and validates with `n>=20`.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0822 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 0 --require-volume-loader nifti && python -m voxtoken.runner.batch_label_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --runs artifacts/e0822/runs --out artifacts/e0822_label && python -m voxtoken.runner.validate_label_metrics_jsonl --in artifacts/e0822_label/label_metrics.jsonl --require-n-ge 20`
  - Resolution: Added `batch_label_eval` + `validate_label_metrics_jsonl`; verified via E0822 (`.rd_queue/results/E0822-*.json`).
- [x] M0018: Make E0830 runnable: generate a CT-RATE report directly from `labels_pos` and validate `f1>=0.99`.
  - Ref: P0023, C0025, E0830
  - Context: `infer_refine` is not label-conditioned; add a simple, auditable baseline that emits finding types derived from `labels_pos` so label evaluation can gate on F1.
  - Acceptance: `ct_rate_report_from_labels` writes a `run.json` with per-line citations and finding types derived from manifest `labels_pos`; `validate_label_metrics` supports F1 thresholds; E0830 passes.
  - Verification: `python -m voxtoken.runner.ct_rate_report_from_labels --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --case-id valid_1_a_1 --out artifacts/e0830 && python -m voxtoken.runner.ct_rate_label_eval --run artifacts/e0830/run.json --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0830_label --case-id valid_1_a_1 && python -m voxtoken.runner.validate_label_metrics --in artifacts/e0830_label/label_metrics.json --require-n-gold-pos-gt 0 --require-f1-ge 0.99`
  - Resolution: Implemented `voxtoken/runner/ct_rate_report_from_labels.py` and extended `validate_label_metrics` with `--require-f1-ge/--require-f1-lt`; verified via E0830 (`.rd_queue/results/E0830-*.json`).
- [x] M0019: Make E0831 runnable: batch-generate reports from `labels_pos` and gate `label_metrics.jsonl` by `f1>=0.99`.
  - Ref: P0024, C0026, E0831
  - Context: single-case label-conditioned generation (E0830) is proven, but we also need a batch runner that produces `runs/<case_id>/run.json` and a JSONL validator that can enforce per-row F1 thresholds.
  - Acceptance: `batch_ct_rate_report_from_labels` writes `runs/<case_id>/run.json` for >=10 labeled cases; `validate_label_metrics_jsonl` supports `--require-f1-ge`; E0831 passes.
  - Verification: `python -m voxtoken.runner.batch_ct_rate_report_from_labels --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0831 --max-cases 0 && python -m voxtoken.runner.batch_label_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --runs artifacts/e0831/runs --out artifacts/e0831_label && python -m voxtoken.runner.validate_label_metrics_jsonl --in artifacts/e0831_label/label_metrics.jsonl --require-n-ge 10 --require-f1-ge 0.99`
  - Resolution: Implemented `voxtoken/runner/batch_ct_rate_report_from_labels.py` and extended `validate_label_metrics_jsonl` with `--require-f1-ge/--require-f1-lt`; verified via E0831 (`.rd_queue/results/E0831-*.json`).
- [x] M0020: Add paper-facing sidecar artifacts for `infer_refine` output dir (`final_report.txt`, `evidence_graph.json`, `trace.jsonl`).
  - Ref: P0025, C0027, E0832
  - Context: Plan A0.3 expects a 3-file export bundle in addition to `run.json` for downstream tooling and paper figures.
  - Acceptance: `infer_refine` writes the sidecar files next to `run.json`; `validate_run` can gate their presence/format; E0832 passes.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0832 --budget 16 --config voxtoken/configs/inference.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0832/run.json --require-final-report-txt --require-evidence-graph-json --require-trace-jsonl`
  - Resolution: Added sidecar writing in `infer_refine` and added sidecar checks to `validate_run`; verified via E0832 (`.rd_queue/results/E0832-*.json`).
- [x] M0021: Make E0833 runnable: batch-run 50 CT-RATE cases and ensure every case writes the full artifact bundle (run + sidecars).
  - Ref: P0026, C0028, E0833
  - Context: Plan M0 asks to run a larger batch and ensure the report/evidence/trace artifacts are consistently produced for downstream tooling.
  - Acceptance: `batch_infer_eval` writes `runs/<case_id>/{run.json,final_report.txt,evidence_graph.json,trace.jsonl}` for >=50 cases and the aggregated `metrics.jsonl` has `n>=50`.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_train_proc_e0804/manifest.jsonl --out artifacts/e0833 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 50 && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0833/metrics.jsonl --require-n-ge 50`
  - Resolution: Updated `batch_infer_eval` to emit the sidecar artifact bundle per case; verified via E0833 (`.rd_queue/results/E0833-*.json`).
- [x] M0022: Add paper-facing grounding columns to `unified_eval` output (`ground_hit@0.0`, `ground_hit@0.1`).
  - Ref: docs/plan.md §6.2, §7.1–7.2
  - Context: `docs/plan.md` table templates require `ground_hit@0.0` / `ground_hit@0.1`, but `metrics.json(l)` did not include them.
  - Acceptance: `voxtoken.runner.unified_eval` writes both keys to `metrics.json` and `metrics.jsonl`; `docs/results_contract.md` example is updated.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16 && python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval && python -m voxtoken.runner.validate_metrics --in artifacts/infer_eval/metrics.json --require-slot-f1-ge 0.99`
  - Resolution: Implemented a box-IoU proxy based on cited-token boxes vs supporting-token boxes; updated results contract example.
- [x] M0023: Make counterfactual eval runnable for paper templates: `permute_omega` + `counterfactual.csv`.
  - Ref: docs/plan.md §8, §7.4; C0014
  - Context: `docs/plan.md` expects counterfactual toggles (incl. `permute_omega`) and a `counterfactual.csv` for Fig 3, but the repo only emitted a minimal JSON.
  - Acceptance: `counterfactual_eval` / `eval/counterfactuals.py` emits `counterfactuals.json(l)` plus `counterfactual.csv` with columns `{cf_type, slot_f1_micro, ground_hit@0.1, unsupported_sent_pct}`.
  - Verification: `python -m voxtoken.runner.counterfactual_eval --in artifacts/infer/run.json --out artifacts/infer_cf && python -m voxtoken.runner.validate_counterfactuals --in artifacts/infer_cf/counterfactuals.json --require-remove-citations-gt-base`
  - Resolution: Added `permute_omega` (breaks grounding proxy) + scenario rows; wrote CSV sidecar via runner + paper-facing wrapper.
- [x] M0024: Allow `infer_refine` to take `budget_B` from `configs/inference.yaml` when `--budget` is omitted.
  - Ref: docs/plan.md §4.4
  - Context: Paper-facing `configs/inference.yaml` includes `refine.budget_B`, but `infer_refine` only read the CLI flag.
  - Acceptance: `python -m voxtoken.runner.infer_refine --config configs/inference.yaml` uses `refine.budget_B` by default.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/cfg_infer --config configs/inference.yaml && cat artifacts/cfg_infer/summary.json`
  - Resolution: Changed `--budget` default to None and wired fallback to `cfg.refine.budget_B` (else 16).
- [x] M0025: Add `voxtoken/data/splits/` module to match the repo skeleton and keep split logic reusable.
  - Ref: docs/plan.md §A8
  - Context: The `docs/plan.md` skeleton lists `voxtoken/data/splits/`, but the repo had split logic inlined in `preprocess.py`.
  - Acceptance: `voxtoken/data/splits/assign_split` exists and `preprocess` uses it.
  - Verification: `python -c "from voxtoken.data.splits import assign_split; print(assign_split('case_1', seed=0, train_p=0.8, val_p=0.1))"`
  - Resolution: Added `voxtoken/data/splits/{__init__.py,deterministic.py}` and refactored `voxtoken/data/preprocess.py` to import it.
- [x] M0026: Add `voxtoken/artifacts/` compatibility marker directory.
  - Ref: docs/plan.md §A8
  - Context: The skeleton tree includes `voxtoken/artifacts/`, while actual outputs live in repo-root `artifacts/`.
  - Acceptance: `voxtoken/artifacts/README.md` documents the alias and prevents skeleton drift.
  - Verification: `test -f voxtoken/artifacts/README.md`
  - Resolution: Added `voxtoken/artifacts/README.md` explaining the root `artifacts/` convention.
- [x] M0027: Make paper-facing train config templates runnable (`train.save_dir` -> runner `out_dir`).
  - Ref: docs/plan.md §4.1–4.3
  - Context: The root `configs/train_{tokenizer,evidence,policy}.yaml` templates specify `train.save_dir`, but the repo-skeleton runners defaulted to `outputs/*` unless `out_dir` was provided.
  - Acceptance: `python -m voxtoken.runner.train_* --config configs/train_*.yaml` writes outputs under the template `train.save_dir`.
  - Verification: `python -m voxtoken.runner.train_tokenizer --config configs/train_tokenizer.yaml && python -m voxtoken.runner.train_evidence --config configs/train_evidence.yaml && python -m voxtoken.runner.train_policy --config configs/train_policy.yaml`
  - Resolution: Added a small config normalization in each `train_*` runner to treat `train.save_dir` as `out_dir` when `out_dir` is not set.
- [x] M0028: Normalize `configs/inference.yaml` paper template keys to runner keys (`generation.require_citation`, `verifier.{alpha,beta,gamma,delta}`, `refine.tau`).
  - Ref: docs/plan.md §4.4
  - Context: The paper-facing inference template uses different key names than the repo-skeleton runner (`gates/realizer/verifier.weights`).
  - Acceptance: Running with the template (or a small variant) affects citations/weights without needing internal `voxtoken/configs/*` files.
  - Verification: `python - <<'PY'\nimport yaml\nfrom pathlib import Path\ncfg = yaml.safe_load(Path('configs/inference.yaml').read_text(encoding='utf-8'))\ncfg = dict(cfg)\n# disable citations to test mapping\ngen = dict(cfg.get('generation', {}) or {})\ngen['require_citation'] = False\ncfg['generation'] = gen\n# set verifier weights to test mapping\nver = dict(cfg.get('verifier', {}) or {})\nver.update({'alpha': 0.1, 'beta': 0.2, 'gamma': 0.3, 'delta': 0.4})\ncfg['verifier'] = ver\nPath('artifacts/tmp_infer_cfg.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')\nPY\n&& python -m voxtoken.runner.infer_refine --out artifacts/tmp_infer_cfg_run --config artifacts/tmp_infer_cfg.yaml\n&& python -c \"import json; d=json.load(open('artifacts/tmp_infer_cfg_run/run.json')); print(len(d.get('citations',[])))\"`
  - Resolution: Added `_normalize_cfg` + stop-rule logic in `infer_refine` to map paper-facing keys into `gates/realizer/verifier.weights` and to interpret `refine.tau` as the marginal stop threshold (ΔV/Δ|T|), with `refine.min_score_delta` kept as a secondary absolute guard.
- [x] M0029: 将 proposal-level 的 M2/M3/M4 提升为可执行的 `C####`/`P####` 并纳入闭环验收。
  - Ref: `docs/plan.md` 顶部 `C0030–C0032` / `P0028–P0031`
  - Context: 用户要求 proposal-level 的 M2/M3/M4 也必须补齐，因此需要把它们转成可验证的工程承诺（DoD/命令/阈值/证据链）。
  - Acceptance: `docs/plan.md` 顶部新增与 M2/M3/M4 对应的 `C####`/`P####`（含验收与验证命令），并在 Changelog 记录范围变更。
  - Verification: `python -c "import re, pathlib; t=pathlib.Path('docs/plan.md').read_text(encoding='utf-8'); assert re.search(r'\\bC0030\\b', t) and re.search(r'\\bP0028\\b', t); print('ok')"`
  - Resolution: 已将 M2/M3/M4 提升为 `C0030–C0032` / `P0028–P0031`，并更新 `docs/plan.md` Goals/Changelog 与 proposal 说明段落（验证通过）。
- [x] M0030: Counterfactual eval 支持基于 GT sentence boxes 的 grounding 评测（不再使用 slot-supported proxy 作为 GT）。
  - Ref: P0030, C0032
  - Context: 需要将 counterfactual grounding 从 “slot-supported proxy” 升级为 RadGenome 风格的 sentence→GT region 评测接口（支持 `--gt`/`--manifest`）。
  - Acceptance: `counterfactual_eval` 支持 `--gt` 或 `--manifest/--case-id` 读取 GT boxes；输出的 `ground_hit@{0.0,0.1}` 基于 GT sentence boxes（并新增 `ground_mean_iou` 供对比）。
  - Verification: `python -m voxtoken.runner.reproduce --exp E0903`
  - Resolution: 更新 `voxtoken/eval/counterfactuals.py` 以加载 GT sentence boxes 并据此计算 `ground_hit@*`；扩展 `voxtoken.runner.counterfactual_eval` / `eval/counterfactuals.py` CLI 支持 `--gt`/`--manifest`（验证通过）。
- [x] M0031: 补齐 M2 grounding 子集的最小可跑闭环（RadGenome-synth ingest→preprocess→infer→grounding eval→可视化）。
  - Ref: P0028, P0029, C0030, E0900, E0901
  - Context: 需要从 `docs/experiment.md` 一键复现 grounding 子集闭环（含 ingest/preprocess、GT-based 指标、overlay 可视化）。
  - Acceptance: `docs/experiment.md` 新增 E0900–E0901（含 1GPU script + 验证器）；跑通后产出 `metrics.json(l)`（含 grounding 指标）和 grounding overlay SVG/JSON。
  - Verification: `python -m voxtoken.runner.reproduce --exp E0901`
  - Resolution: 新增 RadGenome-synth ingest/preprocess configs（E0900）与 grounding 评测闭环（E0901），并修复 `unified_eval` 在 GT 模式下未创建 out_dir 的 bug（验证通过：E0900/E0901）。
- [x] M0032: 补齐 M3 learned split policy 的“证明 > heuristic”闭环（以 grounding 指标为主）。
  - Ref: P0031, C0031, E0902
  - Context: 需要给出可复现的 learned-policy 训练数据生成与评测实验，并用 validator gate 证明 learned > heuristic。
  - Acceptance: 在固定预算下 `ground_mean_iou` learned 显著高于 heuristic，并通过 `validate_policy_improvement` 闸门。
  - Verification: `python -m voxtoken.runner.reproduce --exp E0902`
  - Resolution: 新增 RadGenome-synth policy dataset 构建（`build_policy_dataset`）、dataset 训练支持（`train_policy`）、改进验证器（`validate_policy_improvement`）与实验 E0902（验证通过）。
- [x] M0033: 补齐 M4 反事实因果实验的最小可跑闭环（grounding drop + mask sanity gate）。
  - Ref: P0030, C0032, E0903
  - Context: 反事实实验（permute Ω / citation swap / mask sanity）需要在 GT sentence boxes 上验证 grounding drop，而不是 proxy grounding。
  - Acceptance: E0903 产出 `counterfactuals.json`，并通过 `validate_counterfactuals` 的 grounding drop gate（permute_omega 与 swap_citations 相对 base 有显著下降）。
  - Verification: `python -m voxtoken.runner.reproduce --exp E0903`
  - Resolution: 基于 GT sentence boxes 跑通 counterfactual 评测与验证闸门（新增实验 E0903，验证通过）。
