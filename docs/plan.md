# Plan

## Goals

- Maintain an auditable, doc-driven closed loop for the repo skeleton (interfaces-only), with stable artifacts/metrics contracts.
- Keep all *runnable commitments* in `C####`/`P####` (including M2/M3/M4 promoted from the proposal); the remaining long-horizon proposal content below stays as notes unless explicitly promoted.

## Algorithm 1 → Repo Mapping (VoxToken++ Inference)

| Algorithm step | Repo implementation | Outputs (contracts) |
|---|---|---|
| 1) init coarse tokens `T_0` | `voxtoken/runner/infer_refine.py:RefineRunner.run_case` → `Tokenizer3D.build_pyramid` + `Tokenizer3D.select_tokens(active_nodes=[])` | `run.json.tokens` (level-0 tokens with `omega_box_mm`) |
| 2) generate `y_0` (with citations) | `RefineRunner._gen_verify` → `EvidenceHead.forward` → `Planner.build_plan` → `Realizer.realize` + `require_citations` | `run.json.report`, `run.json.citations`, `run.json.plan` |
| 3) verifier issues + `𝒱(y_0,T_0)` | `RefineRunner._gen_verify` → `Verifier.verify` | `run.json.verifier_score`, `run.json.issues` |
| 4) refine loop `k=0..K-1` | `RefineRunner.run_case` loop → `_featurize_tokens` → `_select_splits` (`SplitPolicy.score`) → `_refine_select` → `_gen_verify` | `run.json.trace` (per-step split ids + score deltas + latency) |
| 4b) stop rule | `RefineRunner._stop` implements `Δ𝒱/Δ|T| < τ` + `min_score_delta` guard | early stop recorded via shorter `trace` than `max_rounds` |
| 5) final report + token supports + issues | `voxtoken/runner/infer_refine.py:_write_sidecar_artifacts` | `final_report.txt`, `evidence_graph.json` (includes token Ω), `trace.jsonl` |

## Claims (C####)

- [x] C0001: Baseline smoke runs and produces `run.json` + `summary.json` that conform to `docs/results_contract.md`.
  - Evidence: E0000
  - Proof rule: `python -m voxtoken.runner.smoke --out artifacts/smoke` exits 0 and writes the required files/keys.
  - Notes: Legacy claim ID `CLAIM-M0-1`.
- [x] C0002: Unified eval runs on baseline `run.json` and produces `metrics.json` + `metrics.jsonl` that conform to `docs/results_contract.md`.
  - Evidence: E0001
  - Proof rule: `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` exits 0 and writes the required files/keys.
  - Notes: Legacy claim ID `CLAIM-M0-2`.
- [x] C0003: `infer_refine` runs and produces a non-empty report where every sentence has citations.
  - Evidence: E0100
  - Proof rule: `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16` exits 0; `artifacts/infer/run.json` has non-empty `report` and citations for every sentence.
  - Notes: Legacy claim ID `CLAIM-M1-1`.
- [x] C0004: Unified eval can read `tokens_used`/`verifier_score` from an `infer_refine` run and write them to `metrics.json(l)`.
  - Evidence: E0101
  - Proof rule: `python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval` exits 0 and writes `tokens_used`/`verifier_score`.
  - Notes: Legacy claim ID `CLAIM-M1-2`.
- [x] C0005: Heuristic refinement (split) runs and produces a non-empty `trace` with at least one split, while staying within token budget.
  - Evidence: E0200
  - Proof rule: `python -m voxtoken.runner.infer_refine --out artifacts/e0200 --budget 16 --config voxtoken/configs/inference_heuristic.yaml` exits 0 and writes `run.json` with `trace[0].split_token_ids` non-empty and `tokens_used <= budget_B`.
  - Notes: Implements backlog idea E0200 (heuristic split).
- [x] C0006: Policy training produces a reusable checkpoint, and inference can load it to drive split scoring.
  - Evidence: E0300
  - Proof rule: `python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy_e0300.yaml` writes `outputs/train_policy/E0300/checkpoint.json`, then `infer_refine` can run with that checkpoint and produce a valid `run.json`.
  - Notes: Implements backlog idea E0300 (learned split policy scaffold).
- [x] C0007: No-citation ablation produces measurable unsupported sentences (unsupported_rate > 0) under unified eval.
  - Evidence: E0400
  - Proof rule: Run `infer_refine` with citations disabled and then `unified_eval`; `unsupported_rate` in `metrics.json(l)` is > 0 (expected ~1.0 for this skeleton).
  - Notes: Part of backlog idea E0400.
- [x] C0008: No-constrained ablation allows overclaim and is penalized by the verifier (verifier_score decreases and overclaim issues appear).
  - Evidence: E0401
  - Proof rule: Run `infer_refine` with constraints disabled + overclaim enabled; `run.json` contains at least one `Issue(type=\"overclaim\")` and `verifier_score` is lower than the constrained baseline.
  - Notes: Part of backlog idea E0400 (split into a dedicated experiment).
- [x] C0009: Tokenizer training writes a deterministic checkpoint, and inference can load it to assign non-null token codes.
  - Evidence: E0500
  - Proof rule: `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer_e0500.yaml` writes `outputs/train_tokenizer/E0500/checkpoint.json`, then `infer_refine` with that checkpoint writes `run.json` whose `meta.tokenizer.checkpoint_path` matches and `meta.tokenizer.codes_enabled=true`.
  - Notes: Synthetic-only training for the repo skeleton (stdlib-only).
- [x] C0010: Evidence head training writes a deterministic checkpoint, and inference can load it to change finding types based on token codes.
  - Evidence: E0600
  - Proof rule: `python -m voxtoken.runner.train_evidence --config voxtoken/configs/train_evidence_e0600.yaml` writes `outputs/train_evidence/E0600/checkpoint.json`, then inference with that checkpoint produces a report with >=2 distinct finding types.
  - Notes: Synthetic-only training for the repo skeleton (stdlib-only).
- [x] C0011: Unified eval computes `slot_f1` from `run.json` (plan vs extracted slots) and returns ~1.0 for constrained baseline inference.
  - Evidence: E0700
  - Proof rule: Run `infer_refine` (baseline) then `unified_eval`; `slot_f1 >= 0.99`.
  - Notes: Replaces placeholder `slot_f1=0.0`.
- [x] C0012: Inference records `latency_ms.total` in `run.json`, and unified eval propagates it to `metrics.latency_ms.total` (> 0).
  - Evidence: E0701
  - Proof rule: Run `infer_refine` then `unified_eval`; `metrics.latency_ms.total > 0`.
- [x] C0013: `reproduce` can re-run an experiment by `E####` by parsing `docs/experiment.md` and executing its `1GPU script`.
  - Evidence: E0800
  - Proof rule: `python -m voxtoken.runner.reproduce --exp E0200` exits 0 and the underlying command chain completes successfully.
- [x] C0014: Counterfactual evaluation can measure how `unsupported_rate` changes when citations are removed.
  - Evidence: E0801
  - Proof rule: Run E0801; the produced `counterfactuals.json` satisfies `remove_citations > base`.
- [x] C0015: Data ingest + preprocess produces a processed manifest where every case has a deterministic `split` field.
  - Evidence: E0802
  - Proof rule: Run E0802; `manifest.jsonl` exists, has `n>=10`, and every row has a valid `split` in `{train,val,test}`.
- [x] C0016: CT-RATE ingest+preprocess works from `/data/ct_rate` (or `/data/CT-RATE`) and produces a processed manifest with existing report/volume paths.
  - Evidence: E0803
  - Proof rule: Run E0803; `manifest.jsonl` exists, has `n>=20`, `report_path` and `volume_path` exist per row, and splits are valid.
- [x] C0017: CT-RATE train split ingest+preprocess can produce a processed manifest with existing report paths (volume paths may be empty).
  - Evidence: E0804
  - Proof rule: Run E0804; `manifest.jsonl` exists, has `n>=200`, all `report_path` exist, and `case_id` starts with `train_`.
- [x] C0018: `infer_refine` can run by selecting a case from a JSONL manifest and (when available) loading a small NIfTI volume.
  - Evidence: E0805
  - Proof rule: Run E0805; `run.json.meta.input` exists with `case_id` and `volume_loader=nifti`, and the run passes `validate_run`.
- [x] C0019: `infer_refine` can also run on CT-RATE train manifests that have reports but no resolvable volumes (falls back to dummy volume).
  - Evidence: E0806
  - Proof rule: Run E0806; `run.json.meta.input` exists with `volume_loader=dummy`, report_path exists, and the run passes `validate_run`.
- [x] C0020: Batch inference can run over multiple CT-RATE validation cases with real volumes and emit an aggregated `metrics.jsonl`.
  - Evidence: E0810
  - Proof rule: Run E0810; `artifacts/e0810/metrics.jsonl` exists with `n>=5` unique case_ids and all runs use `volume_loader=nifti`.
- [x] C0021: Batch inference can run over the full CT-RATE validation manifest (all rows) when `--max-cases 0`.
  - Evidence: E0811
  - Proof rule: Run E0811; `artifacts/e0811/metrics.jsonl` exists with `n>=20` unique case_ids and all runs use `volume_loader=nifti`.
- [x] C0022: CT-RATE ingest can join predicted multi-abnormality labels into the manifest (`labels_pos`).
  - Evidence: E0820
  - Proof rule: Run E0820; processed manifest has `labels_pos` and at least one row has non-empty labels.
- [x] C0023: Label evaluation can compute per-case multi-label metrics (`precision/recall/f1`) from `run.json` + CT-RATE predicted labels.
  - Evidence: E0821
  - Proof rule: Run E0821; `label_metrics.json` exists and passes `validate_label_metrics` with `n_gold_pos > 0`.
- [x] C0024: Batch label evaluation can produce `label_metrics.jsonl` for a CT-RATE subset.
  - Evidence: E0822
  - Proof rule: Run E0822; `label_metrics.jsonl` exists with `n>=20` and passes `validate_label_metrics_jsonl`.
- [x] C0025: Label-conditioned report generation from CT-RATE `labels_pos` can achieve `f1>=0.99` on a labeled case.
  - Evidence: E0830
  - Proof rule: Run E0830; `label_metrics.json` passes `validate_label_metrics` with `n_gold_pos > 0` and `f1 >= 0.99`.
- [x] C0026: Batch label-conditioned report generation from CT-RATE `labels_pos` can achieve `f1>=0.99` for at least 10 labeled cases.
  - Evidence: E0831
  - Proof rule: Run E0831; `label_metrics.jsonl` passes `validate_label_metrics_jsonl` with `n>=10` and per-row `f1 >= 0.99`.
- [x] C0027: `infer_refine` writes sidecar artifacts (`final_report.txt`, `evidence_graph.json`, `trace.jsonl`) for paper-facing exports.
  - Evidence: E0832
  - Proof rule: Run E0832; sidecar files exist next to `run.json` and pass `validate_run` sidecar checks.
- [x] C0028: Batch inference over 50 CT-RATE cases writes the full artifact bundle per case (run.json + report/evidence/trace sidecars).
  - Evidence: E0833
  - Proof rule: Run E0833; `metrics.jsonl` has `n>=50` and batch runner validates sidecar outputs per case.
- [x] C0029: Counterfactual citation swap increases `unsupported_rate` when unsupported is defined as "citation does not support the slot".
  - Evidence: E0834
  - Proof rule: Run E0834; `validate_counterfactuals` passes with `citation_swap > base` and `remove_citations > base`.
- [x] C0030: Grounding pipeline runs on RadGenome-synth and unified eval reports GT-based grounding metrics.
  - Evidence: E0901
  - Proof rule: Run E0901; `metrics.json` contains `ground_hit@0.0`, `ground_hit@0.1`, `ground_mean_iou` and the grounding overlay artifacts exist.
- [x] C0031: Learned split policy beats heuristic on grounding (higher `ground_mean_iou` under the same budget) on RadGenome-synth.
  - Evidence: E0902
  - Proof rule: Run E0902; `validate_policy_improvement` (or equivalent gate) passes with `ground_mean_iou(learned) > ground_mean_iou(heuristic)`.
- [x] C0032: Counterfactual causality holds on GT grounding: permuting Ω or swapping citations reduces grounding vs base deterministically.
  - Evidence: E0903
  - Proof rule: Run E0903; `validate_counterfactuals` passes grounding drop gates (permute_omega / swap_citations < base) on GT sentence boxes.
- [x] C0033: Tokenizer training emits codebook-usage diagnostics (perplexity) and `recon_error` has sufficient dynamic range for split features.
  - Evidence: E0904
  - Proof rule: Run E0904; tokenizer metrics validate (`perplexity >= 2.0`) and `recon_error` separation validates on a structured dummy volume.
  - Notes: Skeleton uses stdlib-only proxies (k-means on patch means; recon_error=patch variance).
- [x] C0034: Evidence graph sidecar is traceable: every `EvidenceNode.supported_token_ids` maps to tokens with `omega_box_mm` inside `evidence_graph.json`.
  - Evidence: E0905
  - Proof rule: Run E0905; evidence graph validation passes and every evidence node is traceable to token boxes.
- [x] C0035: Verifier is deterministic for the same input/config, and issues are localized to spans + related tokens/evidence.
  - Evidence: E0906
  - Proof rule: Run E0906; verifier stability validation passes comparing two runs on the same case.

- [x] C0036: CT-RATE TS (lung nodules) grounding GT manifest can be built from real CT volumes + TS masks into a runnable per-case GT-box dataset.
  - Evidence: E0907
  - Proof rule: Run E0907; output `manifest.jsonl` exists with `n>=5`, every row has existing `volume_path` + `gt_mask_path`, and `grounding_boxes_by_sent_mm["0"]` is non-empty.
- [x] C0037: Policy training can fit a split policy from CT-RATE TS grounding rewards and write a reusable checkpoint with non-default weights.
  - Evidence: E0908
  - Proof rule: Run E0908; `outputs/train_policy/E0908/checkpoint.json` exists and its `weights` differ from the heuristic defaults.
- [x] C0038: CT-RATE TS grounding benchmark runs fixed/heuristic/learned tokenization across budgets and outputs per-case + aggregate metrics deterministically.
  - Evidence: E0909
  - Proof rule: Run E0909; `metrics.jsonl`/`summary.json` exist and include required keys (`method`, `budget_B`, `tokens_used`, `latency_ms.total`, `ground_mean_iou`, `ground_hit@0.1`).
- [x] C0039: Paper export script generates Table1/2 and Fig2/3 artifacts from benchmark outputs deterministically.
  - Evidence: E0910
  - Proof rule: Run E0910; the CSV tables and figure files exist under `artifacts/paper_e0910/`.

- [x] C0040: Verifier inconsistency rule detects plan-vs-report slot mismatches deterministically.
  - Evidence: E0916
  - Proof rule: `python -m voxtoken.runner.validate_inconsistency_rule` exits 0 and prints `{"ok": true}`.
- [x] C0041: Inference refinement implements the marginal stop rule (ΔV/Δ|T| < tau) to stop early under no-improvement splits.
  - Evidence: E0917
  - Proof rule: `python -m voxtoken.runner.validate_tau_stop --out artifacts/e0917_tau_stop` exits 0 and writes a `run.json` whose `trace` stops before `max_rounds`.
- [x] C0042: Constrained decoding removes overclaim sentences even if the generator attempts to add them.
  - Evidence: E0918
  - Proof rule: `python -m voxtoken.runner.validate_constrained_overclaim --out artifacts/e0918_constrained_overclaim` exits 0; the output report has no hallucinated finding and no `Issue(type="overclaim")`.

- [x] C0100: Papertrack (pseudo-GT): TS lung_nodules case index + GT manifest can be built with non-empty GT boxes and deterministic splits.
  - Evidence: E0911, E0912
  - Proof rule: Run E0911 and E0912; both validators pass and write `cases.jsonl` / `manifest.jsonl` with existing `volume_path` + `gt_mask_path` and non-empty `grounding_boxes_by_sent_mm`.
- [x] C0101: Papertrack (pseudo-GT): policy dataset + policy training + benchmark + bootstrap improvement gate are runnable end-to-end.
  - Evidence: E0913, E0914
  - Proof rule: Run E0913 and E0914; policy dataset validates, checkpoint validates, benchmark summary validates, and the bootstrap improvement gate passes.
- [x] C0102: Papertrack: export CI tables/plots from papertrack benchmark outputs deterministically.
  - Evidence: E0915
  - Proof rule: Run E0915; validator confirms `table1_main_ci.csv` and `fig2_pareto_tokens_ci.png` exist.
- [x] C0103: CT-RATE valid ingest+preprocess yields a deterministic 70/30 train/val split over the available real volumes on this machine.
  - Evidence: E0920, E0921
  - Proof rule: Run E0920 and E0921; raw+processed manifests validate and split counts are non-empty.
- [x] C0104: Effusion pseudo-GT manifests (pleural/pericardial) can be built from CT-RATE valid volumes + TotalSeg masks with non-empty GT boxes.
  - Evidence: E0922, E0923
  - Proof rule: Run E0922 and E0923; every row has existing `gt_mask_path`, `gt_is_pseudo=true`, `coord_system=token_space_mm`, and non-empty GT boxes.
- [x] C0105: Effusion policy datasets (pleural/pericardial, split=train) can be built from GT boxes into a runnable dataset.jsonl.
  - Evidence: E0924, E0925
  - Proof rule: Run E0924 and E0925; dataset validators pass with required feature keys and sufficient rows/cases.
- [x] C0106: Effusion policy training + grounding benchmark (split=val) runs fixed/heuristic/learned across budgets and emits non-empty metrics+summary.
  - Evidence: E0926, E0927
  - Proof rule: Run E0926 and E0927; checkpoint validator passes and benchmark validator confirms required methods exist.

- [x] C0107: Papertrack (GT): RadGenome-ChestCT lung nodule (hi32 token-space) manifest can be built for full-eligible cases with non-empty GT boxes and deterministic splits.
  - Evidence: E0986
  - Proof rule: Run E0986; `validate_ct_rate_ts_manifest` and `validate_split_counts` pass and `coord_system=token_space_mm`.
- [x] C0108: Papertrack (GT): Torch reward-policy training (STOP action; reward regression) writes reusable checkpoints with `model.pt` and non-default weights (multi-seed).
  - Evidence: E1042
  - Proof rule: Run E1042; `validate_policy_checkpoint` passes for seeds 0/1/2 and checkpoints include `model.pt`.
- [x] C0109: Papertrack (GT): RadGenome lung nodule grounding benchmark (reward stop-threshold) runs fixed/heuristic/learned/random/oracle across budgets and emits deterministic summaries (multi-seed).
  - Evidence: E1043
  - Proof rule: Run E1043; `validate_grounding_benchmark` passes per seed and required methods exist.
- [x] C0110: Papertrack (GT): Learned policy achieves a statistically significant improvement over random at budget 32 on RadGenome lung nodule grounding (ΔIoU >= 0.017 and paired Δ CI_low >= 0).
  - Evidence: E1044
  - Proof rule: Run E1044; `validate_improvement_gate` and `validate_paired_delta_ci` pass with the specified thresholds at `budget=32` for `metric=ground_mean_iou`.

## Plan Items (P####)

- [x] P0001: Provide a baseline smoke entrypoint that emits `run.json`/`summary.json` with stable schema.
  - Linked claims: C0001
  - Definition of done: `python -m voxtoken.runner.smoke --out artifacts/smoke` writes `run.json`/`summary.json` per `docs/results_contract.md`.
  - Verification: `python -m voxtoken.runner.smoke --out artifacts/smoke`
  - Touchpoints: `voxtoken/runner/smoke.py`, `voxtoken/schemas.py`, `docs/results_contract.md`
- [x] P0002: Provide a unified evaluation entrypoint that emits `metrics.json`/`metrics.jsonl` with stable schema.
  - Linked claims: C0002, C0004
  - Definition of done: `python -m voxtoken.runner.unified_eval ...` writes `metrics.json`/`metrics.jsonl` per `docs/results_contract.md`.
  - Verification: `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval`
  - Touchpoints: `voxtoken/runner/unified_eval.py`, `docs/eval_protocol.md`, `docs/results_contract.md`
- [x] P0003: Provide a minimal inference closed-loop entrypoint (`infer_refine`) that emits citations + verifier outputs.
  - Linked claims: C0003, C0004
  - Definition of done: `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16` writes `run.json` containing a non-empty `report` and per-sentence `citations`, plus `tokens_used`/`verifier_score`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `voxtoken/models/generator/*`, `voxtoken/verify/*`
- [x] P0004: Implement hierarchical tokenizer + heuristic refinement (split) that records trace steps and respects budget.
  - Linked claims: C0005
  - Definition of done: `infer_refine` can run with `refine.max_rounds>0` and emits `trace` with non-empty split ids; token selection adds finer tokens without exceeding `budget_B`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0200 --budget 16 --config voxtoken/configs/inference_heuristic.yaml`
  - Touchpoints: `voxtoken/models/tokenizer.py`, `voxtoken/runner/infer_refine.py`, `voxtoken/models/policy.py`
- [x] P0005: Implement a minimal policy training loop that outputs a checkpoint and can be loaded for inference scoring.
  - Linked claims: C0006
  - Definition of done: `train_policy` writes `checkpoint.json` with learned weights; `SplitPolicy` can load it; inference run captures the effective weights in artifacts for audit.
  - Verification: `python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy_e0300.yaml`
  - Touchpoints: `voxtoken/runner/train_policy.py`, `voxtoken/models/policy.py`, `voxtoken/runner/infer_refine.py`
- [x] P0006: Add ablation toggles for citations/constraints and verifier rules to reflect unsupported/overclaim differences.
  - Linked claims: C0007, C0008
  - Definition of done: can disable citation emission and/or constraint enforcement via config; verifier reports unsupported/overclaim issues accordingly; unified eval reflects unsupported_rate changes.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0400 --budget 16 --config voxtoken/configs/inference_no_citation.yaml`
  - Touchpoints: `voxtoken/models/generator/*`, `voxtoken/verify/rules.py`, `voxtoken/runner/infer_refine.py`
- [x] P0007: Implement tokenizer training + checkpoint loading, and expose token codes for downstream modules.
  - Linked claims: C0009
  - Definition of done: `train_tokenizer` writes deterministic checkpoint with codebook; `Tokenizer3D` can load it and assign `Token.code`.
  - Verification: `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer_e0500.yaml`
  - Touchpoints: `voxtoken/runner/train_tokenizer.py`, `voxtoken/models/tokenizer.py`
- [x] P0008: Implement evidence head training + checkpoint loading to produce non-trivial finding types in plans/reports.
  - Linked claims: C0010
  - Definition of done: `train_evidence` writes deterministic checkpoint mapping code->finding; `EvidenceHead` loads it and emits different `finding_type` values.
  - Verification: `python -m voxtoken.runner.train_evidence --config voxtoken/configs/train_evidence_e0600.yaml`
  - Touchpoints: `voxtoken/runner/train_evidence.py`, `voxtoken/models/evidence_head.py`, `voxtoken/models/generator/*`
- [x] P0009: Implement `slot_f1` computation in unified eval using a stable slot extractor.
  - Linked claims: C0011
  - Definition of done: `unified_eval` outputs non-placeholder `slot_f1` (plan vs extracted slots) and passes `E0700` validation.
  - Verification: `python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval && python -m voxtoken.runner.validate_metrics --in artifacts/infer_eval/metrics.json --require-slot-f1-ge 0.99`
  - Touchpoints: `voxtoken/runner/unified_eval.py`, `voxtoken/verify/extract_slots.py`
- [x] P0010: Record inference latency in `run.json` and propagate to unified eval metrics.
  - Linked claims: C0012
  - Definition of done: `infer_refine` writes `latency_ms.total > 0`; `unified_eval` copies it to `metrics.latency_ms.total`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16 --config voxtoken/configs/inference.yaml && python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval && python -m voxtoken.runner.validate_metrics --in artifacts/infer_eval/metrics.json --require-latency-total-gt 0`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `voxtoken/runner/unified_eval.py`
- [x] P0011: Implement an experiment reproduction CLI that can re-run `E####` from `docs/experiment.md`.
  - Linked claims: C0013
  - Definition of done: `python -m voxtoken.runner.reproduce --exp E0200` locates the `1GPU script` command in the ledger and executes it (exit code 0).
  - Verification: `python -m voxtoken.runner.reproduce --exp E0200`
  - Touchpoints: `voxtoken/runner/reproduce.py`, `docs/experiment.md`
- [x] P0012: Add counterfactual evaluation runner + validator (citation removal sensitivity).
  - Linked claims: C0014
  - Definition of done: can run counterfactual eval on a `run.json` and validate `remove_citations > base` deterministically.
  - Verification: `python -m voxtoken.runner.counterfactual_eval --in artifacts/infer/run.json --out artifacts/cf_eval && python -m voxtoken.runner.validate_counterfactuals --in artifacts/cf_eval/counterfactuals.json --require-remove-citations-gt-base`
  - Touchpoints: `voxtoken/eval/counterfactuals.py`, `voxtoken/runner/counterfactual_eval.py`, `voxtoken/runner/validate_counterfactuals.py`
- [x] P0013: Add runnable ingest+preprocess configs and a manifest validator for deterministic splits.
  - Linked claims: C0015
  - Definition of done: `voxtoken.data.ingest` + `voxtoken.data.preprocess` can run from configs and `validate_manifest` confirms split fields.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_e0802.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_e0802.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/data_proc_e0802/manifest.jsonl --require-n-ge 10 --valid-splits train val test`
  - Touchpoints: `voxtoken/data/ingest.py`, `voxtoken/data/preprocess.py`, `voxtoken/runner/validate_manifest.py`, `voxtoken/configs/data_*_e0802.yaml`
- [x] P0014: Extend data ingest to support CT-RATE reports and optional volume path resolution.
  - Linked claims: C0016
  - Definition of done: `voxtoken.data.ingest` supports `source: ct_rate` and can read CT-RATE CSVs under `/data/ct_rate` (fallback to `/data/CT-RATE`).
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_e0803.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_e0803.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_proc_e0803/manifest.jsonl --require-n-ge 20 --valid-splits train val test --require-report-path-exists --require-volume-path-exists --require-nonempty-volume-paths-ge 20`
  - Touchpoints: `voxtoken/data/ingest.py`, `voxtoken/data/preprocess.py`, `voxtoken/runner/validate_manifest.py`, `voxtoken/configs/data_*_ct_rate_*.yaml`
- [x] P0015: Add runnable CT-RATE train split ingest+preprocess configs and validate report-path existence.
  - Linked claims: C0017
  - Definition of done: can ingest CT-RATE train reports (subset) into a manifest + per-case report txt files, then preprocess and validate.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_train_e0804.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_train_e0804.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_train_proc_e0804/manifest.jsonl --require-n-ge 200 --valid-splits train val test --require-report-path-exists --require-case-id-prefix train_`
  - Touchpoints: `voxtoken/data/ingest.py`, `voxtoken/configs/data_ingest_ct_rate_train_e0804.yaml`, `voxtoken/configs/data_preprocess_ct_rate_train_e0804.yaml`, `voxtoken/runner/validate_manifest.py`
- [x] P0016: Extend `infer_refine` to optionally select a case from a JSONL manifest and load small NIfTI volumes.
  - Linked claims: C0018
  - Definition of done: `infer_refine` supports `--manifest` + `--case-id`, records `meta.input`, and uses `volume_loader=nifti` when `volume_path` exists.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0805 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --case-id valid_1_a_1 && python -m voxtoken.runner.validate_run --in artifacts/e0805/run.json --require-meta-input --require-meta-input-case-id valid_1_a_1 --require-meta-input-volume-loader nifti --require-meta-input-volume-path-exists`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `voxtoken/runner/validate_run.py`, `docs/experiment.md`
- [x] P0017: Add a runnable experiment for manifest-driven inference on CT-RATE train reports (volume fallback path).
  - Linked claims: C0019
  - Definition of done: `infer_refine` can run on a CT-RATE train manifest row where `volume_path` is empty, and still records `meta.input.report_path`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0806 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_train_proc_e0804/manifest.jsonl --case-id train_1_a_1 && python -m voxtoken.runner.validate_run --in artifacts/e0806/run.json --require-meta-input --require-meta-input-case-id train_1_a_1 --require-meta-input-volume-loader dummy --require-meta-input-report-path-exists`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `docs/experiment.md`, `voxtoken/runner/validate_run.py`
- [x] P0018: Add a batch runner to run `infer_refine` + `unified_eval` over a JSONL manifest and write aggregated metrics.
  - Linked claims: C0020
  - Definition of done: batch runner can select `n` cases from a manifest, run per-case inference/eval, and write `metrics.jsonl` at the top level.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --out artifacts/e0810 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 5 --require-volume-loader nifti && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0810/metrics.jsonl --require-n-ge 5`
  - Touchpoints: `voxtoken/runner/batch_infer_eval.py`, `voxtoken/runner/validate_metrics_jsonl.py`, `voxtoken/runner/infer_refine.py`, `voxtoken/runner/unified_eval.py`
- [x] P0019: Add a scale-test experiment for the batch runner (run all selected rows from the manifest).
  - Linked claims: C0021
  - Definition of done: `--max-cases 0` runs every selected row and writes `metrics.jsonl` with expected row count.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_proc_e0803/manifest.jsonl --out artifacts/e0811 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 0 --require-volume-loader nifti && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0811/metrics.jsonl --require-n-ge 20`
  - Touchpoints: `voxtoken/runner/batch_infer_eval.py`, `docs/experiment.md`
- [x] P0020: Extend CT-RATE ingest to optionally join predicted labels into the manifest and validate `labels_pos`.
  - Linked claims: C0022
  - Definition of done: ingest supports `include_predicted_labels: true` and `validate_manifest` can require `labels_pos` and non-empty counts.
  - Verification: `python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_labeled_e0820.yaml && python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_labeled_e0820.yaml && python -m voxtoken.runner.validate_manifest --in artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --require-n-ge 20 --valid-splits train val test --require-report-path-exists --require-volume-path-exists --require-nonempty-volume-paths-ge 20 --require-labels-pos --require-nonempty-labels-pos-ge 1`
  - Touchpoints: `voxtoken/data/ingest.py`, `voxtoken/runner/validate_manifest.py`, `voxtoken/configs/data_*_e0820.yaml`
- [x] P0021: Add CT-RATE per-case label evaluation runner + validator.
  - Linked claims: C0023
  - Definition of done: `ct_rate_label_eval` can compute metrics from `run.json` + manifest row and validation can gate on `n_gold_pos`.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0821 --budget 16 --config voxtoken/configs/inference.yaml --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --case-id valid_1_a_1 && python -m voxtoken.runner.ct_rate_label_eval --run artifacts/e0821/run.json --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0821_label --case-id valid_1_a_1 && python -m voxtoken.runner.validate_label_metrics --in artifacts/e0821_label/label_metrics.json --require-n-gold-pos-gt 0`
  - Touchpoints: `voxtoken/runner/ct_rate_label_eval.py`, `voxtoken/runner/validate_label_metrics.py`
- [x] P0022: Add CT-RATE batch label evaluation runner + validator.
  - Linked claims: C0024
  - Definition of done: `batch_label_eval` can aggregate `label_metrics.jsonl` across multiple run.json files and validate its schema/size.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0822 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 0 --require-volume-loader nifti && python -m voxtoken.runner.batch_label_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --runs artifacts/e0822/runs --out artifacts/e0822_label && python -m voxtoken.runner.validate_label_metrics_jsonl --in artifacts/e0822_label/label_metrics.jsonl --require-n-ge 20`
  - Touchpoints: `voxtoken/runner/batch_label_eval.py`, `voxtoken/runner/validate_label_metrics_jsonl.py`, `docs/experiment.md`
- [x] P0023: Add CT-RATE label-conditioned report generator from `labels_pos`.
  - Linked claims: C0025
  - Definition of done: `ct_rate_report_from_labels` can generate a `run.json` whose report lines include finding types derived from `labels_pos`, enabling near-perfect label F1 under `ct_rate_label_eval`.
  - Verification: `python -m voxtoken.runner.ct_rate_report_from_labels --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --case-id valid_1_a_1 --out artifacts/e0830 && python -m voxtoken.runner.ct_rate_label_eval --run artifacts/e0830/run.json --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0830_label --case-id valid_1_a_1 && python -m voxtoken.runner.validate_label_metrics --in artifacts/e0830_label/label_metrics.json --require-n-gold-pos-gt 0 --require-f1-ge 0.99`
  - Touchpoints: `voxtoken/runner/ct_rate_report_from_labels.py`, `voxtoken/runner/ct_rate_label_eval.py`, `voxtoken/runner/validate_label_metrics.py`
- [x] P0024: Add batch label-conditioned report generation + JSONL validator thresholds.
  - Linked claims: C0026
  - Definition of done: `batch_ct_rate_report_from_labels` can emit `runs/<case_id>/run.json` for a labeled manifest subset; `validate_label_metrics_jsonl` can gate on per-row F1 thresholds.
  - Verification: `python -m voxtoken.runner.batch_ct_rate_report_from_labels --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --out artifacts/e0831 --max-cases 0 && python -m voxtoken.runner.batch_label_eval --manifest artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --runs artifacts/e0831/runs --out artifacts/e0831_label && python -m voxtoken.runner.validate_label_metrics_jsonl --in artifacts/e0831_label/label_metrics.jsonl --require-n-ge 10 --require-f1-ge 0.99`
  - Touchpoints: `voxtoken/runner/batch_ct_rate_report_from_labels.py`, `voxtoken/runner/validate_label_metrics_jsonl.py`, `docs/experiment.md`
- [x] P0025: Add sidecar artifact outputs for `infer_refine` (report/evidence/trace).
  - Linked claims: C0027
  - Definition of done: `infer_refine` writes `final_report.txt`, `evidence_graph.json`, and `trace.jsonl` next to `run.json` for any run.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0832 --budget 16 --config voxtoken/configs/inference.yaml && python -m voxtoken.runner.validate_run --in artifacts/e0832/run.json --require-final-report-txt --require-evidence-graph-json --require-trace-jsonl`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `voxtoken/runner/validate_run.py`, `docs/experiment.md`
- [x] P0026: Ensure batch inference outputs the full artifact bundle (including sidecars) for each case.
  - Linked claims: C0028
  - Definition of done: `batch_infer_eval` writes `runs/<case_id>/{run.json,final_report.txt,evidence_graph.json,trace.jsonl}` and gates failures via `validate_run`.
  - Verification: `python -m voxtoken.runner.batch_infer_eval --manifest artifacts/ct_rate_train_proc_e0804/manifest.jsonl --out artifacts/e0833 --budget 16 --config voxtoken/configs/inference.yaml --max-cases 50 && python -m voxtoken.runner.validate_metrics_jsonl --in artifacts/e0833/metrics.jsonl --require-n-ge 50`
  - Touchpoints: `voxtoken/runner/batch_infer_eval.py`, `voxtoken/runner/infer_refine.py`, `voxtoken/runner/validate_run.py`, `docs/experiment.md`
- [x] P0027: Strengthen unsupported definition to be slot-supported (not just "has some citation").
  - Linked claims: C0029
  - Definition of done: `unsupported_rate` is measured as "missing citation OR cited tokens do not support the sentence slot"; counterfactual citation swap increases unsupported.
  - Verification: `python -m voxtoken.runner.infer_refine --out artifacts/e0834 --budget 16 --config voxtoken/configs/inference_evidence_ckpt.yaml && python -m voxtoken.runner.counterfactual_eval --in artifacts/e0834/run.json --out artifacts/e0834_cf && python -m voxtoken.runner.validate_counterfactuals --in artifacts/e0834_cf/counterfactuals.json --require-citation-swap-gt-base --require-remove-citations-gt-base --require-base-le 0.01`
  - Touchpoints: `voxtoken/verify/rules.py`, `voxtoken/runner/unified_eval.py`, `voxtoken/eval/counterfactuals.py`, `voxtoken/runner/validate_counterfactuals.py`, `docs/experiment.md`
- [x] P0028: Add RadGenome(-synth) ingest+preprocess configs to enable grounding-subset experiments.
  - Linked claims: C0030
  - Definition of done: `voxtoken.data.ingest` can produce a RadGenome-synth manifest with per-sentence GT boxes and `preprocess` adds deterministic splits.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0900`
  - Touchpoints: `voxtoken/data/ingest.py`, `voxtoken/data/preprocess.py`, `voxtoken/runner/validate_manifest.py`, `docs/experiment.md`
- [x] P0029: Add GT-backed grounding eval + visualization runners and wire unified eval to consume GT boxes.
  - Linked claims: C0030
  - Definition of done: Can run `unified_eval` / `grounding_eval` / `visualize_grounding` on a RadGenome-synth case and get the required metrics + overlay artifacts.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0901`
  - Touchpoints: `voxtoken/runner/unified_eval.py`, `voxtoken/runner/grounding_eval.py`, `voxtoken/eval/visualize_grounding.py`, `docs/results_contract.md`
- [x] P0030: Make counterfactual eval use GT grounding targets (manifest/gt.json) and expose grounding-drop validator gates.
  - Linked claims: C0032
  - Definition of done: `counterfactual_eval` can read GT sentence boxes and `validate_counterfactuals` can gate grounding drops for `permute_omega` / `swap_citations`.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0903`
  - Touchpoints: `voxtoken/eval/counterfactuals.py`, `voxtoken/runner/counterfactual_eval.py`, `voxtoken/runner/validate_counterfactuals.py`
- [x] P0031: Add a learned-policy-vs-heuristic experiment with a deterministic improvement gate on grounding.
  - Linked claims: C0031
  - Definition of done: Under the same budget, learned policy yields higher grounding metric than heuristic on RadGenome-synth, and the claim is enforced by a validator.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0902`
  - Touchpoints: `voxtoken/runner/train_policy.py`, `voxtoken/models/policy.py`, `voxtoken/runner/infer_refine.py`, `voxtoken/runner/unified_eval.py`, `docs/experiment.md`
- [x] P0032: Add tokenizer diagnostics (perplexity) and a `recon_error` separation gate for split-feature sanity.
  - Linked claims: C0033
  - Definition of done: `train_tokenizer` writes codebook usage/perplexity metrics, and validators can gate `perplexity>=2.0` plus `recon_error` dynamic range on a structured dummy volume.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0904`
  - Touchpoints: `voxtoken/runner/train_tokenizer.py`, `voxtoken/runner/validate_tokenizer_train_metrics.py`, `voxtoken/runner/validate_recon_error_separation.py`, `docs/experiment.md`
- [x] P0033: Make `evidence_graph.json` self-contained and validate evidence→token→Ω traceability.
  - Linked claims: C0034
  - Definition of done: `infer_refine` writes `evidence_graph.json` that includes token boxes (or a token index) so evidence nodes can be traced to `omega_box_mm`, and `validate_run` enforces the invariant.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0905`
  - Touchpoints: `voxtoken/runner/infer_refine.py`, `voxtoken/runner/validate_run.py`, `docs/experiment.md`
- [x] P0034: Add a verifier stability validator and an experiment that gates determinism + issue localization.
  - Linked claims: C0035
  - Definition of done: Running inference twice on the same deterministic input yields identical verifier outputs (score + issues), and issues carry valid spans and related tokens/evidence references.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0906`
  - Touchpoints: `voxtoken/verify/rules.py`, `voxtoken/runner/infer_refine.py`, `voxtoken/runner/validate_verifier_stability.py`, `docs/experiment.md`

- [x] P0035: Build a CT-RATE TS grounding manifest (join CT volumes + TS masks; derive GT boxes).
  - Linked claims: C0036
  - Definition of done: Provide a CLI that produces a manifest with `gt_mask_path` and `grounding_boxes_by_sent_mm` for sentence 0, derived from TS segmentation masks on real CT-RATE volumes.
  - Verification: `python -m voxtoken.data.ct_rate_ts_manifest --in artifacts/ct_rate_labeled_proc_e0820/manifest.jsonl --task lung_nodules --out artifacts/ct_rate_ts_nodule_gt_e0907 --config voxtoken/configs/ct_rate_ts_grounding_e0907.yaml --max-cases 10 && python -m voxtoken.runner.validate_ct_rate_ts_manifest --in artifacts/ct_rate_ts_nodule_gt_e0907/manifest.jsonl --require-n-ge 5`
  - Touchpoints: `voxtoken/data/ct_rate_ts_manifest.py`, `voxtoken/runner/infer_refine.py`
- [x] P0036: Build a multi-case policy dataset from CT-RATE TS GT boxes and train a policy checkpoint.
  - Linked claims: C0037
  - Definition of done: Dataset builder writes `dataset.jsonl` with features+reward for multiple cases; `train_policy` can fit weights from it and write a checkpoint.
  - Verification: `python -m voxtoken.runner.build_policy_dataset_multi --manifest artifacts/ct_rate_ts_nodule_gt_e0907/manifest.jsonl --out artifacts/ct_rate_policy_dataset_e0908/dataset.jsonl --config voxtoken/configs/ct_rate_ts_grounding_e0907.yaml --max-cases 10 --split train && python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy_ct_rate_ts_e0908.yaml && python -m voxtoken.runner.validate_policy_checkpoint --in outputs/train_policy/E0908/checkpoint.json --require-nondefault-weights`
  - Touchpoints: `voxtoken/runner/build_policy_dataset_multi.py`, `voxtoken/runner/train_policy.py`, `voxtoken/models/policy.py`
- [x] P0037: Add a CT-RATE TS grounding benchmark runner (tokenization vs GT boxes) with validators.
  - Linked claims: C0038
  - Definition of done: Benchmark runs fixed/heuristic/learned methods across budgets on a manifest, writes `metrics.jsonl` + `summary.json`, and a validator can gate required fields deterministically.
  - Verification: `python -m voxtoken.runner.ct_rate_grounding_benchmark --manifest artifacts/ct_rate_ts_nodule_gt_e0907/manifest.jsonl --out artifacts/e0909 --config voxtoken/configs/ct_rate_ts_grounding_e0907.yaml --budgets 8 16 32 --max-cases 10 --policy-ckpt outputs/train_policy/E0908/checkpoint.json && python -m voxtoken.runner.validate_grounding_benchmark --in artifacts/e0909/summary.json --require-methods fixed heuristic learned`
  - Touchpoints: `voxtoken/runner/ct_rate_grounding_benchmark.py`, `voxtoken/runner/validate_grounding_benchmark.py`
- [x] P0038: Add a paper export helper that turns benchmark metrics into Table1/2 and Fig2/3.
  - Linked claims: C0039
  - Definition of done: CLI writes CSV tables + PNG/SVG figures from `metrics.jsonl`, with deterministic sorting.
  - Verification: `python -m voxtoken.runner.paper_export --in artifacts/e0909/metrics.jsonl --out artifacts/paper_e0910 && python -m voxtoken.runner.validate_paper_artifacts --dir artifacts/paper_e0910`
  - Touchpoints: `voxtoken/runner/paper_export.py`, `eval/pareto.py`

- [x] P0039: Add a RadGenome-ChestCT GT manifest builder for lung nodule (hi32 token-space; full-eligible).
  - Linked claims: C0107
  - Definition of done: `radgenome_mask_manifest` can produce a `manifest.jsonl` with existing `gt_mask_path`, non-empty `grounding_boxes_by_sent_mm`, deterministic splits, and token-space coordinates.
  - Verification: `python -m voxtoken.runner.reproduce --exp E0986`
  - Touchpoints: `voxtoken/data/radgenome_mask_manifest.py`, `voxtoken/configs/data_ingest_radgenome_chestct_full_e0940.yaml`, `voxtoken/configs/data_preprocess_radgenome_chestct_full_e0940.yaml`, `voxtoken/configs/ct_rate_ts_grounding_e0985.yaml`
- [x] P0040: Add Torch reward-policy training (STOP action; reward regression) for RadGenome lung nodule (multi-seed).
  - Linked claims: C0108
  - Definition of done: `train_policy_torch` trains a reward-regression policy from oracle+DAgger traces, writes `checkpoint.json` + `model.pt`, and checkpoint validation passes.
  - Verification: `python -m voxtoken.runner.reproduce --exp E1042`
  - Touchpoints: `voxtoken/runner/train_policy_torch.py`, `voxtoken/models/policy.py`, `voxtoken/models/policy_mlp.py`, `voxtoken/runner/validate_policy_checkpoint.py`
- [x] P0041: Add reward stop-threshold benchmarking for RadGenome lung nodule with strong baselines (random/oracle) and multi-seed aggregation.
  - Linked claims: C0109
  - Definition of done: Benchmark runner supports methods `fixed,heuristic,learned,random,oracle`, budgets `8/16/32`, and `--seed`; outputs validate deterministically.
  - Verification: `python -m voxtoken.runner.reproduce --exp E1043`
  - Touchpoints: `voxtoken/runner/ct_rate_grounding_benchmark.py`, `voxtoken/runner/validate_grounding_benchmark.py`, `voxtoken/configs/ct_rate_ts_grounding_e1039_reward_stopthr.yaml`
- [x] P0042: Add paper export gates for statistically significant learned-vs-random improvement at budget 32 (paired-delta CI).
  - Linked claims: C0110
  - Definition of done: Paper export aggregates multi-seed metrics into CI tables/plots and validators gate `ΔIoU@B32>=0.017` and `paired Δ CI_low>=0`.
  - Verification: `python -m voxtoken.runner.reproduce --exp E1044`
  - Touchpoints: `voxtoken/runner/paper_export.py`, `voxtoken/runner/validate_improvement_gate.py`, `voxtoken/runner/validate_paired_delta_ci.py`

## Next Steps (Paper-grade Roadmap)

The repo is still “interfaces-first”, but the papertrack grounding loop is now end-to-end and gated (C0110/E1044). To continue toward paper-grade coverage:

1) Multi-task grounding (real masks): replicate the C0110-style paired-Δ gate on at least 2–3 additional mask-defined findings (e.g., pleural effusion, pericardial effusion, pneumothorax) using the existing GT-manifest builders (`ct_rate_ts_manifest`, `radgenome_mask_manifest`) and the reward-policy training/benchmark stack (`train_policy_torch`, `ct_rate_grounding_benchmark`, `paper_export`).
2) Stronger baselines: add at least one non-trivial baseline beyond `random/heuristic` (e.g., fixed ROI crop or uniform-depth refinement) inside `ct_rate_grounding_benchmark`, then extend `paper_export` tables to include it.
3) Explain “why it works”: add a small analysis script to summarize per-case deltas (learned−random and learned−heuristic), and reuse `visualize_grounding` to dump a few qualitative overlays for “wins” and “fails”.
4) Bridge to report generation: run `infer_refine` on real-volume manifests (CT-RATE/RadGenome) with the learned policy checkpoint and track `unsupported_rate/slot_f1/latency_ms` alongside grounding; promote to commitments only after stable contracts/validators exist.

## Changelog

- 2026-02-01: Add papertrack RadGenome lung nodule GT (hi32 full-eligible) reward-policy pipeline and pass the paired-Δ significance gate at B32 (E0986, E1042–E1044).
- 2026-01-31: Promote “paper-grade” CT-RATE TS grounding benchmark into runnable commitments (`C0036–C0039` / `P0035–P0038`) and add experiments E0907–E0910.
- 2026-01-31: Implemented + proved E0907–E0910 (CT-RATE TS GT manifest, policy dataset+training, grounding benchmark, and paper export).
- 2026-01-31: Promote proposal acceptance checks (Stage T/E + A5 verifier stability) into runnable commitments (`C0033–C0035` / `P0032–P0034`) and prove them via E0904–E0906.
- 2026-01-31: Unify the “repo skeleton” tree to match the canonical tokenizer layout (`voxtoken/models/tokenizer.py`), avoiding a conflicting `models/tokenizer/` subpackage sketch.
- 2026-01-31: Fix section-10 repo skeleton code block indentation so the tree is copy-pastable and path-consistent.
- 2026-01-30: Align docs with docs-spec stable IDs (`C####`/`P####`/`E####`) while preserving legacy IDs (`CLAIM-M*`, `EXP-*`) in notes.
- 2026-01-30: Promote proposal-level M2/M3/M4 into runnable commitments (`C0030–C0032` / `P0028–P0031`) per user request, so they are tracked and proved via `docs/experiment.md`.
- 2026-01-30: Implemented + proved E0900–E0903 (RadGenome-synth grounding pipeline, learned split policy > heuristic, and GT-based counterfactual grounding drops).
- 2026-01-30: Verified M0/M1 commands locally: `smoke`, `unified_eval`, `infer_refine`, `infer_eval` (artifacts under `artifacts/`).
- 2026-01-30: Added next backlog scope as explicit `C0005–C0008` / `P0004–P0006` for E0200/E0300/E0400 implementation (unchecked until proved).
- 2026-01-30: Implemented + proved E0200/E0300/E0400/E0401 (see `.rd_queue/results/` and `docs/experiment.md` checkboxes).
- 2026-01-30: Implemented + proved E0500/E0600 (synthetic train_tokenizer/train_evidence checkpoints + inference loading).
- 2026-01-30: Implemented + proved E0700/E0701 (slot_f1 computation + latency propagation in unified eval).
- 2026-01-30: Added next runnable scope E0800–E0802 (reproduce CLI, counterfactual eval, and data ingest/preprocess pipeline) (unchecked until proved).
- 2026-01-30: Implemented + proved E0800–E0802 (see `.rd_queue/results/` and `docs/experiment.md` checkboxes).
- 2026-01-30: Added CT-RATE data pipeline check (E0803) (unchecked until proved).
- 2026-01-30: Implemented + proved E0803 (CT-RATE subset ingest/preprocess) (see `.rd_queue/results/E0803-*.json`).
- 2026-01-30: Added next runnable scope E0804–E0805 (CT-RATE train split ingest + manifest-driven inference) (unchecked until proved).
- 2026-01-30: Implemented + proved E0804–E0805 (see `.rd_queue/results/E0804-*.json` and `.rd_queue/results/E0805-*.json`).
- 2026-01-30: Added E0806 (manifest-driven inference on CT-RATE train reports, dummy volume fallback) (unchecked until proved).
- 2026-01-30: Implemented + proved E0806 (see `.rd_queue/results/E0806-*.json`).
- 2026-01-30: Added E0810 (batch inference/eval over CT-RATE validation cases) (unchecked until proved).
- 2026-01-30: Implemented + proved E0810 (see `.rd_queue/results/E0810-*.json`).
- 2026-01-30: Added E0811 (batch runner scale test: run all selected validation rows) (unchecked until proved).
- 2026-01-30: Implemented + proved E0811 (see `.rd_queue/results/E0811-*.json`).
- 2026-01-30: Added E0820–E0822 (CT-RATE predicted labels join + label eval) (unchecked until proved).
- 2026-01-30: Implemented + proved E0820–E0822 (see `.rd_queue/results/E0820-*.json`, `.rd_queue/results/E0821-*.json`, `.rd_queue/results/E0822-*.json`).
- 2026-01-30: Added + proved E0830 (CT-RATE report from `labels_pos`, gated by `f1>=0.99`) (see `.rd_queue/results/E0830-*.json`).
- 2026-01-30: Added + proved E0831 (batch CT-RATE report from `labels_pos`, JSONL validator supports `--require-f1-ge`) (see `.rd_queue/results/E0831-*.json`).
- 2026-01-30: Added + proved E0832 (infer_refine sidecar artifacts: report/evidence/trace) (see `.rd_queue/results/E0832-*.json`).
- 2026-01-30: Added + proved E0833 (batch 50-case artifact bundle for M0 scale check) (see `.rd_queue/results/E0833-*.json`).
- 2026-01-30: Added + proved E0834 (slot-supported unsupported + citation-swap counterfactual gate) (see `.rd_queue/results/E0834-*.json`).

---

# Repo Plan（M0）& Proposal（Long-horizon）

本仓库当前阶段：**interfaces-only（M0）**。为了让 doc-driven 循环可终止、可审计，本文件在顶部新增 **M0 Claims & Evidence Map**：只有该小节中的 claims 会被视为“必须证明”的工程承诺；其余大段内容是 long-horizon proposal / 设计笔记（为后续研究与实现服务），**不作为当前收敛的阻塞项**。

2026-01-30 更新：用户要求补齐 proposal-level 的 M2/M3/M4，因此已在顶部以 `C0030–C0032` / `P0028–P0031` 形式提升为可验证承诺；其余未提升内容仍作为笔记保留。

## M0 Claims & Evidence Map（收敛判据）

| Claim ID | Claim（可验证陈述） | Evidence（在哪看） | Verify（命令） | Status |
|---|---|---|---|---|
| CLAIM-M0-1 | baseline smoke 可运行，并生成符合 results contract 的 `run.json`/`summary.json` | `docs/experiment.md` → E0000（legacy: EXP-0000）；产物：`artifacts/smoke/` | `python -m voxtoken.runner.smoke --out artifacts/smoke` | PROVED |
| CLAIM-M0-2 | unified eval 可运行，并生成符合 results contract 的 `metrics.json`/`metrics.jsonl` | `docs/experiment.md` → E0001（legacy: EXP-0001）；产物：`artifacts/eval/` | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` | PROVED |

## M1 Claims & Evidence Map（最小可运行推理闭环）

| Claim ID | Claim（可验证陈述） | Evidence（在哪看） | Verify（命令） | Status |
|---|---|---|---|---|
| CLAIM-M1-1 | `infer_refine` 可运行，生成非空 report 且每句都有 citation | `docs/experiment.md` → E0100（legacy: EXP-0100）；产物：`artifacts/infer/run.json` | `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16` | PROVED |
| CLAIM-M1-2 | `unified_eval` 可读取 `infer_refine` 的 `tokens_used/verifier_score` 并写入 `metrics.json(l)` | `docs/experiment.md` → E0101（legacy: EXP-0100）；产物：`artifacts/infer_eval/metrics.jsonl` | `python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval` | PROVED |

## M0 Contracts（契约/协议）

- Results contract：`docs/results_contract.md`
- Unified evaluation protocol：`docs/eval_protocol.md`
- Experiment matrix（baseline-first）：`docs/experiment.md`
- Project index（入口/结构）：`docs/project_index.md`

## Change Log（before/after + rationale）

- 2026-01-30：新增 M0 Claims & Evidence Map；补齐 `docs/project_index.md`；将 `docs/experiment.md` 从占位表修复为可审计矩阵；同步 README（对齐 claims/矩阵/入口命令）。
- 2026-01-30：补齐 M1 最小推理闭环（`infer_refine`）并将其纳入 claims/evidence；统一评测读取 `tokens_used/verifier_score`。

---

本文档给出一份**按 NeurIPS Oral 的“硬标准”反推设计**的完整 proposal（以 **3D tokenization** 为主线），目标是将 reviewer 常见质疑点——**novelty、heuristic、grounding 真实性、active 是否真的 active、指标是否硬、因果性是否成立**——转化为论文的“不可替代贡献”。

本文档不对“中稿概率”作承诺，但结构将**按“面向 Oral 级投稿所需证据链”**组织：按该证据链实施，可显著降低因“像拼装系统”而被一票否决的风险。

---

# VoxToken++：Budgeted Adaptive 3D Tokenization for Proof-Carrying CT Reporting

## 0. TL;DR（投 Oral 的一句话）

我们把 3D CT 报告生成改写为一个**预算约束的 3D tokenization 问题**：

> 在 token 预算 (B) 下，通过**可学习的层级 3D token 分配（octree / hierarchical VQ tokens）**主动细化信息密度，并用**token-citation + 受限解码（constrained decoding）**生成“携带证据”的报告，使 **unsupported claim 机制性趋近 0**，同时在 **correctness–latency Pareto** 上显著优于固定 tokenization / 切片 / ROI 裁块方法。

---

## 1. 研究问题（Problem Statement）

给定 3D CT 体数据 (V) 与 token 预算 (B)（约束计算/显存/时延），学习一个 tokenization 函数与生成器，使得：

* **Tokenize**：输出可变长度 token 集合
  [
  T = {(t_i, \Omega_i)}_{i=1}^{|T|},\quad |T|\le B
  ]
  其中 (t_i) 是 token 表示（离散 id 或连续 embedding），(\Omega_i\subset \mathbb{R}^3) 是其**明确的 3D 支持域（box/voxel set）**；
* **Generate**：生成报告 (y)，并且每条关键医学断言都必须附带 token 引用集合 (C_k\subseteq {1,\dots,|T|})（token-citation）；
* **Guarantee**：报告中的结构化临床事实（finding/side/location/size/negation/uncertainty）必须从 token 证据中“可回溯”，并能用 verifier 程序化检查。

---

## 2. 核心贡献（Contributions，按 Oral 口味写成 3 条“不可替代”）

### C1. **Learned Budgeted Adaptive 3D Tokenization（不是 heuristic crop）**

提出 **VoxToken++**：一个**层级离散 3D tokenizer**（octree + VQ/RVQ），在推理时以 **split 决策**动态分配 token 密度，实现可变长度 token 序列，显式优化 correctness–cost。

### C2. **Proof-Carrying Generation via Token-Citation + Constrained Decoding（机制性消灭 unsupported）**

提出 **token-citation 受限生成**：关键事实的词元只能从证据图中取值，并强制输出引用 token 集合；由此使 unsupported claim 从“评测指标”变成“结构性很难发生”的性质（并提供可验证的 gate）。

### C3. **Verifier-as-Metric-and-Reward（把自检变成算法，而不是 prompt 技巧）**

设计一个可复现 verifier：输出可枚举 issues（missing-slot / inconsistency / overclaim / unsupported），既作为主评测指标，也作为 split policy 的训练 reward，实现 tokenize→generate→verify→refine 的闭环学习。

---

## 3. 方法（Method）：最小不可分闭环 + 每一步都能“被评审承认是方法”

### 3.1 组件总览

* **Tokenizer** ( \mathcal{T}_\phi )：层级离散 tokens（coarse→fine），每个 token 带 3D 支持域 (\Omega)
* **Split Policy** ( \pi_\psi )：决定对哪些 token 做 split（细化）
* **Evidence Head** ( \mathcal{E}_\eta )：从 tokens 提取结构化证据（finding/属性）+（关键子集）mask/measurement
* **Generator** ( \mathcal{G}_\theta )：基于证据图受限生成 + token-citation
* **Verifier** ( \mathcal{V} )：程序化验证与计分（也是 reward）

最终推理：**Tokenize → Generate → Verify → Refine → Regenerate**，直到边际收益不足或预算耗尽。

---

## 3.2 Tokenizer：层级离散 3D tokens（真正的“tokenize”）

### Token 表示

我们采用 **Residual VQ（RVQ）/ VQ-MAE** 风格的 3D tokenizer：

* 3D encoder (f_\phi) 将体数据编码为多尺度特征金字塔：({F^{(0)},F^{(1)},...,F^{(L)}})
* 每层通过 codebook 量化得到离散 token id：
  [
  t^{(\ell)}*{u} = \mathrm{VQ}(F^{(\ell)}*{u}) \in {1,\dots,K_\ell}
  ]
* 每个 token 对应一个明确支持域 (\Omega^{(\ell)}_u)（一个 3D box 或 voxel block）

### 层级（Octree）结构

* Level-0：粗网格 tokens 覆盖全体积（低成本、全局上下文）
* Level-1..L：仅对被选择 split 的节点生成更细 tokens（局部高信息密度）

> **关键点**：本方法不是“固定 grid tokens”，而是“层级 tokens + 可学习 split”。这就是 Oral 级 novelty 的落脚点。

---

## 3.3 Split 决策：显式目标函数 + 可学习策略（回应“heuristic”质疑）

我们把“细化哪些 token”形式化为一个预算约束优化：

### 目标函数（核心）

在第 (k) 次 refinement，我们选择一组 split 操作 (A_k)，最大化**期望验证收益**与**证据覆盖**，并惩罚 token 成本：

[
\max_{A_k} \ \mathbb{E}\big[\Delta \mathcal{V}(y_{k+1},T_{k+1}) \mid T_k, A_k\big] - \lambda \cdot \Delta|T|
]

其中 (\Delta \mathcal{V}) 来自 verifier 的分数提升（见 3.6），(\Delta|T|) 是 split 增加的 token 数。

### 可训练 Split Policy（主线，面向 Oral 级投稿必备）

定义每个候选 token 的状态特征 (x_i)：

* token-level **uncertainty**（证据头输出熵）
* token-level **reconstruction error**（VQ/MAE 残差）
* token-level **citation pressure**（生成时 attention/citation 频次）
* anatomy prior（若有器官 mask，可作为 token 的器官分布）
* history（该 token 是否已 split、split 深度）

策略：
[
\pi_\psi(\text{split}\mid x_i) \in [0,1]
]

训练方式（回应“RL 不稳定”的常见质疑）：

* 先用启发式/teacher 产生 logged data（entropy/recon）
* 再做 **contextual bandit / offline RL**：奖励 = verifier 分数提升 − λ token 成本
* 最终展示：**learned policy > heuristic**（这条是 Oral 关键）

---

## 3.4 Evidence Graph：token → 结构化证据（让引用不是装饰）

我们构建证据图 (G)：

* 节点：token 或 token 组（含 (\Omega)、embedding/id）
* 节点属性：结构化预测

  * finding（present/absent/uncertain）
  * laterality、location、size bins、negation/uncertainty
* 关键 finding 子集：输出局部 mask/weak box，并计算 measurement（diameter/extent/volume）

**为什么这一步是“不可分”？**
因为它把 token-citation 变成“引用证据节点”，而不是“attention 可视化”。如果没有结构化证据，citation 很容易被 reviewer 说成“装饰”。

---

## 3.5 受限生成：Token-citation + Constrained Decoding（机制性 unsupported≈0）

报告生成分两步：

### Step 1：生成结构化报告计划（Plan）

从证据图中抽取一个 report plan（器官→findings→impression），每条事实是一个 slot tuple：
[
(\text{finding_type}, \text{side}, \text{location}, \text{size}, \text{certainty})
]
并附带支持 token 集合 (C_k)。

### Step 2：自然语言表述（Surface Realization）

用 LLM 只负责把 slot tuple 转写成自然语言，但关键 token（finding/side/location/size/certainty）**必须来自 slot**。
每句输出格式：

* sentence text
* cited token ids：`[tok:12, tok:57,…]`

**Constrained decoding**：如果模型试图生成不在 slot 中的 finding/侧别/大小，解码直接禁止或替换为不确定模板。

> 这能把 unsupported claim 从“需要 critic 才能发现”变成“生成机制阻止它出现”。

---

## 3.6 Verifier：指标也是 reward（让“反思”成为算法）

Verifier 输出 issues（程序化可复现）：

* missing-slot：出现 finding 但缺 side/location/size
* inconsistency：左右、否定、数量冲突
* overclaim：证据不确定却用确定语气
* unsupported：无 citation 或 citation token 不包含对应 slot 证据

并输出总分：
[
\mathcal{V} = \text{ClinicalCorrectness} - \alpha #missing - \beta #inconsistency - \gamma #overclaim - \delta #unsupported
]

**训练 split policy** 的 reward：
[
R = \Delta \mathcal{V} - \lambda \Delta|T|
]

---

## 3.7 推理算法（Oral 级必须能写成 Algorithm 1）

**Algorithm: VoxToken++ Inference**

1. 初始化 coarse tokens (T_0)（Level-0）
2. 生成 (y_0)（带 citations）
3. verifier 得到 issues 与 (\mathcal{V}(y_0,T_0))
4. for k = 0..K-1:

   * 根据 (\pi_\psi) 选择要 split 的 token 集合 (A_k)
   * 得到 refined tokens (T_{k+1})
   * 生成 (y_{k+1})，验证得分 (\mathcal{V}(y_{k+1},T_{k+1}))
   * 若边际收益 ( \Delta\mathcal{V} / \Delta|T| < \tau) 或预算耗尽，stop
5. 输出 final report + citations + token supports + issues（应接近 0）

---

# 4. 理论/可解释性（Oral 常见加分点：哪怕弱也要有）

不必做大理论，但至少给出一个**可写进 paper 的“合理性保证”**：

1. **边际收益递减假设**：refinement 使得证据不确定性下降通常呈递减（coarse→fine），因此 (\Delta\mathcal{V}) 在 token 数上呈 diminishing returns（经验上可验证）。
2. **Stop rule 的合理性**：当 (\Delta\mathcal{V} / \Delta|T| < \tau) 时停止，等价于在 correctness–cost Pareto 上选择膝点。
3. **Unsupported 的结构性上界**：在 constrained decoding + verifier gate 下，unsupported claim rate 被强约束到接近 0（可作为“系统性质”陈述）。

---

# 5. 实验设计（按 Oral 反推：必须给出的图/表）

## 5.1 主指标（拒绝只卷 ROUGE）

主结果必须围绕四件事：

1. **Clinical correctness**：结构化事实 F1（finding + side + location + size bins + negation/uncertainty）
2. **Grounding**：sentence→token→GT region 命中（hit-rate / IoU / Dice）
3. **Unsupported claim rate**：应显著低于 baselines（最好接近 0）
4. **Efficiency**：#tokens、latency（wall-clock）
   并给出 **Pareto 曲线**：correctness vs #tokens / latency。

## 5.2 Baselines（防“没对齐 baseline”）

最小不可分 baseline 列表（必须全做）：

* **Fixed-grid 3D tokens**（同 encoder、同生成器，只禁用 adaptive split）
* **2D slice tokens**（uniform slice sampling / 2.5D）
* **ROI crop pipeline**（经典 coarse-to-fine/region-guided）
* **Heuristic split**（entropy / recon error）vs **learned split policy（本文方法）**
* **No-citation**（去掉 constrained/citation）
* **No-refine**（只做一次 tokenization）
* **Citation=attention top-k**（弱基线）vs **证据图 constrained citation（本文方法）**（证明不是可视化）

## 5.3 反事实/因果实验（Oral 级关键）

这是面向 Oral 级投稿的关键证据链，必须写进 proposal：

* **Permutation test**：随机打乱 token 支持域 (\Omega_i)（保持 token embedding 不变）→ grounding 与 correctness 显著下降
* **Citation swap**：交换 cited token ids（保持句子不变）→ verifier 立即报 unsupported
* **Mask-level sanity**：对有 GT mask 的 finding，细化 tokens 应更集中覆盖 lesion（统计 token 密度 vs lesion overlap）

这些实验是用来正面击穿“citation 只是 attention 装饰”的质疑。

## 5.4 关键可视化（Oral 需要的 3 张主图）

* **Fig1**：VoxToken++ 框图（层级 tokens + refine loop + constrained generation）
* **Fig2**：Pareto 曲线（相对于 fixed-grid & ROI crop 形成优势/dominance）
* **Fig3**：Grounding 可视化 + 反事实实验结果（强说服力）

---

# 6. 数据与落地策略（保证可实现“硬 grounding”）

需要确保至少一部分数据具备 **region-level ground truth**，否则 Oral 很难。

最小策略：

1. 主数据集：大规模 CT + 报告（用于 token→text 学习）
2. Grounding 子集：带 lesion mask/box 的 CT 子集（哪怕规模小）用于 sentence→token→region 的硬评测与训练（关键 finding 头）

如果现成数据 grounding 不够：

* 可构建一个**小规模标注子集**（几十到几百例），只标 2–3 个高价值 finding（结节/积液/气胸），但要做到高质量。Oral 认可“少而硬”。

---

# 7. 里程碑（保证是“最小不可分、可执行”的计划）

**M0（2周）**：固定 grid tokens + token-citation + verifier gate 跑通（unsupported 明显下降）
**M1（4周）**：实现 octree adaptive split（先 heuristic），做 Pareto 曲线
**M2（6–8周）**：训练 learned split policy，证明 > heuristic（关键）
**M3（8–10周）**：grounding 子集 + 反事实实验（关键）
**M4（10–12周）**：跨数据集/外部验证（锦上添花但对 Oral 很重要）

---

# 8. 风险与对策（提前写给 reviewer 看）

1. **citation 漂移** → 用证据图 slot 约束 + permutation/counterfactual 证明因果
2. **token 太粗导致伪 grounding** → 强制关键 finding 使用 mask-head；refine 优先 lesion-prone 区域
3. **RL 不稳定** → 先 offline bandit（logged data），再小步 fine-tune；主结果先用 learned policy>heuristic
4. **过度保守（只写 uncertain）** → verifier 同时惩罚 recall/漏报；加入 coverage reward
5. **计算太重** → 以 token budget 为第一约束；报告 wall-clock 与吞吐

---

# 9. 核心卖点（面向 Oral 级投稿；可写在 Introduction 的最后一段）

* 我们不是在做“更强的报告生成器”，而是在做**可解释、可验证、预算可控的 3D tokenization**。
* 通过 **learned adaptive tokens**，我们实现 **同等正确性更低 token/时延** 或 **同等预算更高正确性** 的 Pareto 优势。
* 通过 **token-citation + constrained decoding**，我们把 unsupported claim 从“难测的主观问题”变成“可被机制性约束的系统性质”。
* 通过 **verifier-as-reward**，我们把“慢思考/反思”落到可训练的决策闭环。

---

## 下一步优先事项（两件；决定是否具备 Oral 级证据链）

1. **把 split 决策写成一个可以实现的目标函数 + learned policy**（这是 novelty 的锚）
2. **准备 grounding 子集 + 做 3 个反事实实验**（这是说服力的锚）

---

可选补充（非主线）：

* **NeurIPS 论文大纲（逐段落写什么）**
* **Algorithm 1/2 的伪代码（tokenize/refine/constrained decode/verifier）**
* **主实验表格模板（每一列对应 reviewer 质疑点）**

实现版本选择：更倾向 **离散 VQ tokens 主线**，或先用 **连续 tokens MVP** 并写成“可插拔 VQ”的版本。

下面给出两套内容，**都按“最小不可分（atomic）”**来写： 

* **A. 实现方案（Implementation Plan）**：每个模块的**接口、落地实现、训练/推理闭环、产物**一次性写全。
* **B. 实验方案（Experiment Plan）**：按 NeurIPS Oral 反推的**主指标、baseline、消融、反事实因果实验、图表清单**一次性写全。

本文档将把“能跑通的最小 MVP”与“面向 Oral 级投稿的必备增强”写成同一条链路，确保每一步都有**验收检查（acceptance check）**，避免出现“做了很多但无法证明”的情况。

---

# A. 实现方案（最小不可分，端到端闭环）

## A0. 统一协议与产物（先固定，避免后续接口/评测不一致）

### A0.1 统一坐标与支持域

* 所有 token 必须带**空间支持域** (\Omega_i)：推荐用**原始高分辨率体坐标系的 3D box**（`x0,x1,y0,y1,z0,z1`），必要时再附 voxel mask（仅关键 finding）。
* 必须提供 `low↔high` 映射（若有重采样），保证 token box 可回溯到原 CT。

### A0.2 统一 Schema（JSON 可序列化）

最小对象集合（每个都必须落盘）：

* `Token`: `{token_id, level, code, embedding_ref, omega_box, parent_id, children_ids}`
* `Citation`: `{sent_id, cited_token_ids:[...]}`
* `EvidenceNode`: `{eid, type, attrs{side,location,size,certainty,negation}, supported_token_ids:[...], optional_mask_ref}`
* `Issue`: `{type, span, reason, related_eids/tokens}`
* `Trace`: 每轮 refine 的决策与耗时：`{k, B_used, split_tokens, added_tokens, verifier_before/after, latency}`

### A0.3 端到端输出三件套（不可少）

* `final_report.txt`（含 token-citation）
* `evidence_graph.json`
* `trace.jsonl`（每轮一条）

**验收检查**：任意 case 必须能生成这三件套，即使模型很弱也要跑通。

---

## A1. 数据管线（Ingest & Preprocess）

### A1.1 输入与规范化（MVP）

* 读入 DICOM/NIfTI → HU（CT）→ 重采样到 `target_spacing`（如 1–2mm，按任务定）
* windowing：至少两路（骨窗/软组织窗或肺窗/纵隔窗），拼成 `C=2` 通道
* 归一化：clamp + z-score/min-max
* 输出：`V ∈ R^{C×D×H×W}`

### A1.2 Grounding 子集（硬要求）

为了 Oral，你必须有一部分带 **region GT**（mask/box）。最小策略：

* 主数据：CT + report（规模大，用于 token→text）
* 子集：2–3 个高价值 finding 的 lesion mask/box（规模可以小，但要硬）

**验收检查**：子集里每个样本必须有 `{mask 或 box}`，能计算 IoU/Dice/hit-rate。

---

## A2. Tokenizer（层级离散 3D tokens：VQ/RVQ + octree）

你要的“tokenize”必须是**离散 tokens**，否则 reviewer 会说你只是 patch embedding。

### A2.1 模型结构（最小可跑 + 具备层级）

* 3D Encoder (f_\phi)：输出多尺度特征金字塔 ({F^{(0)},...,F^{(L)}})

  * 实现：3D Conv / 3D Swin / UNETR encoder 都行，但要能输出多尺度。
* 每层 quantizer：RVQ / VQ（每层一个 codebook 或共享 codebook）

  * 输出 token id：(t_u^{(\ell)} = \text{VQ}(F_u^{(\ell)}))

### A2.2 支持域 (\Omega) 的实现（关键）

* Level-(\ell) 的 token 网格大小是 ((D_\ell,H_\ell,W_\ell))
* token (u=(z,y,x)) 的支持域 box 由 stride 与 spacing 决定：

  * `omega = grid_cell_to_box(u, level, spacing, origin)`
* 存 `parent_id`：(\ell) → (\ell+1) 8 个子格（octree）

### A2.3 两种实现路径（推荐先做“选择式”，再做“按需式”）

**路径 A（最省工程，先冲主结果）— 选择式层级 tokenization**

* 一次 forward 得到所有层 tokens（coarse+fine）
* refine 时不重新编码，只是**从更细层“挑更多 tokens”**（split=选择子 token）
* 成本主要体现在生成器 cross-attn token 数增加（#tokens/latency）

**路径 B（更强但更难）— 按需式 tokenization**

* 初始只算 Level-0
* split 某 token 时，对该 (\Omega) 区域 crop 高分块 → 编码 → quantize 得到子 tokens
* 优点：encoder 计算也随预算增长，Pareto 更漂亮（更像“预算化计算”）

**验收检查**：路径 A 先跑通即可；路径 B 作为 Oral 强化项。

### A2.4 Tokenizer 训练（Stage-1：自监督）

* 目标：VQ-MAE / reconstruction

  * loss = 重建 loss + commitment loss + codebook usage 正则（避免 codebook collapse）
* 输出：每层 codebook 使用率、重建误差分布（作为后续 split feature）

**验收检查**：codebook perplexity 不塌；重建误差能区分复杂区域/简单区域（用于 split 特征）。

---

## A3. Evidence Head（token → 结构化证据 +（关键子集）mask/measurement）

这是让 citation “不是装饰”的关键模块。

### A3.1 输入输出

**Input**：选中的 tokens（含 level、embedding、(\Omega)）
**Output**：Evidence nodes（结构化 slots）+ optional mask/measurement（只对关键 finding）

### A3.2 结构（最小可跑）

* 对每个 token embedding 做 multi-head：

  * finding（present/absent/uncertain 或概率）
  * side / location（可先粗粒度，如 left/right/unknown）
  * size bins（先离散 bins，稳定）
  * negation/uncertainty（可并入 certainty）
* 对关键 finding 子集（例如结节/积液/气胸）：

  * **mask head**：在 (\Omega) crop 内做轻量 3D U-Net 输出 mask（或弱 box）
  * measurement：diameter/extent/volume

### A3.3 Evidence Graph 构建（merge/prune）

* merge：同一 finding、空间重叠的 token 合并为一个 evidence node
* prune：每类最多保留 N 条（按 certainty + 是否有 mask 优先）

**验收检查**：Graph 节点数受控；每个 evidence node 都能回溯到 token ids 与 (\Omega)。

---

## A4. Generator（Proof-carrying：token-citation + constrained decoding）

把 unsupported 从“指标”变成“机制性难发生”。

### A4.1 两阶段生成（最小可跑且可控）

**Stage-1：Plan 构建（结构化）**

* 输入：Evidence Graph
* 输出：slot tuples 列表（每条事实绑定 supporting token ids）

  * 最小实现：**非 LLM**（规则/排序）

    * 按器官分组 → 每组取 top evidence → 形成 findings + impression
  * Oral 强化：用小模型/LLM 生成 plan，但仍要 verifier 校验

**Stage-2：Surface Realization（表述）**

* 最小实现：**模板+fill**（保证约束）
* 强化：LLM 文本润色，但关键字段来自 slot，不允许自由编造

### A4.2 Constrained decoding（硬约束）

* finding/side/location/size/certainty 的 token 只能从 plan 的离散集合里取
* 每句必须输出 `cited_token_ids`

**验收检查**：生成文本中所有关键句都有 citation；去掉 citation 直接判 fail。

---

## A5. Verifier（指标 + reward，程序化可复现）

### A5.1 Issue taxonomy（最小集合）

* `missing_slot`
* `inconsistency`
* `overclaim`
* `unsupported`（无 citation / citation 不支持该 slot）

### A5.2 Verifier score

[
\mathcal{V} = \text{ClinicalCorrectness} - \alpha#missing - \beta#inconsistency - \gamma#overclaim - \delta#unsupported
]

* ClinicalCorrectness 的最小实现：结构化 slots 与 GT（或 labeler 抽取）做 F1（见实验部分）

**验收检查**：同一输入多次运行 verifier 输出稳定；issues 能定位到句子 span 与 tokens/evidence。

---

## A6. Split Policy（learned budgeted refinement：bandit/offline RL）

这是“主动”与“非 heuristic”的核心。

### A6.1 状态特征 (x_i)（每个候选 token 一条）

最小必备：

* evidence uncertainty entropy
* tokenizer recon error（VQ residual）
* citation pressure（被引用频次/attention mass）
* depth/history（split 深度、是否已 split）

### A6.2 训练数据（logged trajectories）

用 heuristic policy（entropy+recon）跑一批 case，记录：

* 在 step k 选择 split token set (A_k)
* refine 后 (\Delta \mathcal{V}) 与 (\Delta|T|)
* reward (R=\Delta\mathcal{V}-\lambda\Delta|T|)

### A6.3 学习方式（稳定优先）

* **contextual bandit**（推荐先做）：

  * 训练 (\hat{Q}(x)\approx \mathbb{E}[R|x])，选择 top-m token split
* 进阶：offline RL（CQL/IQL 类）在 bandit 之上再做

**验收检查**：learned policy 在 offline evaluation（IPS/DR 或回放）优于 heuristic；在线小规模验证能提升 Pareto。

---

## A7. 推理闭环（Algorithm 1：Tokenize→Generate→Verify→Refine→Regenerate）

实现上就是一个可复现 runner：

```text
T0 = coarse_tokens(V)
y0, G0 = generate(T0)
V0, issues0 = verify(y0, G0)

for k in 0..K-1:
  candidates = expandable_tokens(Tk)
  Ak = policy_select(candidates, features, budget_left)
  Tk+1 = refine(Tk, Ak)      # select children (path A) or on-demand encode (path B)
  yk+1, Gk+1 = generate(Tk+1)
  Vk+1, issuesk+1 = verify(yk+1, Gk+1)
  if (Vk+1-Vk)/( |Tk+1|-|Tk| ) < tau: break
return y*, G*, trace
```

**验收检查**：每轮 trace 写清：split 哪些 token、增加多少 token、latency、verifier 提升。

---

## A8. 代码结构（最小可落地 repo skeleton）

```
voxtoken/
  configs/
    train_tokenizer.yaml
    train_evidence.yaml
    train_policy.yaml
    inference.yaml
  data/
    ingest.py
    preprocess.py
    splits/
  models/
    encoder3d.py
    vq.py              # VQ/RVQ quantizer
    tokenizer.py       # hierarchical tokens + omega mapping
    evidence_head.py
    generator/
      planner.py       # rule/LLM plan
      realize.py       # template or LLM
      constrained.py   # constraints
    policy.py
  verify/
    rules.py
    scorer.py
    extract_slots.py   # from GT or labeler
  runner/
    train_tokenizer.py
    train_evidence.py
    train_policy.py
    infer_refine.py
  eval/
    metrics.py
    counterfactuals.py
    pareto.py
  artifacts/           # outputs
```

---

# B. 实验方案（最小不可分，Oral 反推）

## B0. 需要证明的 4 个核心命题（每个对应一组实验）

1. **Adaptive tokenization 的 Pareto 优势**：同等 correctness 更少 tokens/latency，或同等预算更高 correctness
2. **token-citation 的因果性**：citation 不是 attention 装饰（反事实实验必须成立）
3. **unsupported claim 机制性趋近 0**：constrained + verifier gate 显著压低 unsupported
4. **learned split policy > heuristic**：主动策略带来额外 Pareto 改善

---

## B1. 数据与切分（必须能支持 grounding）

### B1.1 数据组成（最小）

* **主数据集**：CT + report（用于 token→text）
* **grounding 子集**：带 lesion mask/box（用于 sentence→token→region 硬评测）

### B1.2 切分

* train/val/test：按 patient-level split
* grounding 子集必须覆盖 test（否则容易被质疑为“只做训练不做评测”）

---

## B2. 指标（主指标必须硬）

### B2.1 Clinical correctness（主）

* 从生成 report 中抽取结构化 slots（finding/side/location/size/negation/uncertainty）
* 与 GT slots 做 micro/macro F1
* 若 GT 无结构化 slots：用固定抽取器（规则或医学 IE 模型）做一致评测（关键是可复现）

### B2.2 Grounding（主）

* sentence → cited tokens → (\Omega) 与 GT mask/box 的关系

  * hit-rate：是否命中（IoU>0 或 Dice>0）
  * IoU/Dice（可选）
* token density vs lesion overlap（refine 后是否更集中）

### B2.3 Unsupported claim rate（主）

* `unsupported = 句子无 citation 或 citation 不支持其 slot`
* 目标：本文方法接近 0，baseline 明显更高

### B2.4 Efficiency（主）

* #tokens（每轮、最终）
* wall-clock latency（必须真实计时）
* GPU memory peak（可选）

### B2.5 Pareto（必须图）

* correctness vs #tokens
* correctness vs latency

---

## B3. Baselines（最小不可分清单，必须齐）

1. **Fixed-grid 3D tokens**（同 encoder + 同 generator，只禁用 adaptive split）
2. **2D slice tokens / 2.5D**（uniform sampling）
3. **ROI crop pipeline**（传统 coarse-to-fine，非 token）
4. **Heuristic split**（entropy/recon）
5. **Learned split policy（本文方法）**
6. **No-citation**（去掉 citation/constrained）
7. **No-refine**（只用 level-0 或一次性 tokens）
8. **Citation = attention top-k**（弱引用） vs **evidence-graph constrained citation（本文方法）**（强引用）

---

## B4. 消融（每条都要能回答一个 reviewer 问题）

* A1：adaptive split vs fixed tokens（证明 tokenization 贡献）
* A2：learned split vs heuristic split（证明非 heuristic）
* A3：constrained vs unconstrained（证明 unsupported 机制性下降）
* A4：mask-head on/off（证明 grounding 不是伪）
* A5：path A（选择式）vs path B（按需式）（若实现 B，这是 Oral 强加分）

---

## B5. 反事实/因果实验（Oral 级关键；最小三条必须做）

1. **Permutation test**：随机打乱 token 支持域 (\Omega_i)（保持 embedding 不变）

   * 预期：grounding 与 correctness 明显下降
2. **Citation swap**：交换 cited token ids（句子不变）

   * 预期：verifier 立即报 unsupported，上升显著
3. **Mask-level sanity**：对有 GT mask 的 finding

   * 统计 refine 后新增 tokens 的 (\Omega) 与 lesion overlap 分布显著上升（相对随机/heuristic）

> 这三条是用来击穿“citation 只是 attention 可视化”的。

---

## B6. 结果呈现（Oral 必备 3 张主图 + 2 张主表）

### 主图（必须）

* **Fig1**：方法框图（层级 tokens + refine loop + constrained generation）
* **Fig2**：Pareto（correctness–#tokens / correctness–latency）
* **Fig3**：Grounding 可视化 + 反事实结果（bar/curve）

### 主表（必须）

* **Table1**：主指标对比（correctness、grounding、unsupported、latency）
* **Table2**：消融（split policy、constrained、mask head）

---

## B7. 训练/评测协议（可复现）

* 固定随机种子（≥3 seeds）
* 同等 token budget 下比较（避免“我 token 更多所以更好”的喷点）
* 同等 backbone 下比较（只换 tokenization/策略/约束）
* 所有输出 artifacts（report/evidence/trace）写入 `artifacts/run_id/`

---

# C. 最小可执行里程碑（每一步都有“必须验收”的标准）

## M0（跑通闭环）

* fixed tokens（level-0）+ template generator + verifier + citations
* **验收**：unsupported rate 显著低于 no-citation；trace 完整

## M1（层级 tokens + refine）

* 选择式 split（从 level-1/2 挑子 tokens）
* **验收**：Pareto 曲线出现边际收益递减；refine 有效

## M2（grounding 子集）

* mask head 只做 2–3 类 finding
* **验收**：Fig3 能画出 token box/mask overlay + grounding 指标

## M3（learned split policy）

* contextual bandit 训练 (\hat{Q}(x)) 并在线验证
* **验收**：learned policy > heuristic（Pareto dominate 或在固定预算上提升显著）

## M4（反事实因果）

* permutation/citation swap/mask sanity 全部跑通
* **验收**：反事实显著劣化（统计显著），否则 Oral 很难

---

## 第一周任务清单（可直接开工，无需额外前置信息）

1. 把 Schema + trace runner 写出来（A0/A7）
2. 实现 fixed-grid tokens（先用连续 embedding 也行，但接口按离散 token 设计）
3. 实现 rule-based planner + template realization + citations（A4）
4. 实现 verifier（A5）
5. 跑 50 个样本输出 artifacts，确保三件套全有（验收 M0）

---

可选：将上述内容进一步工程化到可直接写代码的粒度：

* **每个模块的函数签名**（Python typing）
* **4 个 config 模板**（tokenizer/evidence/policy/infer）
* **Table1/Table2/Fig2/Fig3 的具体列名与作图脚本接口**（确保结果格式不会在写论文时崩）。

下面给出一份**“最小不可分（atomic）可落地”**的完整方案：将 A/B 进一步**工程化到“可直接开 repo 写代码/跑实验”的粒度**——包含：

* **模块边界与接口（Python typing 级别）**
* **训练/推理闭环（Algorithm 1/2 可直接进论文）**
* **4 份 YAML 配置模板**
* **评测脚本与产物格式**
* **Table/Figure 的列名与作图输入约定**
* **反事实实验的可执行实现方式**

论文主线采用**离散 VQ tokens**（面向 Oral 级表述），同时提供一个**连续 tokens 的 MVP 替身**（保持接口不变，便于先跑通再替换为 VQ）。

---

## 0. 两条原则（不写清楚会导致后续接口/评测不一致）

### 0.1 “Token = 表示 + 3D 支持域 + 可追溯父子关系”

无论先用连续还是离散，**Token 数据结构必须固定**，并且每个 token 必须具备：

* `omega_box_mm = (x0,x1,y0,y1,z0,z1)`：在**原始物理坐标（mm）**的 3D box（不是 resample 后 index）
* `level`、`parent_id`、`children_ids`：支持层级 split
* `code`：离散 VQ 时为 `int`，连续 MVP 时可为 `None`
* `embed_ref`：embedding 存储索引（避免 json 塞大向量）

### 0.2 “三件套产物（任何样本、任何 baseline 都必须输出）”

每个 case 必须落盘：

1. `final_report.txt`（每句带 citations）
2. `evidence_graph.json`（结构化 slots + token 支持）
3. `trace.jsonl`（每轮 refine：split、预算、时延、verifier 前后）

这是后续做 Pareto、做反事实、做可视化的**唯一可信证据链**。

---

## 1. 数据选择（让 grounding 变硬，不靠嘴）

至少需要一个“报告生成主数据”+ 一个“硬 grounding 子集”。

推荐组合（公开可复现、论文可引用）：

* **CT-RATE**：大规模胸部 CT 体数据 + 放射科报告（用于 report learning / 自监督 tokenizer）([arXiv][1])
* **RadGenome-Chest CT**：在 CT-RATE 基础上扩展的**region-guided**数据：提供大量**句子↔分割 mask**级别 grounding（用于最关键的“硬 grounding”主评测）([Nature][2])
* （可选）**RAD-ChestCT**：大规模多异常/位置标签胸 CT（用于外部泛化或弱监督证据头）([Zenodo][3])
* （可选）**LIDC-IDRI**：肺结节 CT 数据（可用来做结节 mask/box 级 sanity 或训练 mask-head）([癌症影像库][4])

> 关键点：RadGenome 提供 “**sentence→mask**” 的硬评测抓手；无需一开始就拿到所有 lesion mask，也可以先把“citation/因果”部分做成硬结论。

---

## 2. 代码级实现方案（Atomic 模块 + 接口）

下面给出**可直接照着写**的模块边界与 typing（建议用 `dataclasses` + `pydantic` 管 json）。

### 2.1 核心数据结构（必须固定）

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal, Any

BoxMM = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1

@dataclass
class Token:
    token_id: int
    level: int
    omega_box_mm: BoxMM
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # representation
    code: Optional[int] = None          # VQ token id (discrete); None for continuous MVP
    embed_ref: Optional[int] = None     # index into a tensor bank on disk (np.memmap/pt)

@dataclass
class Citation:
    sent_id: int
    cited_token_ids: List[int]

@dataclass
class EvidenceNode:
    eid: str
    finding_type: str                   # e.g., nodule/effusion/atelectasis/normal...
    attrs: Dict[str, Any]               # side, location, size_bin, certainty, negation...
    supported_token_ids: List[int]
    optional_mask_ref: Optional[str] = None  # path to mask in npy/nii.gz
    optional_measure: Optional[Dict[str, float]] = None  # diameter/volume...

IssueType = Literal["missing_slot","inconsistency","overclaim","unsupported"]

@dataclass
class Issue:
    type: IssueType
    span: Tuple[int, int]               # char span in report, or (sent_id, token_span)
    reason: str
    related_eids: List[str] = field(default_factory=list)
    related_tokens: List[int] = field(default_factory=list)

@dataclass
class TraceStep:
    k: int
    budget_total: int
    budget_used: int
    split_token_ids: List[int]
    added_token_ids: List[int]
    verifier_score_before: float
    verifier_score_after: float
    latency_ms: Dict[str, float]        # tokenize/generate/verify/total
```

---

## 2.2 Tokenizer（离散主线 + 连续 MVP 兼容）

### 2.2.1 Tokenizer 接口（保持不变）

```python
import torch
from typing import NamedTuple

class TokenPyramid(NamedTuple):
    # tokens_by_level[l] is a list of Token (with omega) and a tensor bank ref
    tokens_by_level: Dict[int, List[Token]]
    # embed_bank_path points to saved tensor bank on disk for each level
    embed_bank_path: Dict[int, str]

class Tokenizer3D(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...
    @torch.no_grad()
    def build_pyramid(self, volume: torch.Tensor) -> TokenPyramid:
        """
        volume: (C, D, H, W) float32
        returns: multi-level tokens, each token has omega_box_mm.
        """
        ...

    @torch.no_grad()
    def select_tokens(self, pyramid: TokenPyramid, active_nodes: List[int], budget_B: int) -> List[Token]:
        """
        Path A (选择式)：从 pyramid 中选 token（包含 coarse + 被 split 的 fine tokens）
        """
        ...
```

### 2.2.2 两种 tokenizer 落地路径（先 A 后 B）

* **Path A（推荐先做，工程最省）**：一次前向拿到全部 level 的 token embedding / code，然后 refine 就是“**选择更多细层 token**”。
* **Path B（Oral 强化，真正 budget 化计算）**：只对被 split 的节点 crop→encode→quantize（encoder 计算也随预算走）。

论文写法：**方法定义支持 Path B**，实现先用 A，后面补 B（有时间就上）。

---

## 2.3 Evidence Head（让 citation 有“可检验语义”）

### 2.3.1 接口

```python
class EvidenceHead(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...

    def forward(self, tokens: List[Token], embed_bank: torch.Tensor) -> List[EvidenceNode]:
        """
        tokens: selected tokens
        embed_bank: (N, d) embeddings aligned with tokens
        returns: evidence nodes with structured attrs & supported_token_ids
        """
        ...
```

### 2.3.2 最小可跑实现（M0/M1 必须能做）

* **每 token 多头分类**（即使很弱也要跑通）：

  * `finding_type`（多标签或 top-k）
  * `certainty/negation`
  * `laterality`（left/right/na）
  * `coarse_location`（lung/pleura/mediastinum/other…，先粗粒度）

* **Graph merge**：空间重叠 + finding_type 相同 → 合并为一个 EvidenceNode（supported_token_ids 合并）

### 2.3.3 Oral 强化（M2/M3 才上）

* 对 2–3 个高价值 finding（例如 nodule/effusion/pneumothorax）加：

  * `mask_head(omega crop) -> lesion mask`
  * `measurement`（diameter/extent/volume）

> 可用 RadGenome 的句子↔mask 做“器官/区域 grounding”的硬评测，再用少量 lesion 标注把 mask-head 做硬 sanity（两条线并行）。

---

## 2.4 Planner + Realizer（两阶段生成，天然可约束）

### 2.4.1 Plan 的结构（slot tuple + token 支持）

```python
@dataclass
class FactSlot:
    finding_type: str
    side: str
    location: str
    size_bin: str
    certainty: str
    supported_token_ids: List[int]

@dataclass
class ReportPlan:
    facts: List[FactSlot]
    impression: List[FactSlot]  # or derived summary facts
```

### 2.4.2 Planner（M0 用规则就够；M2 再换 LLM）

```python
class Planner:
    def __init__(self, cfg: Dict[str, Any]): ...

    def build_plan(self, evidence: List[EvidenceNode]) -> ReportPlan:
        """
        Deterministic: sort by severity/certainty, group by anatomy, cap per organ.
        """
        ...
```

### 2.4.3 Realizer（M0 模板填充；M2 加 LLM 润色但不放权）

```python
class Realizer:
    def __init__(self, cfg: Dict[str, Any]): ...

    def realize(self, plan: ReportPlan) -> Tuple[str, List[Citation]]:
        """
        Returns report text and per-sentence citations.
        """
        ...
```

---

## 2.5 Constrained Decoding（把 unsupported 变成“结构性难发生”）

即使使用 LLM，也要把约束写成**可执行 gate**：

* **关键词集合约束**：finding/side/location/size/certainty 只能来自 plan
* **每句必须输出 citation**：没有 citation → verifier 判 unsupported（不可协商）

最小实现：模板天然满足；LLM 强化实现：先生成草稿，再做**slot 对齐与替换**（不通过就 fallback 到模板句）。

---

## 2.6 Verifier（指标=reward=可复现实验开关）

### 2.6.1 Verifier 接口

```python
class Verifier:
    def __init__(self, cfg: Dict[str, Any]): ...

    def verify(self, report_text: str, citations: List[Citation], plan: ReportPlan) -> Tuple[float, List[Issue]]:
        """
        1) parse report -> slots (or use the plan as canonical)
        2) check missing_slot / inconsistency / overclaim / unsupported
        returns score and issues
        """
        ...
```

### 2.6.2 评分（建议固定，别每次改）

[
\mathcal{V} = \text{SlotF1} - \alpha #missing - \beta #inconsistency - \gamma #overclaim - \delta #unsupported
]

* **M0**：SlotF1 可以先用“对 reference report 的 slot 抽取”（silver），哪怕抽取器简单也行，但必须固定可复现。
* **M2**：RadGenome 的 grounded sentence/region 可提供硬对齐的部分（location/organ 维度）([Nature][2])

---

## 2.7 Split Policy（learned active 的最短路径：contextual bandit）

### 2.7.1 特征与动作

```python
@dataclass
class TokenFeatures:
    token_id: int
    level: int
    recon_error: float
    evidence_entropy: float
    citation_pressure: float
    history_splits: int

class SplitPolicy(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...

    def score(self, feats: List[TokenFeatures]) -> List[Tuple[int, float]]:
        """returns (token_id, score)"""
        ...
```

### 2.7.2 训练（离线 bandit，稳）

* 用 heuristic（entropy+recon）跑出 logged trajectories（每一步 refine 后的 (\Delta \mathcal{V})）
* 训练一个 ( \hat{Q}(x)\approx \mathbb{E}[\Delta \mathcal{V}-\lambda\Delta|T| \mid x] )
* 推理时选 top-m split（受 budget 约束）

> 这样可以非常清楚地在论文中陈述：**learned policy > heuristic**，且训练稳定。

---

## 2.8 推理 Runner（Algorithm 1 可直接进 paper）

```python
class RefineRunner:
    def __init__(self, tokenizer, evidence_head, planner, realizer, verifier, policy, cfg): ...

    @torch.no_grad()
    def run_case(self, volume, budget_B: int) -> Dict[str, Any]:
        pyramid = self.tokenizer.build_pyramid(volume)          # Path A
        Tk = self.tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=budget_B)

        plan, report, cites, score, issues = self._gen_verify(Tk, pyramid)
        trace = []

        for k in range(self.cfg["refine"]["max_rounds"]):
            feats = self._featurize_tokens(Tk, pyramid, cites, issues)
            split_ids = self._select_splits(feats, budget_left=budget_B-len(Tk))
            Tk2 = self._refine_select(pyramid, Tk, split_ids, budget_B)

            plan2, report2, cites2, score2, issues2 = self._gen_verify(Tk2, pyramid)
            trace.append(TraceStep(...))

            if self._stop(score, score2, len(Tk), len(Tk2)): 
                break
            Tk, plan, report, cites, score, issues = Tk2, plan2, report2, cites2, score2, issues2

        return self._dump_artifacts(report, cites, plan, trace, issues)
```

---

# 3. 训练方案（按“最小不可分”拆成 3 个 stage）

## Stage T（Tokenizer 自监督：VQ-MAE / VQ-VAE）

**目标**：得到离散 codebook + 多尺度 tokens；并产出 recon error 特征。

* 输入：CT-RATE volumes（无需报告）([arXiv][1])
* loss：recon + commitment + codebook usage regularization
* 产物：`tokenizer.ckpt` + 每层 `codebook.pt` + `recon_error_stats.json`

**验收**：

* codebook perplexity 不塌
* recon error 能区分“复杂区域/简单区域”（用于 split feature）

---

## Stage E（Evidence Head：结构化 slots + 可选 mask-head）

**目标**：token→finding/attrs，能构 evidence graph。

* 输入：

  * CT-RATE（弱监督：全局 abnormality labels）
  * RadGenome（强监督：句子↔mask 对齐，用来训练 location/organ 相关与 grounding）([Nature][2])
* 产物：`evidence_head.ckpt`

**验收**：

* evidence graph 节点数受控（每 organ cap N）
* 每个 evidence node 都能追溯 `supported_token_ids + omega_box_mm`

---

## Stage P（Policy：offline contextual bandit）

**目标**：learned split > heuristic split。

* 数据：用 heuristic 跑出来的 `trace.jsonl`（logged trajectories）
* 训练：回归 ( \hat{Q}(x) ) 或 pairwise ranking（谁 split 更赚）
* 产物：`policy.ckpt`

**验收**：

* 在离线回放/重放上，learned 策略在相同预算下 (\mathcal{V}) 更高或 Pareto 更优

---

# 4. 配置模板（4 份 YAML，可直接照抄改路径）

## 4.1 `configs/train_tokenizer.yaml`

```yaml
data:
  dataset: ct_rate
  root: /data/ct_rate
  spacing_mm: [1.0, 1.0, 3.0]
  windows:
    - {center: -600, width: 1500}   # lung
    - {center: 40, width: 400}      # mediastinum
model:
  encoder: conv3d_unet
  levels: [0,1,2]
  level_grids:
    - [8, 8, 8]
    - [16,16,16]
    - [32,32,32]
  quantizer:
    type: rvq
    codebook_size: [1024, 1024, 1024]
    embed_dim: 256
train:
  objective: vq_mae
  mask_ratio: 0.6
  batch_size: 2
  lr: 2e-4
  epochs: 50
  save_dir: artifacts/tokenizer
```

## 4.2 `configs/train_evidence.yaml`

```yaml
data:
  datasets:
    - {name: radgenome, root: /data/radgenome_chestct, use_grounded_sentences: true}
    - {name: ct_rate, root: /data/ct_rate, use_abnormality_labels: true}
model:
  evidence_head:
    embed_dim: 256
    heads:
      finding_type: {num_classes: 20}
      laterality: {num_classes: 3}
      location: {num_classes: 12}
      certainty: {num_classes: 3}
    mask_head:
      enabled: false
train:
  batch_size: 4
  lr: 1e-4
  epochs: 30
  save_dir: artifacts/evidence
```

## 4.3 `configs/train_policy.yaml`

```yaml
data:
  traces_glob: artifacts/runs/*/trace.jsonl
features:
  use_recon_error: true
  use_evidence_entropy: true
  use_citation_pressure: true
  use_history: true
model:
  type: mlp_regressor
  hidden: [128, 64]
train:
  lr: 1e-3
  epochs: 20
  save_dir: artifacts/policy
```

## 4.4 `configs/inference.yaml`

```yaml
runtime:
  device: cuda
  seed: 0
refine:
  budget_B: 512
  max_rounds: 5
  tau: 1e-3
generation:
  mode: template   # template | llm_rewrite
  require_citation: true
verifier:
  alpha: 0.5
  beta: 1.0
  gamma: 1.0
  delta: 5.0
output:
  save_dir: artifacts/runs
```

---

# 5. 实验方案（按 Oral 反推：4 个命题 → 4 组结果）

## 命题 H1：Adaptive tokenization 给出 correctness–cost Pareto 优势

**做法**：在多个 budget B（如 128/256/512/1024）下画 Pareto。

* X 轴：`#tokens` 或 `latency_ms`
* Y 轴：`Slot-F1`（主）+ `Grounding hit-rate`（主）

**关键对比**：

* Fixed-grid（同 tokenizer/同生成器，只禁用 split）
* ROI crop pipeline（传统 coarse-to-fine）
* 2D/2.5D slice tokens

---

## 命题 H2：Citation 不是 attention 装饰（因果反事实必须成立）

最小三件套（proposal 里那三条，必须可执行）：

1. **Permutation((\Omega))**：打乱 token 的 `omega_box_mm`（embedding 不变）
   预期：grounding 与 correctness 显著下降（统计显著）。

2. **Citation swap**：交换句子引用的 token ids（句子文本不变）
   预期：verifier 的 `unsupported` 立刻暴涨。

3. **Mask-level sanity**（在 RadGenome 的 region mask 上做也行）：
   统计 refine 后新增 tokens 与目标 mask 的 overlap 分布显著上升（相对随机/heuristic）。

RadGenome 之所以适合：它有**句子↔mask**规模化数据，能把上述反事实做成“硬结论”。([Nature][2])

---

## 命题 H3：Unsupported claim rate 机制性趋近 0（不是靠 prompt）

对比：

* No-citation（去掉 citation/约束）
* Attention-topk citation（弱引用）
* 本文方法：EvidenceGraph + constrained generation + verifier gate（强引用）

**主指标**：

* `Unsupported (%)`：句子无 citation 或 citation 不支持 slot
* `Overclaim (%)`：证据不确定但用确定语气

目标：本文方法接近 0，baseline 显著更高。

---

## 命题 H4：Learned split policy > heuristic split（“active 真 active”）

对比：

* heuristic split（entropy / recon error）
* learned bandit policy（本文方法）
* random split（下界）

呈现方式：

* 固定预算 B：比较 (\mathcal{V})、Slot-F1、Grounding、latency
* Pareto：learned 曲线应整体优于 heuristic 或至少在关键 budget 上显著更好

---

# 6. 指标定义（论文里需要写得像“硬标准”，而不是 NLP 指标）

## 6.1 Clinical correctness（Slot-F1）

slot 维度建议固定为：

* finding_type
* laterality
* coarse_location
* size_bin（M0 可先不做；M2 再加）
* certainty/negation

计算：micro/macro F1（建议 micro 为主，macro 辅助）

## 6.2 Grounding

给定一句话的 citations → token boxes → 与 GT mask/box 的关系：

* `Hit@τ`：是否存在 cited token 与 GT mask overlap > τ
* `Mean IoU` 或 `Dice`（有 mask 时）

RadGenome 提供“句子对应 region mask”的直接评测通道。([Nature][2])

## 6.3 Unsupported / Overclaim（Verifier 输出）

* unsupported：缺 citation / citation 不支持 slot
* overclaim：certainty mismatch（如 evidence=uncertain 但文本=definite）

## 6.4 Efficiency

* `#tokens`
* `latency_ms`（tokenize / gen / verify / total）
* （可选）`peak_mem`

---

# 7. 表格与作图模板（可直接按列落 CSV）

## 7.1 Table 1（主表：对齐 reviewer 所有硬喷点）

`table_main.csv` 列名建议：

* `method`
* `budget_B`
* `tokens_final`
* `lat_total_ms`
* `slot_f1_micro`
* `ground_hit@0.0`（overlap>0）
* `ground_hit@0.1`
* `unsupported_sent_pct`
* `overclaim_sent_pct`
* `missing_slot_per_report`
* `inconsistency_per_report`

## 7.2 Table 2（消融表）

`table_ablation.csv` 列名建议：

* `variant`（no_refine / no_citation / no_constrained / heuristic_policy / learned_policy / no_mask_head …）
* `budget_B`
* `slot_f1_micro`
* `ground_hit@0.1`
* `unsupported_sent_pct`
* `lat_total_ms`

## 7.3 Fig 2（Pareto）

输入格式：每条 run 输出 `metrics.json`，包含：

* `budget_B`
* `tokens_final`
* `lat_total_ms`
* `slot_f1_micro`

作图脚本契约：

* `eval/pareto.py --glob "artifacts/runs/*/metrics.json" --x tokens_final --y slot_f1_micro`

## 7.4 Fig 3（Grounding + 反事实）

输入：`counterfactual.csv`

* `cf_type`（permute_omega / swap_citation / random_split …）
* `slot_f1_micro`
* `ground_hit@0.1`
* `unsupported_sent_pct`

---

# 8. 反事实实验怎么“可执行”（可直接写进 eval/counterfactuals.py）

提供三个开关（跑出来就是论文结论）：

1. `permute_omega(tokens)`：对 tokens 的 `omega_box_mm` 做随机置换（保持 embedding 顺序不变）
2. `swap_citations(report)`：在同一 report 内随机交换句子 citations
3. `mask_sanity(new_tokens, gt_mask)`：统计新增 tokens 与 gt_mask overlap 的分位数（p50/p75/p90）

所有反事实都必须落盘：`artifacts/runs/<id>/counterfactual/*.json`

---

# 9. 里程碑（每一步都“验收即有论文证据”）

## M0：闭环跑通（1 周内必须做到）

* fixed tokens（level-0）+ template + citations + verifier
* 输出三件套
* **验收**：`unsupported` 相对 no-citation 显著下降

## M1：层级 tokens + refine（2–3 周）

* Path A 选择式 split
* 画 Pareto（正确性 vs tokens/latency）
* **验收**：边际收益递减（曲线有膝点）

## M2：RadGenome grounding 主评测（并行推进）

* 用句子↔mask 做 grounding hit-rate
* 做 permutation/citation swap
* **验收**：反事实显著劣化（因果链成立）([Nature][2])

## M3：learned split policy（bandit）

* learned > heuristic
* **验收**：相同预算下 (\mathcal{V})/Slot-F1 提升或 Pareto dominate

## M4：外部验证（加分但别拖主线）

* RAD-ChestCT 或别的中心数据做 robustness（可选）([CVIT - Center for Virtual Imaging Trials][5])

---

## 10. 可直接照着写的“最小 repo 骨架”

（与当前 draft 类似，但脚本接口固定为“产物驱动评测”）

```text
voxtoken/
  configs/
  data/
    adapters/ct_rate.py
    adapters/radgenome.py
    preprocess.py
  core/
    schemas.py          # Token/Evidence/Trace dataclasses
    geometry.py         # grid<->box(mm) mapping
  models/
    encoder3d.py
    vq.py
    tokenizer.py        # hierarchical tokens + omega mapping
    evidence_head.py
    policy.py
  generation/
    planner.py
    realizer.py
    constrained.py
  verify/
    verifier.py
    slot_extract.py
  runner/
    train_tokenizer.py
    train_evidence.py
    train_policy.py
    infer_refine.py
  eval/
    metrics.py
    pareto.py
    counterfactuals.py
    visualize_grounding.py
```

---

## 最关键的实现取舍（默认决策）

1. **论文主线：离散 VQ/RVQ tokens（VoxToken++ 名字才站得住）**
2. **工程启动：允许连续 tokens MVP，但接口必须按离散 token 设计**（`code=None`，其余完全一致）
3. **先用 Path A（选择式 split）拿 Pareto/反事实/learned policy 结果**，再决定是否上 Path B（按需编码）去拿更漂亮的 latency 叙事。

---

按本方案实施，将形成一条非常“NeurIPS reviewer 友好”的证据链：

* **tokenization 有 budget 的形式化定义**（不是 crop heuristics）
* **active split 有 learned policy + reward（verifier）**（不是“主观认为这里重要”）
* **grounding 有句子↔mask 的硬评测 + 反事实因果实验**（不是 attention 可视化）
* **unsupported 被 constrained+verifier 机制性压低**（不是靠评测后验抓）

下一步最优先的第一步不是训练大模型，而是：
**先把 schemas + runner + verifier + 三件套产物固定**，跑 50 个样本把 trace/metrics 全打通——这一步一旦完成，后续模块替换（VQ、更强 evidence、更强 policy）都不会“做了很多但无法证明”。

[1]: https://arxiv.org/abs/2403.17834 "[2403.17834] Developing Generalist Foundation Models ..."
[2]: https://www.nature.com/articles/s41597-025-05922-9 "Development of a large-scale grounded vision language dataset for chest CT analysis | Scientific Data"
[3]: https://zenodo.org/records/6406114 "RAD-ChestCT Dataset"
[4]: https://www.cancerimagingarchive.net/collection/lidc-idri/?utm_source=chatgpt.com "LIDC-IDRI - The Cancer Imaging Archive (TCIA)"
[5]: https://cvit.duke.edu/resource/rad-chestct-dataset/ "RAD-ChestCT Dataset - CVIT - Center for Virtual Imaging Trials"
