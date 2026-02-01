# Proof Audit
Generated at (UTC): `2026-02-01T20:21:33.538679+00:00`

## Inputs
- plan: `docs/plan.md`
- ledger: `docs/experiment.md`
- results_dir: `.rd_queue/results`

## Summary
- total_claims: **58**
- checked_claims: **58**
- proved_checked: **58**
- not_proved_checked: **0**
- pending_claims: **0**

## Claims
| Claim | Text | Evidence | Ledger smoke/full | Results smoke/full | Status |
|---|---|---|---|---|---|
| C0001 | Baseline smoke runs and produces `run.json` + `summary.json` that conform to `docs/results_contract.md`. | E0000 | Y/Y | Y/Y | PROVED |
| C0002 | Unified eval runs on baseline `run.json` and produces `metrics.json` + `metrics.jsonl` that conform to `docs/results_contract.md`. | E0001 | Y/Y | Y/Y | PROVED |
| C0003 | `infer_refine` runs and produces a non-empty report where every sentence has citations. | E0100 | Y/Y | Y/Y | PROVED |
| C0004 | Unified eval can read `tokens_used`/`verifier_score` from an `infer_refine` run and write them to `metrics.json(l)`. | E0101 | Y/Y | Y/Y | PROVED |
| C0005 | Heuristic refinement (split) runs and produces a non-empty `trace` with at least one split, while staying within token budget. | E0200 | Y/Y | Y/Y | PROVED |
| C0006 | Policy training produces a reusable checkpoint, and inference can load it to drive split scoring. | E0300 | Y/Y | Y/Y | PROVED |
| C0007 | No-citation ablation produces measurable unsupported sentences (unsupported_rate > 0) under unified eval. | E0400 | Y/Y | Y/Y | PROVED |
| C0008 | No-constrained ablation allows overclaim and is penalized by the verifier (verifier_score decreases and overclaim issues appear). | E0401 | Y/Y | Y/Y | PROVED |
| C0009 | Tokenizer training writes a deterministic checkpoint, and inference can load it to assign non-null token codes. | E0500 | Y/Y | Y/Y | PROVED |
| C0010 | Evidence head training writes a deterministic checkpoint, and inference can load it to change finding types based on token codes. | E0600 | Y/Y | Y/Y | PROVED |
| C0011 | Unified eval computes `slot_f1` from `run.json` (plan vs extracted slots) and returns ~1.0 for constrained baseline inference. | E0700 | Y/Y | Y/Y | PROVED |
| C0012 | Inference records `latency_ms.total` in `run.json`, and unified eval propagates it to `metrics.latency_ms.total` (> 0). | E0701 | Y/Y | Y/Y | PROVED |
| C0013 | `reproduce` can re-run an experiment by `E####` by parsing `docs/experiment.md` and executing its `1GPU script`. | E0800 | Y/Y | Y/Y | PROVED |
| C0014 | Counterfactual evaluation can measure how `unsupported_rate` changes when citations are removed. | E0801 | Y/Y | Y/Y | PROVED |
| C0015 | Data ingest + preprocess produces a processed manifest where every case has a deterministic `split` field. | E0802 | Y/Y | Y/Y | PROVED |
| C0016 | CT-RATE ingest+preprocess works from `/data/ct_rate` (or `/data/CT-RATE`) and produces a processed manifest with existing report/volume paths. | E0803 | Y/Y | Y/Y | PROVED |
| C0017 | CT-RATE train split ingest+preprocess can produce a processed manifest with existing report paths (volume paths may be empty). | E0804 | Y/Y | Y/Y | PROVED |
| C0018 | `infer_refine` can run by selecting a case from a JSONL manifest and (when available) loading a small NIfTI volume. | E0805 | Y/Y | Y/Y | PROVED |
| C0019 | `infer_refine` can also run on CT-RATE train manifests that have reports but no resolvable volumes (falls back to dummy volume). | E0806 | Y/Y | Y/Y | PROVED |
| C0020 | Batch inference can run over multiple CT-RATE validation cases with real volumes and emit an aggregated `metrics.jsonl`. | E0810 | Y/Y | Y/Y | PROVED |
| C0021 | Batch inference can run over the full CT-RATE validation manifest (all rows) when `--max-cases 0`. | E0811 | Y/Y | Y/Y | PROVED |
| C0022 | CT-RATE ingest can join predicted multi-abnormality labels into the manifest (`labels_pos`). | E0820 | Y/Y | Y/Y | PROVED |
| C0023 | Label evaluation can compute per-case multi-label metrics (`precision/recall/f1`) from `run.json` + CT-RATE predicted labels. | E0821 | Y/Y | Y/Y | PROVED |
| C0024 | Batch label evaluation can produce `label_metrics.jsonl` for a CT-RATE subset. | E0822 | Y/Y | Y/Y | PROVED |
| C0025 | Label-conditioned report generation from CT-RATE `labels_pos` can achieve `f1>=0.99` on a labeled case. | E0830 | Y/Y | Y/Y | PROVED |
| C0026 | Batch label-conditioned report generation from CT-RATE `labels_pos` can achieve `f1>=0.99` for at least 10 labeled cases. | E0831 | Y/Y | Y/Y | PROVED |
| C0027 | `infer_refine` writes sidecar artifacts (`final_report.txt`, `evidence_graph.json`, `trace.jsonl`) for paper-facing exports. | E0832 | Y/Y | Y/Y | PROVED |
| C0028 | Batch inference over 50 CT-RATE cases writes the full artifact bundle per case (run.json + report/evidence/trace sidecars). | E0833 | Y/Y | Y/Y | PROVED |
| C0029 | Counterfactual citation swap increases `unsupported_rate` when unsupported is defined as "citation does not support the slot". | E0834 | Y/Y | Y/Y | PROVED |
| C0030 | Grounding pipeline runs on RadGenome-synth and unified eval reports GT-based grounding metrics. | E0901 | Y/Y | Y/Y | PROVED |
| C0031 | Learned split policy beats heuristic on grounding (higher `ground_mean_iou` under the same budget) on RadGenome-synth. | E0902 | Y/Y | Y/Y | PROVED |
| C0032 | Counterfactual causality holds on GT grounding: permuting Ω or swapping citations reduces grounding vs base deterministically. | E0903 | Y/Y | Y/Y | PROVED |
| C0033 | Tokenizer training emits codebook-usage diagnostics (perplexity) and `recon_error` has sufficient dynamic range for split features. | E0904 | Y/Y | Y/Y | PROVED |
| C0034 | Evidence graph sidecar is traceable: every `EvidenceNode.supported_token_ids` maps to tokens with `omega_box_mm` inside `evidence_graph.json`. | E0905 | Y/Y | Y/Y | PROVED |
| C0035 | Verifier is deterministic for the same input/config, and issues are localized to spans + related tokens/evidence. | E0906 | Y/Y | Y/Y | PROVED |
| C0036 | CT-RATE TS (lung nodules) grounding GT manifest can be built from real CT volumes + TS masks into a runnable per-case GT-box dataset. | E0907 | Y/Y | Y/Y | PROVED |
| C0037 | Policy training can fit a split policy from CT-RATE TS grounding rewards and write a reusable checkpoint with non-default weights. | E0908 | Y/Y | Y/Y | PROVED |
| C0038 | CT-RATE TS grounding benchmark runs fixed/heuristic/learned tokenization across budgets and outputs per-case + aggregate metrics deterministically. | E0909 | Y/Y | Y/Y | PROVED |
| C0039 | Paper export script generates Table1/2 and Fig2/3 artifacts from benchmark outputs deterministically. | E0910 | Y/Y | Y/Y | PROVED |
| C0040 | Verifier inconsistency rule detects plan-vs-report slot mismatches deterministically. | E0916 | Y/Y | Y/Y | PROVED |
| C0041 | Inference refinement implements the marginal stop rule (ΔV/Δ\|T\| < tau) to stop early under no-improvement splits. | E0917 | Y/Y | Y/Y | PROVED |
| C0042 | Constrained decoding removes overclaim sentences even if the generator attempts to add them. | E0918 | Y/Y | Y/Y | PROVED |
| C0100 | Papertrack (pseudo-GT): TS lung_nodules case index + GT manifest can be built with non-empty GT boxes and deterministic splits. | E0911, E0912 | Y/Y | Y/Y | PROVED |
| C0101 | Papertrack (pseudo-GT): policy dataset + policy training + benchmark + bootstrap improvement gate are runnable end-to-end. | E0913, E0914 | Y/Y | Y/Y | PROVED |
| C0102 | Papertrack: export CI tables/plots from papertrack benchmark outputs deterministically. | E0915 | Y/Y | Y/Y | PROVED |
| C0103 | CT-RATE valid ingest+preprocess yields a deterministic 70/30 train/val split over the available real volumes on this machine. | E0920, E0921 | Y/Y | Y/Y | PROVED |
| C0104 | Effusion pseudo-GT manifests (pleural/pericardial) can be built from CT-RATE valid volumes + TotalSeg masks with non-empty GT boxes. | E0922, E0923 | Y/Y | Y/Y | PROVED |
| C0105 | Effusion policy datasets (pleural/pericardial, split=train) can be built from GT boxes into a runnable dataset.jsonl. | E0924, E0925 | Y/Y | Y/Y | PROVED |
| C0106 | Effusion policy training + grounding benchmark (split=val) runs fixed/heuristic/learned across budgets and emits non-empty metrics+summary. | E0926, E0927 | Y/Y | Y/Y | PROVED |
| C0107 | Papertrack (GT): RadGenome-ChestCT lung nodule (hi32 token-space) manifest can be built for full-eligible cases with non-empty GT boxes and deterministic splits. | E0986 | Y/Y | Y/Y | PROVED |
| C0108 | Papertrack (GT): Torch reward-policy training (STOP action; reward regression) writes reusable checkpoints with `model.pt` and non-default weights (multi-seed). | E1042 | Y/Y | Y/Y | PROVED |
| C0109 | Papertrack (GT): RadGenome lung nodule grounding benchmark (reward stop-threshold) runs fixed/heuristic/learned/random/oracle across budgets and emits deterministic summaries (multi-seed). | E1043 | Y/Y | Y/Y | PROVED |
| C0110 | Papertrack (GT): Learned policy achieves a statistically significant improvement over random at budget 32 on RadGenome lung nodule grounding (ΔIoU >= 0.017 and paired Δ CI_low >= 0). | E1044 | Y/Y | Y/Y | PROVED |
| C0111 | Papertrack (GT): CT-RATE pleural_effusion GT manifest (hi32 token-space) can be built from real CT volumes + TotalSeg masks with non-empty GT boxes and deterministic splits. | E1045 | Y/Y | Y/Y | PROVED |
| C0112 | Papertrack (GT): Oracle+STOP policy dataset for CT-RATE pleural_effusion can be built from GT boxes into a runnable dataset.jsonl. | E1047 | Y/Y | Y/Y | PROVED |
| C0113 | Papertrack (GT): Torch reward-policy training (reward regression; weight_decay=1e-4) writes reusable checkpoints with `model.pt` and non-default weights (multi-seed) for CT-RATE pleural_effusion. | E1048 | Y/Y | Y/Y | PROVED |
| C0114 | Papertrack (GT): CT-RATE pleural_effusion grounding benchmark (reward stop-threshold; hi32) runs fixed/heuristic/learned/random/oracle across budgets and emits deterministic summaries (multi-seed). | E1049 | Y/Y | Y/Y | PROVED |
| C0115 | Papertrack (GT): Learned policy achieves a statistically significant improvement over random at budget 32 on CT-RATE pleural_effusion grounding (ΔIoU >= 0.017 and paired Δ CI_low >= 0). | E1050 | Y/Y | Y/Y | PROVED |

## Not Proved

- (empty)
