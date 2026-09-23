# Build the News-first Vol label-reliability experiment

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. It is maintained in accordance with `PLANS.md`.

## Purpose / Big Picture

The current News-first Vol result improves pair-balanced validation MAE over persistence by only about `6.53e-7`, while a read-only raw-trade pilot found much larger target-label reconstruction uncertainty. This change creates a branch-local, reproducible experiment that measures that uncertainty before training and tests whether filtering or softly down-weighting unreliable training labels improves out-of-time performance. A user can prepare an immutable experiment root, run the audit bootstrap, exercise the exact GPU schedule without optimization, launch three conditional training stages, and render a self-contained report. Q3 is unavailable to arm selection and Q4 is unavailable to loaders, prediction, selection, and reporting.

## Progress

- [x] (2026-08-20 18:10Z) Confirmed the branch-local compatibility boundary and frozen folds, arms, weighting formula, gates, and maximum 96-job matrix.
- [x] (2026-08-20 18:20Z) Located the reusable News-first Vol training, inference, resource-monitoring, and fail-closed lineage helpers.
- [ ] Implement the immutable orchestrator, bootstrap-to-training profile materialization, conditional registry expansion, and GPU preflight.
- [ ] Implement fold-only selection statistics, gated one-SE choice, restricted Q3 evaluation, and terminal no-winner behavior.
- [ ] Implement the self-contained HTML report and focused tests.
- [ ] Run focused unit tests, Ruff, `py_compile`, `git diff --check`, and prepare/dry-run only after integration approval.

## Surprises & Discoveries

- Observation: the repository has no release tag, and all neighboring RQ3 experiment interfaces are untracked branch-local files.
  Evidence: the release-tag lookup returned an empty value and `git status --short` lists the RQ3 sweep modules as untracked.
- Observation: the bootstrap surfaces must remain audit-only; the trainer consumes only a six-column pair profile, never a bank of reconstructed surfaces.
  Evidence: the frozen training contract accepts `(tolerance_minutes, fold_id, pair_id, included, normalized_label_weight, reliability_score)` plus independent manifest and profile hashes.

## Decision Log

- Decision: use a new schema and immutable root with no compatibility shim.
  Rationale: the CLI and persisted records are branch-local and unreleased; reusing an older experiment root would weaken provenance.
  Date/Author: 2026-08-20 / Codex.
- Decision: materialize only the 48 stage-one jobs before selection, then append exactly 24 stage-two and 24 stage-three jobs for baseline A and the immutable winner.
  Rationale: this preserves the maximum 96-job contract and makes it impossible to infer nonselected arms on Q3.
  Date/Author: 2026-08-20 / Codex.
- Decision: a 24-slot/GPU root that breaches 20 GB peak GPU memory or 85% host RAM terminates with an instruction to create a new 12-slot/GPU root; the resolved config is never mutated in place.
  Rationale: formal runtime configuration must be frozen after preflight and resume must remain fail-closed.
  Date/Author: 2026-08-20 / Codex.

## Outcomes & Retrospective

Implementation is in progress. Formal launch occurs only after integrated verification and a passing resource preflight; the root agent decides and executes it. The final update will record the public APIs, test evidence, prepared/dry-run root if approved, and any remaining infrastructure limitation.

## Context and Orientation

`scripts/rq3/news_first_vol_training.py` owns the existing training-process boundary and artifact validation. `scripts/rq3/news_first_vol_comparison_analysis.py` owns checkpoint inference and pair metrics. `src/wgan_option/surface_generation/label_reliability.py` owns the raw-trade Bayesian bootstrap audit. The new `scripts/rq3/news_first_vol_label_reliability.py` coordinates those pieces without importing torch or the bootstrap implementation until the corresponding action runs.

There are four chronological folds. F1 trains before 2022-07-01 and validates in 2022Q3; F2 trains before 2022-10-01 and validates in 2022Q4; F3 trains before 2023-01-01 and validates in 2023Q1; F4 trains before 2023-04-01 and validates in 2023Q2. Q3 is 2023Q3 and can be evaluated only after the winner and all eligible F4 checkpoint hashes are frozen. Q4 begins 2023-10-01 and is forbidden.

Arm A is the unchanged pair-balanced baseline. Arm B filters training pairs with fewer than 16 joint strict-support cells. Arm C retains all pairs and applies the reliability weight. Arm D applies both. For fold `f`, `tau_f` is the training-pair 75th percentile of estimable `u_i`. With `c_i`, `q_i`, `m_i`, `u_i`, `v_i`, and `h_i` supplied by the audit, the reliability calculation is:

    s_i = min(1, sqrt(c_i / 16))
    d_i = min(1, sqrt(q_i / 8))
    e_i = min(1, sqrt(m_i / 4))
    b_i = 1 / (1 + (u_i / tau_f)^2), or zero when u_i is unestimable
    R_i = (s_i * d_i * e_i * b_i * v_i * h_i)^(1/6)
    r_i = 0.5 + 0.5 * R_i

Included training-pair weights are normalized to mean one and must remain in `[0.5, 2]`. Validation and later evaluation weights remain one.

## Plan of Work

Add `configs/rq3/news_first_vol_label_reliability.yaml` with the frozen data, model, audit, fold, selection, and runtime contracts. Add the orchestrator with `prepare`, `bootstrap`, `dry-run`, `worker`, `launch`, and `postprocess` actions. Preparation freezes source/config/code hashes and the stage-one Cartesian matrix. Bootstrap lazily calls the audit core, verifies that every audit row is pre-Q3, writes four arm-specific six-column manifests per tolerance and fold, and records independent file and canonical-profile hashes.

Stage one runs 48 Regression 5m jobs. Analysis evaluates their best-learned checkpoints only on their matching fold validation intervals and performs 10,000 paired seed-to-fold-to-CME-session bootstrap draws for B, C, and D versus A, followed by Holm correction. A candidate must have negative mean log ratio, a confidence-interval upper bound below zero, Holm-adjusted p below .05, at least three nonworse folds, and at least two nonworse seeds. Among passing candidates within the best candidate's one bootstrap standard error, priority is C, B, D. No passing candidate creates an immutable terminal no-winner result and no later jobs.

After a winner is frozen, append Regression 30m and WGAN 5m jobs for A and the winner only. These stages do not reselect. Once all F4 checkpoint hashes are frozen, postprocessing may evaluate A and the winner on Q3. Q4 is never passed to an evaluator.

## Concrete Steps

Run from the repository root:

    PYTHONPATH=src:. python -m unittest \
      tests.test_scripts.test_rq3_news_first_vol_label_reliability \
      tests.test_scripts.test_rq3_news_first_vol_label_reliability_analysis

    ruff check scripts/rq3/news_first_vol_label_reliability.py \
      scripts/rq3/news_first_vol_label_reliability_analysis.py \
      scripts/rq3/news_first_vol_label_reliability_report.py \
      tests/test_scripts/test_rq3_news_first_vol_label_reliability.py \
      tests/test_scripts/test_rq3_news_first_vol_label_reliability_analysis.py

    python -m py_compile scripts/rq3/news_first_vol_label_reliability*.py
    git diff --check

After integration approval only:

    python scripts/rq3/main.py train-news-first-vol-label-reliability prepare \
      --config configs/rq3/news_first_vol_label_reliability.yaml \
      --output-dir outputs/experiments/rq3_news_first_vol_label_reliability_q097_103_ttm07_38_v1

    python scripts/rq3/main.py train-news-first-vol-label-reliability bootstrap \
      --config configs/rq3/news_first_vol_label_reliability.yaml \
      --output-dir outputs/experiments/rq3_news_first_vol_label_reliability_q097_103_ttm07_38_v1 \
      --resume

    python scripts/rq3/main.py train-news-first-vol-label-reliability dry-run \
      --config configs/rq3/news_first_vol_label_reliability.yaml \
      --output-dir outputs/experiments/rq3_news_first_vol_label_reliability_q097_103_ttm07_38_v1 \
      --resume

## Validation and Acceptance

The config validator rejects any altered fold, arm, seed, model, LR, mask, residual, noise, or Q4 contract. Preparation creates exactly 48 unique stage-one jobs and no stage-two/three jobs. Bootstrap creates unique six-column manifests whose canonical profile hashes agree with every training YAML. Tests prove B/D retention gates, bounded normalized weights, deterministic bootstrap selection, Holm correction, C-over-B-over-D one-SE priority, terminal no-winner behavior, conditional expansion to 96 jobs, cross-balanced GPUs, Q3 checkpoint allowlisting, Q4 rejection, source/config/code tamper rejection, and resume idempotence.

## Idempotence and Recovery

Every action validates `resolved_config.sha256`, source hashes, code hashes, job-spec hashes, manifest hashes, and artifact hashes before writing new state. `--resume` skips only hash-valid completed work. A failed 24-slot preflight never rewrites the resolved config; the user prepares a distinct 12-slot root. Selection is immutable: an existing selection file may be reread only if its self-hash and recomputed content match. Postprocessing never launches workers.

## Artifacts and Notes

Expected durable outputs include `bootstrap/`, `reliability_profiles/`, `registry/jobs.json`, `label_reliability_stage_status.json`, `label_reliability_selection.json`, `q3_checkpoint_manifest.csv`, fold/Q3 pair metrics, bootstrap contrasts, Holm decisions, resource summaries, and `report/label_reliability_report.html`.

## Interfaces and Dependencies

The orchestrator exports `prepare_label_reliability_experiment`, `run_label_reliability_bootstrap_action`, `run_label_reliability_worker`, `launch_label_reliability_experiment`, `postprocess_label_reliability_experiment`, and `run_news_first_vol_label_reliability`. The analysis exports `seed_fold_session_bootstrap`, `build_arm_comparisons`, `select_reliability_winner`, and `run_label_reliability_analysis`. The report exports `render_label_reliability_report`. The audit core is imported only inside the bootstrap action via `run_label_reliability_bootstrap(config, output_dir, resume=False)`.
