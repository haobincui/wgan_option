# Run five FiLM U-Net text arms directly from a common initialization

This ExecPlan is a living document. The sections Progress, Surprises &
Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as
work proceeds. Maintain it in accordance with `PLANS.md` at the repository
root.

## Purpose / Big Picture

Run a single-seed, four-fold comparison in which matched LP, shuffled LP,
no-text, BoW, and sentiment models train directly from an identical seeded
random initialization. This removes the previous
`parent -> no-text continuation -> frozen text branch` training lineage while
retaining the same FiLM U-Net and same-shape NoLP Critic. A completed run is
observable through 20 independently selected checkpoints, 20 frozen prediction
cells, 2,500 pair-level test rows, paired bootstrap comparisons, and a
self-contained report.

The branch-local CLI is
`python -m scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42`. Its formal
root is
`outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_5arm_seed42_exact_ttm_rolling_v1`.
Existing experiment roots are read-only.

## Progress

- [x] (2026-08-30 00:00Z) Freeze the direct five-arm design, seed, folds,
  architecture, training budget, estimands, output root, and interpretation.
- [x] (2026-08-30 00:00Z) Add the independent YAML experiment contract and
  this living ExecPlan.
- [x] (2026-08-30 12:20Z) Add the thin branch-local CLI, direct-training orchestrator, analysis,
  report, and regression tests without changing historical experiment meaning.
- [x] (2026-08-30 12:25Z) Pass static checks, targeted tests, the 20-cell one-epoch benchmark, and
  all prelaunch gates.
- [x] (2026-08-30 12:26Z) Launch the detached supervisor and record its PID, lock, journal stage,
  selected concurrency, and resource snapshot.
- [ ] Complete 20/20 training jobs, freeze the checkpoint allowlist, and
  complete 20/20 predictions.
- [ ] Produce the RQ1/representation statistics, reports, output SHA manifest,
  and passing terminal QA.

## Surprises & Discoveries

- Observation: the repository has no `v*` release tag, so this experiment's
  CLI and configuration schema are branch-local interfaces.
  Evidence: prior release-tag discovery recorded no matching local or remote
  tag.
- Observation: the completed ten-seed U-Net experiment contains suitable
  seed-42 parent and continuation results for historical context, but those
  jobs used a 30-epoch parent plus a continuation budget and are not
  compute-matched to a direct 240-epoch arm.
  Evidence: its frozen configuration and terminal artifacts record
  `parent_num_epochs: 30` and `num_epochs: 240`.
- Observation: a real-data prepare probe produced exactly 20 pending jobs,
  20 development overlays, one common Generator hash, one common Critic hash,
  and zero evaluation files. All four train/validation/test pair and session
  counts matched the frozen table.
  Evidence: `/tmp/direct5arm_prepare_probe.POLHFm/root` passed its own root
  validator; its pair-universe summary records 12 expected partition rows.
- Observation: a one-cell GPU execution probe completed the F1 matched-LP job
  with 382 train pairs, 144 validation pairs, and zero test samples. It
  instantiated exactly 827,745 G and 729,157 D parameters, retained the
  epoch-zero and best-learned checkpoints, and updated both models.
  Evidence: the selected run under the prepare-probe root completed epoch 1
  without OOM, NaN, or test-loader materialization.
- Observation: the focused and relevant regression stack currently passes
  56/56 tests, including FiLM U-Net, full-state, overlay, benchmark-hardening,
  and historical-profile isolation contracts.
  Evidence: py312 `unittest` run at 2026-08-30 12:22Z.
- Observation: the complete 20-cell one-epoch benchmark passed at the primary
  10-workers/GPU concurrency. Peak GPU memory was 3,895 MiB on GPU0 and
  3,803 MiB on GPU1; peak host RAM was 13.36%, and all 44 telemetry samples
  were healthy.
  Evidence: `..._control/benchmark_result.json`, payload SHA
  `b19fee75c1f9dca11459ad395034f114153da8d5dac664d3f827dcfe87a9204c`.
- Observation: the mandatory verification helper exits before project checks
  because this research repository has no `make format` target.
  Evidence: `.agents/skills/code-change-verification/scripts/run.sh` exited 2
  with `No rule to make target 'format'`; Ruff format/check, `py_compile`,
  `git diff --check`, and the 56-test native stack passed independently.
- Observation: the detached formal supervisor is live as PID/SID `2988401`.
  It froze 10 workers/GPU, prepared the 20-job formal registry, and started all
  20 direct jobs in one wave; the first live snapshot showed all 20 statuses as
  `running`, GPU memory 2,239/1,606 MiB, and no test inputs.
  Evidence: `..._control/pipeline.pid`, `pipeline.lock`,
  `pipeline_journal.json`, and formal `registry/task_registry.json`.

## Decision Log

- Decision: train every arm independently with
  `training.protocol: independent_random_init_v1`, while requiring the initial
  Generator and Critic state hashes to match within each fold.
  Rationale: the experiment is intended to remove the backbone/continuation
  lineage without confounding the comparison with different initial weights.
  Date/Author: 2026-08-30 / Codex.
- Decision: keep the no-text arm structurally identical and feed it a strict
  zero 1024-vector rather than deleting the text encoder or FiLM modules.
  Rationale: equal parameter counts isolate sample-varying text information.
  Date/Author: 2026-08-30 / Codex.
- Decision: use one seed and label all results
  `retrospective_rolling_development_single_seed_descriptive`.
  Rationale: paired fold/session resampling measures test-panel uncertainty but
  cannot establish robustness across random initialization.
  Date/Author: 2026-08-30 / Codex.
- Decision: freeze every checkpoint before any test loader is opened and use
  one MC64 noise bank across arms.
  Rationale: this prevents test-driven checkpoint selection and makes paired
  forecast errors comparable.
  Date/Author: 2026-08-30 / Codex.

## Outcomes & Retrospective

Implementation and formal execution are not yet complete. Update this section
with the selected epoch distribution, five-arm ranking, RQ1 and representation
Holm results, persistence comparisons, benchmark peaks, terminal-QA result,
and final output-manifest SHA. Do not infer completion from an idle GPU or a
quiet log.

## Context and Orientation

The Generator mode `film_unet_mask_coords_v1` consumes four 16 by 16 spatial
channels: masked current volatility, support mask, normalized strike
coordinate, and normalized maturity coordinate. LP1024 passes through 256 and
128 dimensions and controls six FiLM modulations. The c32 Generator has 827,745
parameters. The Critic mode `lp_disabled_same_shape_v1` retains the historical
text-branch parameter shape but zeros its encoded text feature exactly; it has
729,157 parameters. The combined model therefore has 1,556,902 parameters.

The source data is the exact-TTM 16 by 16 dataset. The maturity axis is
`[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]`. Each market pair is one sample;
articles are deduplicated before LP and sentiment aggregation. BoW vocabulary
and sentiment scaling are fit on the fold's training pairs only. Shuffled LP
uses a deterministic derangement within each split, so it has no fixed point
and never crosses train, validation, or test boundaries.

The four rolling test folds are 2023Q1 through Q4. Only the five-minute
alignment dataset is used, and the forecast target is always the future
five-minute surface. Across folds, each arm has 500 test pairs and 148 CME
sessions. Five arms times four folds times seed 42 yields exactly 20 training
jobs and 20 prediction cells.

## Plan of Work

Add a thin CLI under `scripts/rq3` that resolves the independent configuration,
prepares immutable data/model/runtime manifests, and materializes one
`direct_arms` registry. Each job resets Python, NumPy, Torch CPU/CUDA, and
DataLoader randomness from the common fold/seed initialization contract. It
then trains its own arm for at most 240 epochs with minimum 30, patience 20,
and its own scheduler and validation selection. There is no parent state,
continuation state, branch recipe, LR replay, or branch-final checkpoint rule.

Training uses G/D learning rates `5e-7`, floor `5e-8`, zero warmup, batch 16,
five critic steps, validation MC16, Gaussian32 noise, `raw_joint` support,
`current_support_masked`, and identity residual output. Every job selects its
own `best_learned` checkpoint using `val_hybrid_score`. Freeze an evaluation
allowlist only after all 20 checkpoint files, job specifications, sizes, and
SHA-256 values pass verification. Only then construct test loaders and produce
MC64 predictions from a shared per-pair noise bank.

Postprocessing reports pooled and equal-fold MAE, persistence-relative change,
fold results, selected epoch, final learning rates, and training diagnostics.
RQ1 compares LP with no-text and shuffled LP; the representation family
compares LP with BoW and sentiment. Each family uses 10,000 paired
`fold -> CME-session` bootstrap replicates and Holm-2. A separate Holm-5 family
compares all arms with persistence. The old seed-42 parent and continuation
results may appear only in a clearly separated, non-compute-matched historical
table.

## Concrete Steps

Run from the repository root with the py312 environment:

    /home/haobin_cui/.conda/envs/py312/bin/python -m scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42 prepare --config configs/rq3/news_first_vol_film_unet_direct_5arm_seed42.yaml --output-dir <temporary-root>
    /home/haobin_cui/.conda/envs/py312/bin/python -m scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42 dry-run --config configs/rq3/news_first_vol_film_unet_direct_5arm_seed42.yaml --output-dir <temporary-root>
    /home/haobin_cui/.conda/envs/py312/bin/python -m pytest <targeted-direct-experiment-tests>
    ruff format --check <changed-python-files>
    ruff check <changed-python-files>
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile <changed-python-files>
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

Before creating the formal root, run all 20 cells for one epoch. Assign F1/F3
to GPU 0 and F2/F4 to GPU 1. Freeze 10 workers per GPU only when peak memory is
below 20 GiB on each GPU, host RAM stays below 85%, both G/D update, and all
metrics remain finite. Otherwise benchmark six workers per GPU and run two
waves; a second gate failure prevents formal-root creation.

After every gate passes, launch one detached supervisor:

    nohup setsid env PYTHONPATH=src:. \
      /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42 \
      run-pipeline --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_5arm_seed42.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_5arm_seed42_exact_ttm_rolling_v1 \
      > outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_5arm_seed42_exact_ttm_rolling_v1_control/pipeline.log \
      2>&1 < /dev/null &

Record the supervisor PID/SID, lock path, control log, journal stage, job and
prediction counts, selected workers, and resource peaks in this document.

## Validation and Acceptance

The resolved configuration must instantiate exactly 20 unique jobs and 20
prediction cells, with 10 jobs assigned to each GPU. All five epoch-zero
Generator hashes and all five epoch-zero Critic hashes must agree within each
fold, and every completed job must show finite metrics plus changed G/D
parameters. The model parameter counts must equal 827,745 and 729,157. NoLP
Critic output must be invariant to text and expose zero text gradient; the
Generator's FiLM path must expose finite nonzero text-projection gradients.

The no-text overlay must be exactly zero. LP, BoW, and sentiment coverage must
be 100%; shuffle must have no fixed points or cross-split mapping. Prepared
fold counts, support hash, grid axis/hash, and five-minute target must match the
configuration. Before evaluation freeze, test-loader and prediction counters
must both be zero. After freeze, every arm must use the same MC64 noise-bank
hash for a given pair.

Terminal acceptance requires 20 completed jobs, 20 frozen predictions, exactly
2,500 pair-metric rows, complete RQ1/representation/bootstrap/Holm outputs,
Markdown and self-contained HTML reports, a passing `qa.json`, and a complete
output SHA manifest. Every statistical output and report must carry the
single-seed descriptive interpretation and omit cross-seed support claims.

## Idempotence and Recovery

The supervisor holds an exclusive lock and writes its PID and stage journal.
Resume may skip a job only when the resolved config, input/data/grid/support
hashes, source manifest, job specification, selected checkpoint, artifact size,
and artifact SHA all match. Partial attempts remain inspectable but are not
treated as complete. Any mismatch fails closed. A terminal-complete root is
strictly read-only, and no command deletes or mutates older experiment roots.

## Artifacts and Notes

Core artifacts are the resolved configuration, dataset/grid/profile manifests,
20-job registry, per-job specifications and status, checkpoint allowlist, 20
prediction manifests, 2,500-row pair-metrics table, arm ranking, bootstrap and
Holm tables, resource summary, Markdown/HTML reports, `qa.json`, pipeline log,
PID/journal files, and complete output SHA manifest.

## Interfaces and Dependencies

The branch-local CLI exposes `benchmark`, `prepare`, `dry-run`, `worker`,
`launch`, `freeze-evaluation`, `predict`, `postprocess`, `qa`, `status`, and
`run-pipeline`. The authoritative training protocol field is
`training.protocol: independent_random_init_v1`; no parent or branch-recipe
schema is accepted for this experiment. The implementation reuses the existing
exact-TTM data and FiLM U-Net/NoLP model modes and introduces no third-party
dependency.
