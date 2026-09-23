# Implement the FiLM/NoLP 10-seed RQ1–RQ3 experiment

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. It is maintained in accordance with `PLANS.md` in the repository root.

## Purpose / Big Picture

The completed change repairs the RQ2 external-feature coverage contract and adds a single reproducible experiment that trains the exact same legacy-size FiLM Generator and NoLP Critic across ten random seeds, four expanding temporal folds, and the 5-minute and 30-minute news-alignment datasets. RQ1, RQ2, and RQ3 then reuse one frozen prediction bank instead of training overlapping model sets. A user can observe success in a terminal experiment root containing 400 completed jobs, 400 frozen prediction manifests, the three research analyses, a self-contained report, and an all-file SHA manifest. The formal run is controlled by one background supervisor and is resumable only while every frozen input and artifact hash remains unchanged.

## Progress

- [x] (2026-08-22 14:00Z) Audited the existing RQ1/RQ2/RQ3 data and training paths, identified the RQ2 pre-support feature coverage bug, and froze the ten seeds, four folds, two tolerances, six arms, and exact-TTM grid.
- [x] (2026-08-22 14:10Z) Chose a branch-local compatibility boundary because the repository has no `v*` release tag; existing formal roots remain read-only and weights-only checkpoints remain inference-only.
- [x] (2026-08-22 14:18Z) Installed the project-declared py312 dependencies, including `openai`, `nbformat`, and `nbconvert`, without making an API request.
- [x] (2026-08-22 15:35Z) Repaired RQ2 dual-universe feature generation and proved all four real-data folds prepare and reuse correctly in `/tmp/rq2-dual-universe-prepare.972Chs` without modifying the failed root or hard-linked source split.
- [x] (2026-08-22 16:05Z) Implemented pair-level LP/BoW/sentiment/shuffle inputs and versioned full-state continuation; generated and strictly reloaded all 40 development overlays and exercised a real F1/5m loader with 382 train and 144 validation pairs.
- [x] (2026-08-22 23:35Z) Implemented the 400-job orchestrator, gated 400-cell prediction freeze, RQ1/RQ2/RQ3 analyses, self-contained report, terminal QA, two-phase terminal commit, common-noise contract, process/job locks, and interruption-safe resume guards.
- [x] (2026-08-22 23:45Z) Ran 158 focused and related regression tests successfully, plus Ruff format/check, py_compile, and diff-check. The mandatory verification script was also run and stopped only because this repository has no `make format` target; the native checks passed.
- [x] (2026-08-22 23:55Z) Passed the 36-job all-legacy dual-A30 one-epoch benchmark: 36/36 completed in 75.7 seconds, peak memory was 7.64 GiB on each GPU, peak host RAM was 17.81%, and formal concurrency froze at 18 workers per GPU.
- [x] (2026-08-23 00:20Z) Completed the disposable v1 400-cell one-epoch smoke: 400/400 jobs updated both networks with finite learned metrics and no test loader.
- [x] (2026-08-23 01:10Z) Audited the stopped v1 supervisor. It completed 80/80 parents, then failed before any continuation completed because CUDA-mapped RNG tensors were passed to `torch.set_rng_state`; every explicit-grid parent checkpoint also lacked a non-empty architecture-profile SHA and therefore failed the production inference loader. The v1 root and its source config are now immutable failure evidence.
- [x] (2026-08-23 12:45Z) Fixed CUDA-to-CPU RNG restoration in the shared full-state core, proved exact next-update equivalence on a real A30, and added deterministic legacy architecture-profile lineage to the RQ123 orchestrator, training configs, full-state contracts, jobs, and checkpoints.
- [x] (2026-08-23 13:05Z) Cut a branch-local v2 config, experiment kind, and output root; added a fail-closed config/output-root boundary so the v2 CLI cannot write to v1; and added a mandatory GPU parent-to-continuation recovery/checkpoint-load canary before matrix smoke or formal prepare.
- [x] (2026-08-23 13:20Z) Passed 71/71 RQ123 tests, including the real learned-checkpoint production-loader regression and CUDA restoration tests, plus focused Ruff, `py_compile`, and diff checks. The mandatory verification script was rerun and again stopped only because this repository has no `make format` target.
- [ ] Run the fresh v2 36-job benchmark, mandatory one-cell GPU recovery canary, and 400-cell matrix smoke. The v2 formal root must remain absent until all three gates pass.
- [ ] Complete the 400 jobs and 400 predictions, validate terminal hashes, and summarize the retrospective results.

## Surprises & Discoveries

- Observation: the old RQ2 loader asks for external features before applying the raw-joint support filter, while the feature artifact was built only from the post-support lineage. A text-valid pair with zero joint support therefore fails as “missing feature” before the loader can correctly discard it.
  Evidence: the failing pair `74bba154c5a4a2b4d7c3` has current support zero and target support 66, so it belongs to feature coverage but not the final fit universe.
- Observation: ten seeds increase initialization coverage but do not increase the number of independent scheduled events or market-jump episodes.
  Evidence: the planned 30-minute scheduled primary window has about 32 pairs from 20 releases, while the frozen jump detector has about 31 all-tier pairs before control matching; explicit sample-size gates are therefore part of the analysis contract.
- Observation: the current free disk budget is much tighter than GPU or host memory.
  Evidence: pre-implementation inspection showed about 106 GiB free, so selected-only checkpoint retention and a 30 GiB post-projection floor are mandatory.
- Observation: training overlays must exclude test pairs entirely, but evaluation overlays cannot simply reuse the training artifacts.
  Evidence: BoW and sentiment evaluation overlays need the frozen train-fold vocabulary/scaler, while shuffled LP needs a separate deterministic test-partition derangement. The evaluation overlay builder is therefore gated behind the 400-checkpoint allowlist.
- Observation: a narrow hand-maintained source ledger did not cover the full lazy import closure used by training and prediction.
  Evidence: the initial ledger listed 19 files while runtime paths imported additional model, loader, calendar, and surface modules. The new experiment freezes all `src/**/*.py` plus its direct script dependencies and rejects a changed job-universe SHA at every resume boundary.
- Observation: the narrow scheduled window and frozen market-jump source require explicit estimability downgrades rather than hard failures or optimistic inference.
  Evidence: the real 30-minute panel yields 30 matched pairs / 18 sessions / four folds for `[0,+30]`, 28 / 17 / four folds for `[-10,+20]`, and only 6 / 5 / two folds for `[0,+5]`. The frozen jump source has 31 candidates but only 21 matchable pairs across 20 sessions, so it is explicitly reported as `not_estimable_due_to_coverage`.
- Observation: a save-only one-epoch matrix smoke does not exercise dynamic full-state restore or the production explicit-grid checkpoint loader.
  Evidence: v1 passed all 400 save-dynamic smoke cells but then failed its first continuation restore; independently, all 80 v1 parent Generator checkpoints were unreadable because `architecture_profile_sha256` was empty.

## Decision Log

- Decision: create a new `scripts.rq123.news_first_vol_film_nolp_10seed` CLI and schema rather than generalizing an already frozen RQ3 experiment in place.
  Rationale: the new experiment changes the task matrix, pair sampling, persisted training state, text representations, and analysis families. Prior formal roots and their code ledgers must remain reproducible.
  Date/Author: 2026-08-22 / Codex.
- Decision: preserve two RQ2 pair universes: pre-support feature coverage and post-support fitting/evaluation.
  Rationale: external feature lookup occurs before support filtering, but vocabulary and scaler fitting must not include pairs that never enter training.
  Date/Author: 2026-08-22 / Codex.
- Decision: compare all text branches from one immutable parent full state and replay the continuation-selected extra epoch count and LR traces.
  Rationale: this isolates text representation from random initialization, optimizer history, training duration, and learning-rate selection.
  Date/Author: 2026-08-22 / Codex.
- Decision: label every result `retrospective_rolling_development` and automatically downgrade under-covered RQ3 contrasts.
  Rationale: 2023Q3/Q4 and the selected architecture have historical exposure, and seed replication cannot repair low independent-event counts.
  Date/Author: 2026-08-22 / Codex.
- Decision: keep only selected dynamic checkpoints and final branch checkpoints in the formal root, while retaining initial/final checkpoints in the disposable benchmark.
  Rationale: this satisfies the 30 GiB residual-space gate without weakening parameter-update evidence or continuation lineage.
  Date/Author: 2026-08-22 / Codex.
- Decision: admit scheduled-release coverage using the deterministic release actually selected for each matched pair, while retaining every overlapping eligible release as a separate audit count.
  Rationale: overlapping eligible releases must not inflate a 19-release estimand into a nominal 20-release gate pass.
  Date/Author: 2026-08-22 / Codex.
- Decision: preserve the failed v1 config/root byte-for-byte and move the corrected interface directly to v2 without a compatibility shim.
  Rationale: source/config hashes are experimental evidence, and resuming v1 after changing restoration or checkpoint lineage would violate the source-drift fail-closed contract.
  Date/Author: 2026-08-23 / Codex.
- Decision: require one real-GPU `parent save_dynamic -> continuation resume_dynamic` canary plus production `load_vol_generator` GPU loads before matrix smoke and formal prepare.
  Rationale: the canary covers the two exact failure boundaries missed by the save-only 400-cell smoke. Its self-hashed evidence binds parent/continuation state, exact restored G/D tensors, learned updates, checkpoint SHA/size, architecture/model/grid lineage, physical GPU mapping, finite metrics, and finite inference.
  Date/Author: 2026-08-23 / Codex.

## Outcomes & Retrospective

The first v1 launch is a documented failed experiment, not a resumable run: 80 parents completed, zero continuations completed, and zero predictions were opened. The v1 benchmark and matrix smoke were useful evidence but did not cover restore or explicit-grid inference loading. The corrected v2 implementation now freezes a non-empty deterministic legacy architecture SHA and adds the missing GPU recovery/load gate. Focused orchestrator tests, an actual learned-checkpoint inference round trip, Ruff, and compilation pass; the fresh v2 benchmark/canary/smoke and formal background launch remain to be executed. This section will record completed v2 job/prediction counts, terminal SHA validation, scientific results, and limitations after the new supervisor finishes.

## Context and Orientation

The source surfaces live under `data/processed/rq3/news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1`. The 5-minute and 30-minute workbooks contain current and future 16×16 implied-volatility surfaces, LP embeddings, temporal identifiers, and support metadata. The maturity axis is exactly `[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]` business days. `raw_joint` means loss and evaluation use only cells supported on both the current and target raw surfaces.

The model is the existing `wgan_option` WGAN-GP with `film_conv_bottleneck_concat_v1` in the Generator and `lp_disabled_same_shape_v1` in the Critic. The latter retains its text branch parameters but replaces its encoded text feature with exact zeros. The legacy profile has 4,020,288 Generator parameters and 729,157 Critic parameters.

The four folds are expanding windows. Each fold trains on all earlier data, validates on the immediately preceding quarter, and tests once on 2023Q1, Q2, Q3, or Q4. “Tolerance 30” changes which news-to-market alignment is admitted but still predicts the next five-minute surface.

## Plan of Work

First, repair `scripts/rq2_pair/pair_features.py` and `scripts/rq2_pair/rq2_pair_experiment.py` so feature files cover all strict-text-valid pre-support pairs while vocabulary, PCA, and sentiment scaling use only final support-positive train pairs. The source split remains immutable. A fresh temporary four-fold prepare must reproduce the frozen fit counts.

Second, add pair-level text preparation and a versioned full training state beneath `src/wgan_option`. State is captured after the scheduler step and includes both networks, both Adam optimizers, both schedulers, Python/NumPy/Torch/CUDA RNG states, and the train-loader generator. Loading a weights-only checkpoint as a continuation parent fails closed. Old configs default to the old non-continuation behavior.

Third, add the config in `configs/rq123`, the module CLI in `scripts/rq123`, and focused tests. Prepare freezes source, code, dataset, grid, pair universe, text transform, event calendar, market detector, job, and model-contract hashes. Parents and continuations train with dynamic validation; branch recipes freeze the continuation’s extra epoch count and G/D LR traces; text branches start from the same parent and train to final-E without arm-specific selection.

Fourth, freeze all 400 checkpoints before opening test data. Generate one pair-level prediction bank with a common 64-draw noise contract. RQ1, RQ2, and RQ3 analyze only this bank. The finalizer creates resource summaries, Markdown and self-contained HTML reports, terminal QA, a frozen registry snapshot, and an all-file output hash manifest.

Finally, verify the implementation, run the all-legacy benchmark in a disposable root, and only then create the formal root. A detached supervisor executes the pipeline under an exclusive lock. Any source, config, parent, recipe, dataset, or artifact drift stops resume rather than silently mixing runs.

## Concrete Steps

Run all commands from `/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone` with `PYTHONPATH=src:.` and `/home/haobin_cui/.conda/envs/py312/bin/python`.

Run the all-legacy benchmark without creating the formal root:

    python -m scripts.rq123.news_first_vol_film_nolp_10seed benchmark --config configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml --output-dir outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2

Run the real-GPU restore/checkpoint-load canary, the literal 400-cell one-epoch smoke, then prepare the formal root:

    python -m scripts.rq123.news_first_vol_film_nolp_10seed recovery-canary --resume --config configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml --output-dir outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2
    python -m scripts.rq123.news_first_vol_film_nolp_10seed dry-run --resume --config configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml --output-dir outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2
    python -m scripts.rq123.news_first_vol_film_nolp_10seed prepare --resume --config configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml --output-dir outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2

The single supervisor performs those preflight steps in the same order. Create its log directory before shell redirection, then start it:

    mkdir -p outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2_control
    nohup setsid env PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m scripts.rq123.news_first_vol_film_nolp_10seed run-pipeline --resume --config configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml --output-dir outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2 > outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2_control/pipeline.log 2>&1 < /dev/null &

Use the `status` action for a read-only progress snapshot. Do not issue a second supervisor command while its PID or lock is live.

## Validation and Acceptance

Focused tests must demonstrate two RQ2 universes; pair collapse and train-only feature fitting; deterministic shuffle; exact model parameter counts; full-state next-step equivalence and scheduler continuity; 400 unique jobs and predictions; 40/40 GPU block balance; no test materialization before freeze; the three statistical families and their coverage downgrades; lock, retry, tamper, and completed-root read-only behavior.

The implementation passes Ruff format/check, py_compile, related regression tests, and `git diff --check`. The mandatory repository verification script is run even if the known missing `make format` target remains an infrastructure failure.

The benchmark passes only if all 36 jobs update Generator and Critic parameters, produce finite metrics, stay below 20 GiB peak memory on each A30, and keep host RAM below 85%. The recovery canary must then restore the chosen benchmark parent on a real GPU, prove the continuation initial G/D tensors exactly equal the parent best-learned tensors, update both models for one finite learned epoch, and load both parent and continuation Generator checkpoints through the production explicit-grid inference loader on the recorded physical GPU. Prepare freezes either 18 or 12 workers per GPU only after the benchmark, recovery canary, and 400-cell smoke all pass. The formal root is not created when any gate fails.

Terminal acceptance requires exactly 400 completed training jobs, 400 prediction cells, all expected RQ1/RQ2/RQ3 outputs, no under-covered contrast misreported as inferential, QA status `passed`, and a unique path/role/size/SHA row for every output. Re-running a completed pipeline must not change any file hash or modification time.

## Idempotence and Recovery

Prepare and freeze actions publish temp files by atomic rename. A completed task is skipped only after its job spec, registry row, status, and every artifact size and hash all match. A failed or interrupted attempt starts a new attempt directory from the same immutable parent; it never overwrites partial checkpoints. No mid-epoch recovery is claimed.

The supervisor owns a root lock and PID record. Resume validates the current code/config/source manifests, then stage journals, parent states, branch recipes, checkpoint allowlist, predictions, and terminal manifest in order. A completed root returns the existing report after read-only validation. Existing RQ2 and RQ3 formal roots are never mutated or deleted.

## Artifacts and Notes

The corrected formal root is `outputs/experiments/rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2`. The failed v1 root is read-only. The v2 companion control directory contains benchmark, recovery-canary and matrix-smoke evidence plus the supervisor PID, command, and log. Key scientific artifacts include the 400-row task registry, 80 parent-state manifests, 80 branch recipes, 400 prediction manifests, pair metrics for both tolerances, RQ1/RQ2 bootstrap and Holm tables, RQ3 scheduled/jump matching tables, resource summaries, reports, terminal QA, and output hashes.

## Interfaces and Dependencies

The public command is `python -m scripts.rq123.news_first_vol_film_nolp_10seed ACTION`. The corrected configuration is `configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml`; the v1 config remains immutable failure evidence. The full-state artifact kind is `news_first_wgan_full_training_state_v1`, captured at `end_of_epoch_after_scheduler_step`. The deterministic legacy architecture profile SHA is `5285d6c973d66b3b0fa78491bed99a7d707c68b02b8a4a6634ecbbea50c44124`. Pair text manifests and branch recipes are versioned, hash-addressed JSON/CSV artifacts and are required inputs to every downstream job spec.

Revision note (2026-08-22): created this plan after the user froze the 400-task protocol; implementation evidence will be appended as milestones complete.

Revision note (2026-08-23): recorded the v1 restore/checkpoint-lineage failure, froze v1 as read-only evidence, upgraded the branch-local experiment to v2, and added the mandatory real-GPU parent-continuation/checkpoint-load canary.
