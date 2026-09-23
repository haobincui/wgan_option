# Deliver the exact-TTM Generator-FiLM by Critic-LP two-stage experiment

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. Maintain this document in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The experiment asks two controlled architecture questions on the exact-TTM 16×16 news-first volatility data: whether Generator FiLM conditioning improves the existing bottleneck concatenation, and whether the Critic needs LP1024 text. It holds Small capacity, learning rate `5e-7`, `current_support_masked`, identity residual output, Gaussian32 noise, seed 42, and every existing loss setting fixed.

The user can first develop all 16 `Generator × Critic × text × tolerance` cells on data before 2023Q3 and select checkpoints on Q3. The system then freezes each cell's trained epoch and per-epoch Generator/Critic learning-rate trace, refits the same 16 cells on every row before Q4 without creating a validation or test loader, and permits Q4 access only through an explicit action. The observable safety property is that no Q4 workbook, sample object, loader, prediction, or evaluation exists during development or refit.

## Progress

- [x] (2026-08-21 09:05Z) Audited the existing news-first training, scheduling, checkpoint, and loader contracts.
- [x] (2026-08-21 10:05Z) Added the frozen exact-TTM factorial YAML configuration and mode-specific Small parameter contracts.
- [x] (2026-08-21 11:20Z) Added the Stage A/Stage B/Q4-gated orchestrator, benchmark policy, hash lineage, balanced GPU scheduling, resume checks, and CLI actions.
- [x] (2026-08-21 11:45Z) Added dedicated tests, including a production `Config` plus production loader test proving Stage B is training-only when `train_end == validation_end`.
- [ ] Integrate the analysis-owned Q3 selection, recipe freeze, Q4 analysis, and report modules under their approved module names.
- [ ] Run the complete targeted regression suite and mandatory code-change verification after all parallel files settle.
- [ ] Record final verification evidence and close Outcomes & Retrospective.

## Surprises & Discoveries

- Observation: the legacy training artifact collector always requires `generator_best.pt` and `discriminator_best.pt`, but a validation-free refit intentionally creates only final checkpoints.
  Evidence: `scripts/rq3/news_first_vol_training.py::_artifact_rows` assumes validation artifacts, while `WGAN_GP.train` disables best tracking when `val_loader is None`. The factorial orchestrator therefore has a stage-specific artifact validator: development requires initial/best/best-learned/final, whereas refit requires only final checkpoints and training lineage.

- Observation: a 16×16 Small WGAN has 149,333 parameters for bottleneck concatenation and 150,285 for FiLM; disabling Critic LP with `lp_disabled_same_shape_v1` preserves the Critic shape and parameter count.
  Evidence: production instantiation tests cover all four Generator/Critic combinations and compare exact counts and conditioning fingerprints.

- Observation: a refit with no validation needs `train_end_utc == validation_end_utc`, which older split validation rejected.
  Evidence: the production `Config` and `create_configured_vol_surface_dataloaders` test now loads a physical pre-Q4 workbook and returns `val_loader=None`, `test_loader=None`, zero validation/test samples, and `mode=news_first_training_only`.

- Observation: merely hashing `dataset_output_sha256.txt` is not enough to prove that required source workbooks still match the declaration.
  Evidence: factorial preparation now compares the 5m/30m workbook and support-audit SHA values with their declared relative-path hashes before writing its own source manifest.

## Decision Log

- Decision: use only the released branch-local field names `generator_conditioning_mode` and `critic_conditioning_mode`, with values `bottleneck_concat_v1`/`film_conv_bottleneck_concat_v1` and `lp_concat_v1`/`lp_disabled_same_shape_v1`.
  Rationale: these names are the frozen runtime and checkpoint compatibility boundary; aliases would weaken lineage.
  Date/Author: 2026-08-21 / Codex.

- Decision: Stage A contains exactly 16 jobs and Stage B contains exactly 16 jobs, all at seed 42. Stage B inverts the deterministic Stage A GPU assignment for each cell.
  Rationale: a single seed limits formal inference, while balanced and inverted GPU assignment avoids confounding a conditioning mode with one physical GPU.
  Date/Author: 2026-08-21 / Codex.

- Decision: Q3 analysis owns selection and recipe production. The orchestrator only validates the exact five-field recipe schema, registers file hashes, and creates refit jobs.
  Rationale: statistical/checkpoint selection belongs in the analysis layer and must not be duplicated by orchestration code.
  Date/Author: 2026-08-21 / Codex.

- Decision: the formal root cannot be created until an isolated 16-worker, one-epoch benchmark finishes. The benchmark selects 8 workers/GPU only below both 20 GiB/GPU and 85% host RAM; otherwise the new formal root freezes 4 workers/GPU.
  Rationale: concurrency must be decided before immutable job/config hashes are created.
  Date/Author: 2026-08-21 / Codex.

- Decision: preparation may scan the source workbook while materializing pre-Q4 files, but it records that fact honestly. It does not materialize a Q4-specific workbook until `evaluate-q4`.
  Rationale: source XLSX is not physically partitioned, so claiming that its Q4 bytes were never scanned would be false; the relevant hard boundary is that trainer and loader receive only pre-Q4 workbooks and no Q4 sample objects exist.
  Date/Author: 2026-08-21 / Codex.

## Outcomes & Retrospective

The configuration, orchestration, CLI, model/grid contracts, exact split counts, and dedicated tests are implemented. The formal experiment has not been prepared or launched. Final completion still depends on the approved analysis and report modules and the final repository-wide verification pass. The most important implementation lesson is that validation-free refit is not simply “training with a later cutoff”: it requires different loader, scheduler, early-stop, checkpoint, and artifact contracts, all of which must be hash-bound.

## Context and Orientation

The entrypoint is `scripts/rq3/main.py`. The new orchestration module is `scripts/rq3/news_first_vol_generator_film_critic_factorial.py`, and the frozen configuration is `configs/rq3/news_first_vol_generator_film_critic_factorial.yaml`. Statistical work belongs in `scripts/rq3/news_first_vol_generator_film_critic_factorial_analysis.py`; rendering belongs in `scripts/rq3/news_first_vol_generator_film_critic_factorial_report.py`.

The source dataset root is `data/processed/rq3/news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1`. Its surface grid is 16 strikes from 0.970 to 1.030 in steps of 0.004 and exact maturity nodes `[1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38]`. “Stage A” means development training before 2023-07-01 with Q3 validation from 2023-07-01 through 2023-09-30. “Stage B” means training-only refit on all observations before 2023-10-01. “Stage C” means the one explicitly gated Q4 evaluation from 2023-10-01 through 2023-12-31.

A “recipe” is a JSON object with exactly five fields: `schema_version`, `refit_mode`, `num_epochs`, `generator_lr_trace`, and `discriminator_lr_trace`. Both traces contain contiguous `{epoch, lr}` rows for epochs 1 through `num_epochs`. The runtime replays those values and rejects any mismatch.

## Plan of Work

The orchestrator resolves and validates the frozen config, verifies the source dataset's declared hashes and exact grid, materializes physical pre-Q4 workbooks, and writes a split manifest with fail-closed row/pair/session counts. It builds 16 development jobs and distributes every Generator, Critic, text, and tolerance level equally across both GPUs.

After all development artifacts validate, Q3 analysis freezes the selection and writes 16 recipe files under `analysis/refit_recipes` plus `analysis/refit_recipe_manifest.json`. The orchestrator validates and hashes those files, creates the 16 refit configs, and checks that they disable validation/test materialization, scheduling, early stopping, and initial evaluation. Refit completion freezes an allowlist containing only final Generator/Critic checkpoints.

The `evaluate-q4` action validates every allowlisted checkpoint, opens the irreversible Q4 gate, materializes the common 5m Q4 workbook, and delegates predictions/statistics to analysis. `postprocess` is allowed only after Q4 analysis succeeds.

## Concrete Steps

Run all commands from the repository root with the project Python environment and `PYTHONPATH=src:.`.

First run the isolated resource benchmark; this creates only the sibling `_benchmark` root:

    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial benchmark

Then prepare and run development:

    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial prepare
    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial launch-development

Freeze Q3 selection and recipes, then refit all 16 cells:

    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial freeze-selection
    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial launch-refit

Only after Stage B and its checkpoint allowlist are frozen, run Q4 once and render the report:

    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial evaluate-q4
    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial postprocess
    python scripts/rq3/main.py train-news-first-vol-generator-film-critic-factorial qa

No command above has been run against the formal output root during implementation.

## Validation and Acceptance

The dedicated suite must report all tests passing:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m unittest \
      tests.test_scripts.test_rq3_news_first_vol_generator_film_critic_factorial -v

It proves exact grid and parameter contracts, 16+16 unique jobs, mixed GPU assignment, exact recipe schema, Stage B training-only loader behavior, Q4 pre-gate rejection, and every CLI action.

Static and repository checks are:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile \
      scripts/rq3/news_first_vol_generator_film_critic_factorial.py \
      scripts/rq3/main.py
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

Acceptance also requires observing `development_job_count=16`, `refit_job_count=16`, no Q4 directory before the gate, a 32-row final G/D checkpoint allowlist after refit, and `q4_evaluated=true` only after the explicit action.

## Idempotence and Recovery

All formal actions are fail-closed. A pre-existing root requires `--reuse` only for an identical prepared root, while interrupted or failed work requires `--resume`. A completed artifact is skipped only when its config and every recorded artifact SHA still match. Source, code, config, recipe, selection, and checkpoint drift stop the run. Q4 evaluation is never silently repeated; after success, `--resume` only validates and returns the existing result.

If the benchmark chooses the fallback concurrency, create the formal root only after that decision. Never change slots inside an existing formal root. To retry before the formal root exists, use a distinct temporary benchmark root or deliberately remove only the isolated failed benchmark directory after reviewing its logs; do not modify existing historical experiments.

## Artifacts and Notes

Frozen expected supported counts are:

    Stage A train 5m:  968 rows / 748 pairs / 203 sessions
    Stage A Q3 5m:    148 rows / 135 pairs /  33 sessions
    Stage B refit 5m: 1116 rows / 883 pairs / 236 sessions
    Stage A train 30m: 1659 rows / 1083 pairs / 237 sessions
    Stage A Q3 30m:    238 rows /  186 pairs /  41 sessions
    Stage B refit 30m: 1897 rows / 1269 pairs / 278 sessions
    Q4 common 5m:      167 rows /  143 pairs /  45 sessions

The initial dedicated test transcript was:

    Ran 8 tests in 2.892s
    OK

## Interfaces and Dependencies

The public CLI command is `train-news-first-vol-generator-film-critic-factorial` with actions `prepare`, `benchmark`, `dry-run`, `launch-development`, `freeze-selection`, `launch-refit`, `evaluate-q4`, `postprocess`, `worker`, and `qa`.

The orchestrator exports `run_news_first_vol_generator_film_critic_factorial(config_path, output_dir, ...)`. It calls these analysis/report interfaces lazily:

    run_film_critic_q3_analysis(root)
    freeze_refit_recipes(root, selection)
    run_film_critic_q4_analysis(root)
    render_film_critic_report(root)

The core runtime fields are `generator_conditioning_mode`, `critic_conditioning_mode`, `news_first_refit_mode`, `news_first_refit_recipe_path`, `news_first_refit_recipe_sha256`, and `news_first_materialize_validation_loader`. No alternative text-prefixed conditioning field is accepted by this experiment.

Revision note (2026-08-21): added exact Stage A/Q3 split count gates, declared dataset hash verification, analysis-owned recipe interfaces, and mode-specific model contracts after integration review.
