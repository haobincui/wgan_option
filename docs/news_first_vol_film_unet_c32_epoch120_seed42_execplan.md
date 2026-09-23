# Run four c32 Mask+Coords FiLM U-Nets for up to 120 epochs

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. It is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The completed three-seed capacity experiment showed that `c32` was the Q3 point leader for the Text128 LP, Text64 LP, and Text64 NoLP architectures, while the seed-42 validation MAE of all four c32 architectures was still improving at epoch 60. This experiment starts four fresh seed-42 runs, fixes capacity at `c32`, and raises the maximum horizon to 120 epochs. A user can inspect an isolated four-job registry, compare each new best Q3 MAE with its exact epoch-60 seed-42 predecessor, and determine whether the longer horizon produces material gains or merely extends validation-panel optimization. The completed 60-epoch root remains read-only and Q4 is never materialized.

## Progress

- [x] (2026-08-27 17:00Z) Confirmed the four c32 seed-42 epoch-60 traces are still decreasing at epoch 60 and already use the LR floor `5e-8`.
- [x] (2026-08-27 17:00Z) Froze the new four-job, seed-42, max-120, Q3-only protocol and compatibility boundary.
- [x] (2026-08-27 21:06Z) Implemented the independent configuration, four-job lifecycle CLI, predecessor comparison postprocessing, registry/status hardening, and twelve focused tests.
- [x] (2026-08-27 21:07Z) Passed 19 new/previous-orchestrator tests, 33 model/input regressions, Ruff, py_compile, and diff checks. The mandatory wrapper was invoked and stopped only because this repository has no `make format` target.
- [x] (2026-08-27 21:07Z) Completed the four-graph real-data dry run with the frozen Train/Q3 panels and no Q4 loader.
- [x] (2026-08-27 21:09Z) Completed the four-job one-epoch benchmark: 4/4 finite, every G/D updated, GPU peaks 0.769/0.771 GiB, host RAM peak 7.52%, and the strict gate passed.
- [x] (2026-08-27 21:10Z) Launched detached `run-pipeline` supervisor PID 1411630; all four formal jobs are running, each GPU owns two, and zero failures are recorded.
- [x] (2026-08-27 21:27Z) Completed and postprocessed all four jobs with zero failures; verified all 40 registered artifact hashes and updated Outcomes & Retrospective.

## Surprises & Discoveries

- Observation: all four c32 seed-42 validation MAE traces improve monotonically through epoch 60 even though both learning rates are already at their `5e-8` floor.
  Evidence: the final rows of each `training_metrics.csv` under `outputs/experiments/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_exact_ttm_v1/runs/*/c32/seed_042`.
- Observation: no local or remote `v*` release tag exists.
  Evidence: the repository release-tag discovery command returned an empty value on 2026-08-27.

## Decision Log

- Decision: use seed `42` as the single seed.
  Rationale: it is the conventional first seed in the completed matrix, and all four exact predecessor runs exist for paired trajectory comparison.
  Date/Author: 2026-08-27 / Codex.
- Decision: train from fresh initialization rather than continue epoch-60 weights.
  Rationale: the retained checkpoints are weights-oriented and do not establish a complete optimizer, scheduler, data-loader, and RNG continuation boundary. Fresh initialization preserves the existing reproducible seed contract and makes all four arms comparable.
  Date/Author: 2026-08-27 / Codex.
- Decision: use max epoch 120, early-stop minimum 30, patience 20, LR `5e-7`, no warm-up, and the existing ReduceLROnPlateau factor, patience, and floor.
  Rationale: this is a strict single-factor epoch-budget extension of the completed c32 experiment. Every seed-42 arm improved strictly through epoch 60, so the unchanged early-stop rule cannot truncate the new run before reproducing the old horizon; changing the early-stop rule would confound the requested 60-to-120 comparison.
  Date/Author: 2026-08-27 / Codex.
- Decision: add a branch-local CLI/config/root and no migration shim.
  Rationale: no release tag exists and the completed 60-epoch registry records immutable source/config hashes; modifying its orchestrator would weaken replay auditing.
  Date/Author: 2026-08-27 / Codex.
- Decision: preserve each arm's physical GPU assignment from its exact c32/seed42 epoch-60 predecessor while retaining a two/two split.
  Rationale: the main comparison is within-arm epoch 60 versus epoch 120, so same-arm hardware continuity is more important than colocating a different cross-arm contrast.
  Date/Author: 2026-08-27 / Codex.

## Outcomes & Retrospective

The experiment completed 4/4 jobs with zero failures. Every job trained through epoch 120 and selected epoch 120. Text128 + LP-concat Critic is the single-seed Q3 point leader at MAE `0.001799373712`, followed by Text64 + LP-concat at `0.001800076098`, Text64 + projection at `0.001800203527`, and Text64 + same-shape NoLP at `0.001800284154`.

Relative to each exact c32/seed42 epoch-60 predecessor, Q3 MAE improved by `0.009827%`, `0.011526%`, `0.008182%`, and `0.010437%`, respectively. The fresh runs reproduce their predecessor epoch-60 values within roughly `2e-9` to `6e-9` absolute MAE, supporting trajectory comparability without claiming bitwise equality. From epoch 61 through 120, all four validation MAE traces decrease at every epoch; the final ten epochs also improve while both learning rates remain at the `5e-8` floor. Therefore the 60-epoch result was right-censored and the 120-epoch result remains right-censored, although the absolute incremental gains are very small.

All 40 registered artifacts passed size/SHA verification, no Q4/test prediction artifact exists, and the final ranking is single-seed descriptive evidence only. It does not supersede the three-seed architecture comparison or establish statistical support.

## Context and Orientation

The Generator is `src/wgan_option/models/generator.py`. All four arms use `film_unet_mask_coords_v1`, which feeds masked current IV, support mask, moneyness coordinates, and log-TTM coordinates through a U-Net and applies LP text through FiLM. Text128 uses `LP1024 -> 256 -> 128`; the other arms use `LP1024 -> 64 -> 64`. The Critic in `src/wgan_option/models/discriminator.py` is respectively LP concatenation, LP concatenation, same-shape NoLP, or LP projection. At c32, the Text128 WGAN has 1,556,902 parameters and each Text64 WGAN has 1,273,638 parameters.

Training uses the exact-TTM 16 by 16 grid with SHA-256 `7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8`. Train timestamps are before 2023-07-01; Q3 validation is 2023-07-01 through 2023-09-30. The frozen panels contain 968 training rows / 748 pairs / 203 sessions and 148 Q3 rows / 135 pairs / 33 sessions. Q4 and test loaders remain disabled.

The formal root is `outputs/experiments/rq3_news_first_vol_film_unet_c32_epoch120_seed42_exact_ttm_v1`. The benchmark root is `outputs/benchmarks/rq3_news_first_vol_film_unet_c32_epoch120_seed42_epoch1_v1`. The exact predecessor root is `outputs/experiments/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_exact_ttm_v1` and is read-only.

## Plan of Work

Add `configs/rq3/news_first_vol_film_unet_c32_epoch120_seed42.yaml` and `scripts/rq3/news_first_vol_film_unet_c32_epoch120_seed42.py`. The lifecycle CLI will implement `benchmark`, `prepare`, `dry-run`, `worker`, `launch`, `status`, `postprocess`, and `run-pipeline`. It will instantiate all four executable graphs, freeze parameter and initial-state hashes, assign two jobs to each A30, persist source/config/model/artifact hashes, and fail closed on drift. Postprocessing will create four rows containing best epoch, Q3 MAE, persistence improvement, exact predecessor epoch-60 MAE, and the paired delta.

Add `tests/test_scripts/test_rq3_news_first_vol_film_unet_c32_epoch120_seed42.py` to protect the four-job matrix, parameter counts, two-per-GPU mapping, fresh seed-42 state, epoch/early-stop/LR contract, Q3-only boundary, benchmark one-epoch override, predecessor linkage, and resource gate.

Before the formal root exists, run all four one-epoch jobs with two workers per GPU. The gate requires finite metrics, both models updated in every job, less than 20 GiB peak memory on each GPU, and less than 85% host RAM. Only after it passes may the formal supervisor create the formal registry and launch the four 120-epoch jobs.

## Concrete Steps

From the repository root, run:

    /home/haobin_cui/.conda/envs/py312/bin/python -m unittest tests.test_scripts.test_rq3_news_first_vol_film_unet_c32_epoch120_seed42
    ruff format --check scripts/rq3/news_first_vol_film_unet_c32_epoch120_seed42.py tests/test_scripts/test_rq3_news_first_vol_film_unet_c32_epoch120_seed42.py
    ruff check scripts/rq3/news_first_vol_film_unet_c32_epoch120_seed42.py tests/test_scripts/test_rq3_news_first_vol_film_unet_c32_epoch120_seed42.py
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile scripts/rq3/news_first_vol_film_unet_c32_epoch120_seed42.py tests/test_scripts/test_rq3_news_first_vol_film_unet_c32_epoch120_seed42.py
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

Then run the benchmark and, only after its gate passes, launch the formal supervisor with `nohup setsid`, redirecting its output to the sibling `_control/pipeline.log`.

## Validation and Acceptance

The resolved matrix must contain exactly four unique jobs, all at c32 and seed 42, balanced two per physical GPU. Text64 Generators must share an initial-state hash; LP and same-shape NoLP Critics must share an initial-state hash; projection is parameter matched but excluded from state equality because its executable graph differs. Generated formal configs must specify 120 epochs, early-stop minimum 30, patience 20, LR `5e-7`, no warm-up, validation MC16, and no Q4/test loader. Benchmark configs must specify one epoch and disable early stopping without changing the formal registry.

Every completed status must register frozen artifact size and SHA. Postprocessing must reject incomplete or drifted artifacts, produce four rows, and link each row to exactly one c32 seed-42 predecessor from the completed 60-epoch summary. Formal launch is accepted when the detached PID is alive, all four jobs are running or completed, neither GPU has more than two jobs, and no task has failed.

## Idempotence and Recovery

The benchmark and formal roots are immutable after registry creation. Running without `--resume` must reject an existing registry. Resume may skip a job only when source, config, job spec, model contract, predecessor manifest, and every artifact size/SHA agree. A failed worker stops the supervisor; recovery preserves the attempt log and requires explicit `--resume`. Any source or protocol change requires a new suffixed root. Historical experiment roots are never modified.

## Artifacts and Notes

Expected artifacts include four generated configs, a four-job registry, initialization-fairness and predecessor manifests, per-job logs/checkpoints/metrics, benchmark resource snapshots and summary, a four-row Q3 comparison CSV/JSON, and supervisor status/PID/log files.

## Interfaces and Dependencies

The CLI and config are branch-local. They reuse the existing `Generator`, `Discriminator`, `scripts/train/train_vol.py`, and exact-TTM news-first data loader. No new dependency or checkpoint migration is introduced.
