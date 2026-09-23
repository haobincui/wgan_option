# Run the 60-epoch Mask+Coords FiLM U-Net capacity sweep

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. It is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The completed three-seed architecture ablation found that all four mask-and-coordinate-aware FiLM U-Net variants selected the configured epoch-30 ceiling. This experiment reruns those four variants from fresh initialization for up to 60 epochs across six spatial/adversarial capacity levels. A user can inspect an isolated 72-job registry, checkpointed training runs, per-arm capacity rankings, and one combined Q3 development summary. The completed 21-job experiment and every historical checkpoint remain read-only, and no Q4 loader or prediction is permitted.

## Progress

- [x] (2026-08-27 12:55Z) Audited the completed four-arm result, existing capacity profiles, exact model graphs, release boundary, GPUs, RAM, disk, and prior benchmark evidence.
- [x] (2026-08-27 12:55Z) Froze four architectures, six capacity levels, three seeds, and the 60-epoch Q3-only protocol.
- [x] (2026-08-27 14:56Z) Implemented the isolated configuration, orchestrator, 72-job registry, resource gate, 24-cell postprocessing, and seven focused tests.
- [x] (2026-08-27 14:57Z) Passed seven focused tests, 43 related regressions, Ruff, py_compile, and diff checks. The mandatory wrapper was invoked and stopped only because this repository has no `make format` target; no project-native check failed.
- [x] (2026-08-27 14:56Z) Completed the 24-graph real-data dry run; every arm/capacity graph built the frozen Train/Q3 panels with no Q4 loader.
- [x] (2026-08-27 14:59Z) Completed the 72-cell one-epoch benchmark: 72/72 finite, every G/D updated, both GPUs peaked at 6.066 GiB, and host RAM peaked at 71.32%; the strict gate passed.
- [x] (2026-08-27 15:00Z) Launched the formal 72-job pipeline under detached supervisor PID 1244790; the first wave has 36 running, 36 pending, and zero failures.
- [x] (2026-08-27 16:35Z) Postprocessed all 72 jobs into 24 arm/capacity cells, verified 720 frozen artifact hashes, and updated Outcomes & Retrospective.

## Surprises & Discoveries

- Observation: every seed of every mask-and-coordinate U-Net arm selected epoch 30, so the previous comparison was right-censored by its training horizon.
  Evidence: `outputs/experiments/rq3_news_first_vol_film_pure_cnn_ablation_3seed_exact_ttm_v1/analysis/job_summary.csv`.
- Observation: reusing the historical micro/tiny profiles would change the Critic LP width and make the fixed Text128 adapter dominate the smallest Generator, so those labels would not preserve the four selected architectures.
  Evidence: executable parameter instantiation showed a fixed Text128 encoder alone has 295,808 parameters, while historical micro/tiny spatial widths are only one and two channels.
- Observation: no local or remote `v*` release tag is available. The new CLI and schema are therefore branch-local, while old modes, roots, and checkpoints remain untouched.
  Evidence: the remote-aware release discovery command returned an empty value.
- Observation: both A30s are idle, approximately 325 GiB host RAM and 372 GiB disk are available, and the prior 21-job root occupies about 757 MiB.
  Evidence: `nvidia-smi`, `free -h`, `df -h .`, and `du -sh` on 2026-08-27.
- Observation: all 24 representative graph/config combinations passed the actual loader dry run after formatting and focused regression checks.
  Evidence: `/tmp/film_unet_epoch60_dryrun.PeRjPZ/control/dry_run_manifest.json` records 24 immutable log artifacts and contract SHA `05a45ed2342c284027d31ea640e5c23f4ac0d7c294469a86fb200d1804d85ef0`.
- Observation: the full 72-job benchmark fit comfortably at the frozen concurrency of 18 workers per A30.
  Evidence: `outputs/benchmarks/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_epoch1_v1/analysis/benchmark_summary.json` records 72/72 completion, 6.066 GiB peak on each GPU, 71.32% host RAM peak, and `gate_passed=true`.

## Decision Log

- Decision: include exactly the four previously ranked Mask+Coords variants: Text128 with LP-concat Critic; Text64 with LP-concat Critic; Text64 with same-shape NoLP Critic; and Text64 with projection Critic.
  Rationale: these are the cells selected by the user, and retaining all four separates Generator adapter width from Critic conditioning.
  Date/Author: 2026-08-27 / Codex.
- Decision: use capacities `c04`, `c08`, `c12`, `c16`, `c24`, and `c32`. Generator and Critic base channels are 4, 8, 12, 16, 24, and 32; Critic score hidden widths are 96, 192, 288, 384, 576, and 786. Generator text adapters remain fixed by arm and Critic text width remains 128.
  Rationale: this is a controlled joint spatial/adversarial-capacity sweep. It preserves the four text-conditioning definitions and avoids misleading reuse of historical profile names whose text widths also changed.
  Date/Author: 2026-08-27 / Codex.
- Decision: train from fresh initialization for at most 60 epochs with early-stop minimum 30 and patience 20, LR `5e-7`, no warm-up, and the existing ReduceLROnPlateau settings.
  Rationale: weights-only continuation would not preserve optimizer, scheduler, or RNG state and would be incomparable across new capacities. Extending the ceiling while retaining all other optimization settings isolates the requested change.
  Date/Author: 2026-08-27 / Codex.
- Decision: use a new module, configuration, benchmark root, formal root, and schema rather than modifying the completed 21-job orchestrator.
  Rationale: the completed root records source/config hashes and must remain an auditable historical artifact.
  Date/Author: 2026-08-27 / Codex.

## Outcomes & Retrospective

The experiment completed 72/72 jobs with zero failures. The global descriptive point leader is Text128 + LP-concat Critic at `c32`: mean Q3 MAE `0.001799024249`, a geometric-mean improvement of `0.1167%` over persistence across three seeds. Its three selected epochs are 60, 45, and 46. Text64 + LP Critic and Text64 + NoLP Critic also select `c32`, with mean Q3 MAE `0.001799512731` and `0.001799569231`. Projection Critic's point leader is `c24` at `0.001799962884`; its one-SE candidate is the smaller `c16`.

Extending the horizon modestly improved the prior same-graph `c32` results: the improvement over persistence rose by 0.00696 percentage points for Text128, 0.00570 for Text64 LP, and 0.00552 for Text64 NoLP. Projection's new `c24` leader improved its prior epoch-30 arm result by 0.00867 percentage points. These gains are real in the frozen Q3 summaries but small in absolute MAE.

Longer training did not establish convergence for most cells. Sixty-five of 72 jobs selected epoch 60; 68 trained through epoch 60, while four `c12` Text64 LP/NoLP runs early-stopped at epoch 30. Thus the prior 30-epoch experiment was right-censored, but the new 60-epoch sweep is still right-censored for most configurations. Capacity effects are not monotonic, and this postprocessing is descriptive: Q3 is both the early-stopping and selection panel, no paired bootstrap/Holm inference was run here, and Q4 was never materialized. The defensible conclusion is that `Text128 + LP Critic, c32` is the current Q3 point leader, not that it is a statistically confirmed winner.

## Context and Orientation

The Generator implementation is `src/wgan_option/models/generator.py`, where `film_unet_mask_coords_v1` consumes masked current IV, a current-only support mask, normalized moneyness, and normalized log-TTM as a four-channel 16 by 16 tensor. Encoded LP text enters six FiLM layers, Gaussian32 noise is broadcast at the 4 by 4 bottleneck, bilinear decoding uses two encoder skips, and a zero-initialized one-by-one convolution predicts an identity-anchored residual. The Critic implementation is `src/wgan_option/models/discriminator.py` and supports LP concatenation, parameter-matched NoLP, and parameter-matched projection conditioning.

The exact-TTM grid has moneyness values from 0.970 through 1.030 and maturity days `[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]`, with SHA-256 `7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8`. Training uses timestamps before 2023-07-01 and Q3 validation uses timestamps from 2023-07-01 through 2023-09-30. The frozen panels contain 968 training rows / 748 pairs / 203 sessions and 148 Q3 rows / 135 pairs / 33 sessions. No timestamp on or after 2023-10-01 may be materialized.

The formal root is `outputs/experiments/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_exact_ttm_v1`. The independent benchmark root is `outputs/benchmarks/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_epoch1_v1`.

## Plan of Work

Add `configs/rq3/news_first_vol_film_unet_capacity_epoch60_3seed.yaml` with the frozen four-arm, six-capacity, three-seed matrix. Add `scripts/rq3/news_first_vol_film_unet_capacity_epoch_3seed.py` as an independent lifecycle CLI with `benchmark`, `prepare`, `dry-run`, `worker`, `launch`, `status`, `postprocess`, and `run-pipeline` actions. It will instantiate every graph to freeze actual parameter counts, generate 72 unique job specs, balance them 36/36 across GPU 0 and GPU 1, validate same-initial-state comparisons, persist source/config/data/model hashes, and fail closed on any resume drift.

The six Text128 total parameter counts are expected to be 472,386; 544,738; 644,066; 770,370; 1,103,906; and 1,556,902. The corresponding Text64 totals, identical across its three parameter-matched Critic modes, are 239,298; 304,482; 396,642; 515,778; 834,978; and 1,273,638.

Run a 24-graph dry run, then a complete 72-job one-epoch benchmark with 18 workers per GPU. Formal output must not be created until the benchmark confirms 72/72 completion, finite metrics, both Generator and Critic updates in every cell, peak memory below 20 GiB per GPU, and RAM below 85%. If the concurrency gate fails, prepare a fresh 12-worker benchmark/config/root rather than mutating the frozen root.

After a successful benchmark, launch one detached formal supervisor. Postprocessing will write a 72-row job summary, a 24-row arm-by-capacity summary, within-arm capacity rankings, point leaders, seed consistency, parameter counts, selected epochs, and ratios to persistence. Results are Q3 development evidence only.

## Concrete Steps

From the repository root, run:

    /home/haobin_cui/.conda/envs/py312/bin/python -m unittest tests.test_scripts.test_rq3_news_first_vol_film_unet_capacity_epoch_3seed
    ruff format --check <changed Python files>
    ruff check <changed Python files>
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile <changed Python files>
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

Then run the real benchmark:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m scripts.rq3.news_first_vol_film_unet_capacity_epoch_3seed benchmark --config configs/rq3/news_first_vol_film_unet_capacity_epoch60_3seed.yaml --output-dir outputs/benchmarks/rq3_news_first_vol_film_unet_capacity_epoch60_3seed_epoch1_v1

After it passes, launch the formal pipeline with `nohup setsid`, redirect output into a sibling `_control/pipeline.log`, and save the launcher PID.

## Validation and Acceptance

The configuration must resolve to exactly 72 unique jobs and 24 unique executable graphs. GPU assignment must be exactly 36/36 and balanced within every capacity. Every Text128 arm must retain `LP1024-256-128`; every Text64 arm must retain `LP1024-64-64`; every Critic must retain a 128-dimensional LP encoder. All three Critic modes must have equal parameter counts within capacity. Text64 LP, NoLP, and projection Generators must have identical initial state hashes for a fixed capacity and seed; LP and NoLP Critics must also have identical initial state hashes, while projection is permitted a different state layout.

Generated training configs must specify 60 epochs, early-stop minimum 30, patience 20, LR `5e-7`, no warm-up, batch 16, five Critic steps, Gaussian32, identity residual, `raw_joint`, `current_support_masked`, validation MC16, and zero Q4/test loaders. Resume must reject changed source, config, job, dataset, grid, model-contract, or artifact hashes. A completed root must be read-only under repeat status or run-pipeline calls.

The benchmark must complete all 72 jobs without OOM or NaN and prove that both networks updated. Formal launch is accepted only when the detached PID remains alive, all intended jobs enter running/pending states with zero failures, and resource snapshots remain within the benchmark-frozen gate.

## Idempotence and Recovery

Benchmark and formal roots are immutable after their registries are prepared. A rerun without `--resume` must refuse an existing registry. A resumed job may be skipped only when its spec and every registered artifact size and SHA match. One failed worker stops the supervisor. Recovery requires diagnosing the log, preserving the failed attempt, and explicitly running `--resume`; source or configuration changes require a new suffixed root. No command deletes or modifies historical experiment roots.

## Artifacts and Notes

Expected artifacts include resolved configuration and model-contract manifests; a 72-job registry; capacity/parameter manifests; initial-state fairness hashes; per-job status, logs, checkpoints, and epoch metrics; benchmark resource snapshots and gate summary; Q3 job and arm-capacity summaries; supervisor PID/log/status; and a terminal hash manifest.

## Interfaces and Dependencies

The new CLI is branch-local because the repository has no release tag. It reuses the existing `Generator`, `Discriminator`, `scripts/train/train_vol.py`, and news-first Q3 data loader without changing model or checkpoint interfaces. No new third-party dependency is required.
