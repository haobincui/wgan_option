# Run the FiLM-only and fully convolutional ablation

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. This document is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The experiment will determine, one controlled change at a time, whether the current Generator benefits from removing the duplicate bottleneck text concatenation, replacing the spatial flatten/MLP head with a fully convolutional FiLM encoder-decoder, exposing mask and real grid coordinates, shrinking the LP text adapter, and changing the Critic from late LP concatenation to either no text or projection conditioning. A user can observe the result through an isolated 21-job registry, per-job training artifacts, a Q3 MAE summary, and an arm ranking. Existing experiments and checkpoints remain read-only.

## Progress

- [x] (2026-08-27 00:00Z) Audited the active FiLM Generator, LP Critic, prior three-seed results, GPU/RAM/disk capacity, and release boundary.
- [x] (2026-08-27 00:00Z) Froze the seven-arm, three-seed development matrix and isolated output root.
- [x] (2026-08-27 12:24Z) Implemented additive model modes while preserving all old mode and checkpoint behavior.
- [x] (2026-08-27 12:24Z) Implemented the isolated orchestrator, configuration, registry, status, postprocessing, and benchmark interfaces.
- [x] (2026-08-27 12:24Z) Added and passed model-contract, identity, conditioning, task-grid, initial-state fairness, and resume tests.
- [x] (2026-08-27 12:26Z) Passed 47 targeted tests, Ruff check/format, py_compile, git diff check, and the real 21-cell dual-A30 one-epoch benchmark. The mandatory wrapper remains an infrastructure exception because this repository has no `make format` target.
- [x] (2026-08-27 12:29Z) Launched the formal 21-job experiment under one detached supervisor (PID 1148989); all 21 jobs entered `running` with a frozen 11/10 GPU split and zero failures.
- [x] (2026-08-27 12:44Z) Completed all 21 formal jobs with zero failures and postprocessed the frozen Q3 ranking; Q4 was never materialized.

## Surprises & Discoveries

- Observation: no remote or local `v*` release tag was found, so the new modes and CLI are branch-local compatibility surfaces.
  Evidence: the release-tag discovery command returned an empty value on 2026-08-27.
- Observation: both A30 GPUs are idle and the filesystem has roughly 374 GiB free, so the experiment does not require deleting prior artifacts.
  Evidence: `nvidia-smi`, `df -h .`, and `free -h` recorded before implementation.
- Observation: the working tree already contains many research changes from earlier experiments.
  Evidence: `git status --short`; this work must preserve and avoid reverting all unrelated changes.
- Observation: the mandatory verification wrapper stops before lint/typecheck/tests because this repository has no `make format` target; the py312 environment also does not install pytest.
  Evidence: `bash .agents/skills/code-change-verification/scripts/run.sh` exited 2 with `No rule to make target 'format'`; project-native unittest, Ruff, py_compile, and diff checks pass independently.
- Observation: all 21 one-epoch benchmark cells completed with finite metrics and verified updates to both Generator and Critic parameters while using far less than the resource limits.
  Evidence: `analysis/benchmark_summary.json` reports 21/21 complete and all four safety booleans true; sampled peaks were 4,405 MiB on GPU 0, 4,019 MiB on GPU 1, and approximately 14% host RAM usage, versus gates of 20 GiB per GPU and 85% RAM.

## Decision Log

- Decision: add new conditioning modes instead of altering `film_conv_bottleneck_concat_v1`, `lp_concat_v1`, or `lp_disabled_same_shape_v1`.
  Rationale: old checkpoint state dictionaries and historical results must retain their exact executable meaning.
  Date/Author: 2026-08-27 / Codex.
- Decision: use LR `5e-7` without warm-up, seeds `42, 202, 404`, the exact-TTM 16x16 grid, the existing pre-Q4 train/Q3 split, and no Q4 loader.
  Rationale: the prior ten-epoch warm-up made all three legacy seeds slightly worse and selected epochs before warm-up completed; Q3-only reuse prevents holdout-driven architecture selection.
  Date/Author: 2026-08-27 / Codex.
- Decision: run seven arms: current baseline; FiLM-only dense; one-channel FiLM U-Net with the original 128-dimensional text adapter; mask/coordinate-aware FiLM U-Net with that adapter; mask/coordinate-aware FiLM U-Net with a compact 64-dimensional adapter; compact coordinate-aware FiLM U-Net with same-shape NoLP Critic; and the same Generator with projection Critic.
  Rationale: this ordering changes one interpretable component at a time. In particular, a one-channel U-Net control prevents a bundled decoder-plus-coordinate change from being misattributed to the CNN decoder alone.
  Date/Author: 2026-08-27 / Codex.

## Outcomes & Retrospective

Implementation and the formal Q3 development experiment are complete. All 21/21 jobs finished with zero failures. The point leader was the mask-and-coordinate-aware FiLM U-Net with the original `LP1024-256-128` Generator adapter and LP-concatenating Critic: mean Q3 MAE `0.0017991496`, versus persistence `0.0018011262`, or a descriptive `0.1097%` improvement across three seeds. The compact 64-dimensional adapter ranked second (`0.0017996155`), its NoLP-Critic counterpart third (`0.0017996687`), and the projection-Critic counterpart fourth (`0.0018001190`). Removing the dense Generator's duplicate bottleneck text concatenation ranked fifth and improved the seed mean only because seed 404 was strong; the one-channel U-Net without mask/coordinates ranked last.

The controlled sequence indicates that the main positive architectural contribution is exposing support and true grid coordinates, not merely replacing the dense head with a convolutional decoder. Keeping the 128-dimensional Generator text adapter was consistently better than shrinking it to 64 dimensions. Removing LP from the Critic changed mean MAE negligibly, while projection conditioning was worse than both LP concatenation and NoLP in all three matched seeds. These are descriptive Q3 development findings, not statistical support: all mask/coordinate U-Net arms selected epoch 30, the configured ceiling, and the largest mean improvement over persistence was only about `0.11%`. A longer convergence run and paired session-level uncertainty analysis are required before freezing an architecture.

## Context and Orientation

The current model is implemented in `src/wgan_option/models/generator.py` and `src/wgan_option/models/discriminator.py`. The legacy Generator encodes a masked 16x16 current surface to a 2048-vector, encodes LP1024 to 128 dimensions, applies FiLM at three convolution stages, then concatenates surface, text, and Gaussian32 before a large MLP predicts 256 residual cells. The current Critic flattens convolutional current/future features, concatenates a 128-dimensional LP encoding, and applies an MLP score head.

The new fully convolutional path must never flatten the surface. It will preserve 16x16, 8x8, and 4x4 feature maps, apply FiLM using the encoded LP vector, decode with interpolation/convolution and encoder skips, and use a zero-initialized 1x1 output convolution. One mode receives only the existing masked-IV channel; a second mode additionally receives the support mask, normalized moneyness, and normalized log-TTM. The existing identity-positive residual transform must therefore make epoch-zero output exactly equal to the current surface.

The formal output root is `outputs/experiments/rq3_news_first_vol_film_pure_cnn_ablation_3seed_exact_ttm_v1`. It is development-only: train timestamps are before 2023-07-01, validation timestamps are from 2023-07-01 through 2023-09-30, and no Q4/test loader may be materialized.

## Plan of Work

Add normalized constants and fingerprints for the new Generator and Critic modes without changing old fingerprints. Extend model construction so old modes instantiate byte-for-byte compatible module graphs. Implement a FiLM-only dense mode by excluding the encoded text vector from the dense fusion input. Implement one-channel and mask/coordinate-aware fully convolutional FiLM encoder-decoders using bilinear interpolation, convolutional skip fusion, bottleneck noise injection, and a zero-initialized 1x1 residual head. Implement projection Critic scoring as an unconditional surface score plus an inner product between projected LP and surface features.

Add an isolated RQ3 orchestrator and YAML configuration that materialize exactly 21 unique job specs across seven ordered arms and three seeds. The supervisor must use an exclusive lock, persist its PID and status, balance the registry 10/11 across GPU 0 and GPU 1, verify source/config/job hashes on resume, stop on a failed task, and never create a test loader. Postprocessing will report each job's selected epoch, model MAE, persistence MAE, relative improvement, parameter count, and per-arm mean/ranking.

## Concrete Steps

From the repository root:

    /home/haobin_cui/.conda/envs/py312/bin/python -m pytest <new targeted test files>
    /home/haobin_cui/.conda/envs/py312/bin/python -m scripts.rq3.news_first_vol_film_pure_cnn_ablation dry-run --config configs/rq3/news_first_vol_film_pure_cnn_ablation_3seed.yaml --output-dir <temporary root>
    ruff format --check <changed Python files>
    ruff check <changed Python files>
    python -m py_compile <changed Python files>
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

After a one-epoch GPU smoke root passes, launch the formal supervisor with `nohup setsid`, write its log under a sibling control root, and persist the shell PID. Use the orchestrator `status` action to inspect progress without modifying the registry.

## Validation and Acceptance

All old architecture modes must retain their historical parameter counts and outputs for fixed state/input. The FiLM-only dense mode must not consume text in its bottleneck concatenation but must have finite nonzero FiLM gradients. The fully convolutional mode must have no surface flatten/dense reconstruction head, accept the expected 16x16 inputs, produce a 16x16 output, and produce exactly the current surface with the residual head at initialization. Real versus changed LP must affect the model after a controlled FiLM projection perturbation. Projection Critic scores must vary with LP and expose finite nonzero text gradients; same-shape NoLP scores and text gradients must remain invariant/zero.

The orchestrator must generate exactly 21 unique jobs, balanced 10/11 across the two physical GPUs, with no validation/test leakage and deterministic configuration hashes. A completed job may be skipped only after every registered artifact size and SHA matches. Formal launch is accepted only after targeted tests, static checks, and a GPU one-epoch smoke test pass without OOM or NaN.

## Idempotence and Recovery

The formal output root is created only after prelaunch checks. `prepare --resume` and `run-pipeline --resume` may reuse a job only when the registry, generated config, job specification, source hashes, and completed artifacts match. Any drift fails closed. The supervisor lock prevents duplicate launches. A failed task stops the pipeline and remains inspectable; recovery requires fixing the cause, verifying hashes, and explicitly using `--resume`. No command deletes or mutates prior experiment roots.

## Artifacts and Notes

Expected core artifacts include a resolved config, model-contract manifest, 21-job registry, per-job status JSON, generated training configs, resource snapshots, Q3 job summary CSV, arm ranking CSV/JSON, supervisor log/PID, and terminal QA manifest.

## Interfaces and Dependencies

The implementation will expose new branch-local conditioning identifiers through `wgan_option.models.common`, accept them through the existing training configuration loader, and construct them in `Generator` and `Discriminator`. The orchestrator will be available as `python -m scripts.rq3.news_first_vol_film_pure_cnn_ablation` with actions `prepare`, `dry-run`, `worker`, `launch`, `status`, `postprocess`, and `run-pipeline`. It will reuse the existing news-first training entrypoint and exact-TTM data workbook; no new third-party package is required.
