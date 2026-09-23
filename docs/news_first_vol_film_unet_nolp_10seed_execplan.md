# FiLM U-Net + NoLP Critic: 10-Seed RQ1--RQ3 ExecPlan

This is a living execution record for the branch-local unified experiment. It
must be updated from the frozen supervisor journal and manifests; a task is not
complete merely because its process exited successfully.

## Objective

Run one lineage-closed rolling experiment using
`film_unet_mask_coords_v1` at c32 and
`lp_disabled_same_shape_v1`. The generator consumes a four-channel tensor
(current surface, current-support mask, strike coordinate, maturity
coordinate), maps LP1024 through 256 and 128 dimensions, and injects six FiLM
modulations. The critic retains its text branch shape but zeros the encoded
text feature exactly.

The immutable parameter contract is:

| Component | Parameters |
|---|---:|
| Generator | 827,745 |
| Critic | 729,157 |
| Total | 1,556,902 |

The formal root is
`outputs/experiments/rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240_exact_ttm_rolling_v1`.
The old legacy v2 root and the earlier five-seed Q3 U-Net root are read-only
evidence; neither is a continuation parent for this experiment.

## Frozen design

- Seeds: `42, 202, 404, 382624741, 1607127774, 1662128673, 2041145538,
  2014889368, 1343862330, 779214671`.
- Folds: rolling tests for 2023Q1, Q2, Q3, and Q4.
- Alignments: 5 minutes and 30 minutes; the forecast target is always the
  future five-minute surface.
- Grid: exact-TTM 16 by 16 with maturity days
  `[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]`.
- Surface semantics: `raw_joint`, `current_support_masked`, and identity
  residual output.
- Optimisation: G/D LR `5e-7`, floor `5e-8`, zero warmup, batch 16, five
  critic steps, validation MC 16, prediction MC 64.
- Interpretation: `retrospective_rolling_development`.

For every tolerance, fold, and seed block, `parent_current_only` starts from a
fresh deterministic initialization and runs at most 30 epochs. The selected
parent full state is restored by `continuation_no_text`, which runs at most 240
additional epochs with minimum 30 and patience 20. Its selected added epoch
count `E`, plus both per-epoch LR traces, is frozen into a branch recipe. LP,
shuffled LP, BoW, and sentiment branches begin from the same parent state and
run exactly `E` epochs with the frozen LR traces. They do not early-stop or
select an internal checkpoint. `E=1` is a valid frozen result and must not be
changed retrospectively.

The matrix has 80 parents, 80 no-text continuations, and 240 text branches,
for exactly 400 training jobs. The same allowlist produces exactly 400 frozen
prediction cells. GPU assignment is block-preserving and balanced at 40 blocks
per physical GPU.

## Statistical outputs

RQ1 and RQ2 use the 5-minute panel. RQ1 compares LP with no-text and shuffled
LP. RQ2 compares LP with BoW and sentiment. Each family uses 10,000 paired
`seed -> fold -> CME session` bootstrap replicates and Holm-2. Statistical
support additionally requires at least seven of ten seeds and three of four
folds to be non-worse.

RQ3 uses the 30-minute alignment dataset but retains the five-minute forecast
horizon. Scheduled `[0,+30]` is primary, `[-10,+20]` is robustness, and
`[0,+5]` is descriptive. Market-jump matching remains frozen. The expected 21
matched pairs across 20 sessions does not satisfy the 30-pair inferential
threshold and therefore must produce `not_estimable_due_to_coverage`; the
threshold must not be relaxed after errors are inspected.

## Execution gates

The single idempotent supervisor executes these gates in order:

1. Verify the full-state parent/continuation restore canary.
2. Run the 36-job mixed 5m/30m dual-A30 benchmark.
3. Run the complete 400-cell one-epoch smoke matrix.
4. Freeze runtime concurrency only if GPU peak is below 20 GiB and host RAM is
   below 85%. Prefer 18 workers per GPU; use 12 only after a recorded failed
   18-worker gate. A second failure prevents formal-root creation.
5. Prepare and freeze inputs, profile, code, runtime, task registry, and
   parent-stage allowlist.
6. Train parents, freeze their selected full states, train continuations, and
   freeze all 80 branch recipes.
7. Train text branches, freeze the 400-checkpoint evaluation allowlist, then
   and only then materialize test predictions.
8. Evaluate RQ1, RQ2, and RQ3, build Markdown/HTML reports, and run terminal
   QA plus the complete output SHA manifest.

Resume is allowed only when config, source, input, parent state, recipe, job
specification, and artifact size/SHA all match. Any drift or tampering is a
fail-closed condition. A terminal-complete root must remain read-only.

## Progress

- [x] Freeze the U-Net/NoLP experiment profile, model widths, seeds, folds,
  task counts, epoch policy, and statistical estimands.
- [x] Add the thin U-Net CLI wrapper and independent YAML configuration.
- [x] Parameterize the shared orchestrator without duplicating the legacy
  implementation or changing the legacy defaults.
- [x] Add regression coverage for profile restoration, recursive source
  lineage, model/profile hashes, exact parameter counts, matrix/GPU balance,
  epoch/LR recipes, zero warmup, worker dispatch, legacy compatibility, and
  U-Net full-state next-update equivalence.
- [x] Complete formatting, static checks, targeted tests, and legacy status
  regression.
- [x] Launch the supervisor and record PID, lock, journal stage, and resource
  snapshot.
- [ ] Complete the recovery canary, benchmark, and 400-cell smoke gates.
- [ ] Complete 400/400 formal training jobs and freeze 80 branch recipes.
- [ ] Complete and freeze 400/400 prediction cells.
- [ ] Produce RQ1/RQ2/RQ3 analysis, reports, terminal QA, and output hashes.

## Evidence to record

After launch, append dated entries containing only facts read from artifacts:
supervisor PID, control-log path, current journal stage, completed/failed job
counts, selected workers per GPU, peak GPU/RAM observations, branch `E`
distribution, prediction count, RQ inferential status, terminal-QA result, and
the final output-manifest SHA. Do not infer completion from a quiet log or from
GPU utilization alone.

## Decisions and discoveries

- There is no `v*` release tag, so the wrapper/config schema is a branch-local
  interface. No migration shim is required.
- Sharing the audited orchestrator is intentional: the wrapper temporarily
  installs an immutable model/experiment profile and restores every global,
  including after exceptions. Worker subprocesses use the wrapper module so
  they cannot silently fall back to the legacy architecture.
- The five-seed Q3 checkpoints are excluded from formal continuation because
  they do not contain the required rolling-fold, 30-minute, parent full-state,
  and common-branch lineage.
- Prelaunch native verification passed 150 targeted tests in total, including
  the 12 tests in the new U-Net profile module, Ruff
  check/format for all touched runtime and test files, `py_compile`, YAML
  parsing, `git diff --check`, an explicit parent job-payload build, and a
  read-only status query of the completed 400/400 legacy v2 root.
- A repository-wide Ruff format audit still identifies two unrelated,
  pre-existing files (`audit_sentiment_scores.py` and `corrected_pipeline.py`)
  outside this experiment profile. They were intentionally not rewritten.

## 2026-08-28 launch record

- Supervisor PID/SID: `1965918`; detached parent PID is `1`.
- Control log: `outputs/experiments/rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240_exact_ttm_rolling_v1_control/pipeline.log`.
- Exclusive lock and `pipeline.pid` were created by the supervisor; the saved
  PID matches the live process.
- Launch-time resources: GPU 0/1 each used 14 MiB at 0% utilization; filesystem
  free space was approximately 370 GiB.
- The formal root was absent at launch. The supervisor was in preflight source
  and input hashing before the recovery-canary/benchmark journal appeared.
