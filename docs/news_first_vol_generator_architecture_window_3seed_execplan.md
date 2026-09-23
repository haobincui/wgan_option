# Implement the generator-architecture and alignment-window studies

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries,
Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds.
It is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The change adds two independent, reproducible experiments for the 16 by 16
exact-TTM volatility-surface dataset. The architecture study holds the five-minute
news alignment fixed and compares three new text-conditioning mechanisms against
the existing FiLM U-Net and pure-CNN baselines. The window study holds the existing
FiLM U-Net and pure-CNN architectures fixed and varies only the maximum wait from
news availability to the first market-pair origin: 5, 10, 15, 20, or 30 minutes.
Every model continues to predict the next five-minute surface.

The observable outcome is two terminal experiment roots containing immutable job
registries, checkpoint allowlists, common-noise predictions, paired statistical
comparisons, reports, and SHA manifests:

    outputs/experiments/rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1
    outputs/experiments/rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1

## Progress

- [x] (2026-09-02 19:40Z) Read `PLANS.md`, the implementation-strategy skill, and the mandatory code-change verification skill.
- [x] (2026-09-02 19:44Z) Confirmed that the repository has no `v*` release tag and chose additive branch-local model and schema names with no migration shim.
- [x] (2026-09-02 19:50Z) Audited the existing FiLM/pure-CNN implementation, rolling-data contracts, terminal five-minute baselines, and frozen checkpoint manifests.
- [x] (2026-09-02 20:02Z) Added backward-compatible 20-minute tolerance support to the alignment and surface-building code and its focused tests.
- [x] (2026-09-02 21:28Z) Built and validated the new five-window data root; all 61 output hashes, the fixed five-minute horizon, historical-window equivalence, and `5 subset 10 subset 15 subset 20 subset 30` passed.
- [x] (2026-09-03 11:36Z) Merged and tested the three versioned Generators and the `conditioning_split_lr_v2` optimizer contract after the historical-root source gate opened.
- [x] (2026-09-02 22:36Z) Finished and independently audited the shared orchestrator, worker-config materialization, prediction adapters, statistical families, reports, QA, process-group cleanup, and fail-closed recovery behavior.
- [x] (2026-09-02 23:15Z) Replaced hand-entered matched-capacity profiles with deterministic executable grid search and a signed, registry-bound matched-capacity manifest.
- [x] (2026-09-02 23:16Z) Ran a real CUDA probe for all three staged architectures; all parameter roles had finite nonzero gradients after three updates and peak allocated memory remained below 132 MiB for the probe batch.
- [x] (2026-09-03 11:37Z) Ran 207 focused and legacy unit tests (one intentional skip), Ruff format/check, `py_compile`, both production dry-runs, and `git diff --check`; all native checks passed. The mandatory wrapper was also run and stopped only because this repository has no `make format` target.
- [x] (2026-09-03 11:34Z) Waited for the active 280-task historical experiment to reach terminal QA, confirmed its related Python process count was zero, reran its full root validator, and verified all 288 frozen code-manifest rows before changing shared runtime code.
- [ ] Run canaries and concurrency benchmarks, freeze the largest safe worker count, and launch both supervisors in the background.

## Surprises & Discoveries

- Observation: another formal 280-task experiment is still running from this same
  checkout and validates every Python file below `src/` by SHA before resuming.
  Evidence: its continuation registry still contained running and pending jobs, and
  `stages/continuations/film_unet/code_hashes.csv` contained 288 frozen paths.

- Observation: editing a shared model file while that experiment is active would
  make its fail-closed resume reject otherwise valid checkpoints. The briefly
  touched files were restored byte-for-byte before any old job consumed drifted
  code, and all 288 frozen hashes were revalidated.

- Observation: the label `c64` is not an acceptable approximately doubled FiLM
  capacity. The instantiated FiLM U-Net has 2,115,393 Generator parameters at c64,
  whereas c54 has 1,631,823 and is inside the requested 1,655,490 plus or minus five
  percent interval.

- Observation: the existing exact-TTM data root has 5, 10, 15, and 30-minute
  workbooks but no 20-minute workbook. A new root is required; historical data and
  experiment roots must remain immutable.

- Observation: the mandatory code-change verification wrapper cannot advance past
  its first command because this research repository has no `make format` target.
  Evidence: the wrapper exited 2 with `No rule to make target 'format'`; the
  project-native Ruff, compile, dry-run, diff, and 207-test stack passed separately.

## Decision Log

- Decision: treat all new model modes, optimizer fields, manifests, and CLIs as
  branch-local interfaces while preserving historical defaults and checkpoint
  loading behavior.
  Rationale: no release tag exists, and additive versioned names avoid changing the
  meaning of any frozen experiment.
  Date/Author: 2026-09-02, Codex.

- Decision: do not modify shared `src/` files until the active historical
  experiment has stopped spawning workers that verify the old source manifest.
  Rationale: completing the requested new work must not corrupt a separately
  authorized training run.
  Date/Author: 2026-09-02, Codex.

- Decision: select matched and scaled capacity profiles by deterministic integer
  grid instantiation and exact parameter counting, not by capacity labels.
  Rationale: the relationship between base channels and parameters differs across
  CrossAttn, Transformer, StyleMod, and FiLM.
  Date/Author: 2026-09-02, Codex.

- Decision: keep primary multiple-testing families separate from secondary and
  capacity diagnostics. The architecture primary family is the three new modes
  versus FiLM (Holm-3); the window families are four non-five-minute windows versus
  five minutes per model (Holm-4) and FiLM versus pure CNN at five windows (Holm-5).
  Rationale: combining unrelated planned families would change the specified
  estimands and their error control.
  Date/Author: 2026-09-02, Codex.

- Decision: generate only matched and zero-text counterfactuals in the window
  study, while retaining matched, zero, and shuffled-text counterfactuals in the
  architecture study.
  Rationale: this is the exact planned scope; it yields 324 window evaluation
  units without adding an unplanned window-level shuffled-text family.
  Date/Author: 2026-09-02, Codex.

- Decision: benchmark worker counts in ascending order and safety-skip all larger
  candidates after the first memory or correctness failure.
  Rationale: GPU memory pressure is monotone for these identical-process worker
  pools, so launching a larger candidate after a smaller one fails would add OOM
  risk without a path to acceptance. Every candidate up to the first unsafe count
  is physically measured; skipped candidates are recorded explicitly.
  Date/Author: 2026-09-02, Codex.

- Decision: finish the repository implementation and verification before ending
  this interaction, but do not wait in-dialogue for formal multi-hour training
  results.
  Rationale: the user explicitly requested that the conversation may end once the
  code modification is complete. Formal execution remains protected by its GPU
  benchmark and source-lineage gates.
  Date/Author: 2026-09-02, Codex.

## Outcomes & Retrospective

Implementation and formal execution are in progress. This section will record the
frozen parameter profiles, 20-minute counts, benchmark concurrency, supervisor
PIDs, and terminal verification evidence after they exist.

The production runtime now instantiates CrossAttn, Transformer-token, and StyleMod
Generators with exact matched-capacity parameter counts `824,644`, `834,561`, and
`829,635`; the fixed Critic has `729,157` parameters. Both dry-runs pass their
input, baseline, and runtime preflights without creating a formal root or reading
test metrics. Historical Generator modes retain their golden initialization,
fingerprint, checkpoint-loading, optimizer, and residual-output contracts.

The completed data milestone produced 1,371 strict-support 20-minute pairs. Its
rolling train/validation/test pair and session counts are F1
`508/104, 190/45, 149/39`; F2 `698/149, 149/39, 153/39`; F3
`847/188, 153/39, 177/40`; and F4 `1000/227, 177/40, 194/51`.
The data manifest file SHA-256 is
`d1ce4af5e6e1550086a8674ff6fc75533285156de4aa5094be457b2c784f92a5`.

The completed orchestration milestone freezes 69 new architecture jobs and 96 new
window jobs. It binds 24 read-only baseline cells in each line, produces 324
window evaluation units, and rejects drift in materialized scientific contracts,
source configuration, checkpoints, raw prediction cells, downstream bundles, or
terminal output manifests. Matched-capacity selection now instantiates and hashes
the complete candidate grid rather than trusting capacity labels. Its current
task-focused CPU suite passes 46 tests.

## Context and Orientation

The executable Generator is `src/wgan_option/models/generator.py`; supported model
contracts are in `src/wgan_option/models/common.py`; YAML configuration is parsed by
`src/wgan_option/config.py`; optimizers and schedulers are built in
`src/wgan_option/models/gan_model.py`; checkpoint reconstruction is in
`src/wgan_option/utils/inference_helpers.py`. Existing FiLM and pure-CNN direct-run
orchestration under `scripts/rq3/` provides the training and prediction semantics
that the new shared orchestrator must retain.

A market pair contains a current surface for `[t-5,t)` and target surface for
`[t,t+5)`. Alignment tolerance is only the maximum wait between news availability
and pair origin. It must never alter the five-minute forecasting horizon. The
four rolling folds train strictly before validation and test boundaries. Test
loaders and test metrics cannot exist before the checkpoint allowlist is frozen.

The architecture experiment adds exactly 69 training jobs: nine validation-only
learning-rate screens, 36 matched-capacity formal jobs, and 24 scaled jobs. It also
binds 24 terminal baseline cells without retraining them. The window experiment
has 120 logical cells, reuses 24 terminal five-minute cells, and adds exactly 96
training jobs for the four new tolerances.

## Plan of Work

First build a new exact-TTM data root containing all five cumulative tolerances and
verify canonical pair nesting and historical-window equivalence. Next add the three
Generator modes with identity-initialized residual paths and deterministic parameter
roles named `backbone`, `text_encoder`, and `conditioning_module`. Extend config,
training, scheduler, and inference construction additively.

Then complete a shared orchestrator with thin architecture/window YAML profiles.
Preparation freezes data, code, model, baseline, and job hashes. Screening reads
validation only. Formal architecture selection reads validation only. Evaluation
is opened only after every required checkpoint is immutable. Predictions use a
shared MC64 noise bank and include matched/zero/shuffled counterfactuals for each
text model. Window predictions include both a common five-minute panel and each
window's own panel, plus the pre-news/partial-post-news/full-post-news relation.

Finally run focused and regression tests, perform the required resource canaries,
freeze concurrency, and start both idempotent supervisors. A failed worker stops
its pipeline. Resume is permitted only when every recorded input and artifact SHA
still matches.

## Concrete Steps

All commands run from the repository root with `PYTHONPATH=src:.` and the py312
interpreter where shown.

    python scripts/rq3/main.py build-news-first-vol-surfaces \
      --config configs/rq3/news_first_vol_surfaces_narrow_grid_exact_ttm_with20.yaml \
      --output-dir data/processed/rq3/news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_with20_v1

    python -m unittest tests.test_scripts.test_rq3_news_first_vol_alignment
    python -m unittest tests.test_scripts.test_rq3_news_first_vol_surfaces
    python -m unittest tests.test_scripts.test_news_first_vol_architecture_window_study

    ruff format --check <changed Python files>
    ruff check <changed Python files>
    python -m py_compile <changed Python files>
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

After prelaunch QA, invoke each profile's `run-pipeline --resume` through `nohup
setsid`, redirect logs into a sibling `_control` directory, and write PID plus
process-start ticks so stale PID reuse cannot be mistaken for a live supervisor.

## Validation and Acceptance

Acceptance requires exact finite parameter counts inside the specified capacity
intervals; four-channel mask-and-coordinate inputs; persistence-equivalent epoch
zero output within `1e-7`; finite nonzero gradients in each new text-conditioning
mechanism; exact NoLP Critic invariance; no old-mode checkpoint regression; and
disjoint complete optimizer parameter roles.

Data acceptance requires an unchanged five-minute target horizon, exact-TTM axes,
no rolling leakage, and canonical nesting `5 subset 10 subset 15 subset 20 subset
30`. Orchestration acceptance requires exact 69 and 96 new-job registries, frozen
baseline SHAs, no test access before evaluation freeze, complete common-noise
predictions, correct bootstrap/Holm families, terminal QA, and output SHA manifests.

## Idempotence and Recovery

Historical roots are read-only. New formal roots are never silently overwritten.
Preparation with `--resume` verifies config, data, code, baseline, and registry
hashes. Completed jobs are skipped only after artifact path, size, SHA, and job-spec
lineage checks. Partial or failed jobs retain attempts and may restart; a hash drift
requires a new root rather than weakening the check. Supervisors use exclusive
locks and validate PID start ticks.

## Artifacts and Notes

The 20-minute builder log is written under
`outputs/experiments/news_first_vol_surfaces_exact_ttm_with20_v1_control/`. Formal
pipeline logs and PIDs will live in `_control` siblings of the two experiment roots.

## Interfaces and Dependencies

The final model interface exposes the three exact slugs
`crossattn_unet_mask_coords_v1`, `transformer_tokens_mask_coords_v1`, and
`stylemod_unet_mask_coords_v1`. The new optimizer profile is
`conditioning_split_lr_v2`; every trainable Generator parameter belongs to exactly
one of `backbone`, `text_encoder`, or `conditioning_module`.

The shared CLI exposes `prepare`, `benchmark`, `screen-lr`, `freeze-screen`,
`launch-training`, `freeze-selection`, `launch-scaled`, `freeze-evaluation`,
`predict`, `analyze`, `bootstrap`, `report`, `qa`, `status`, and `run-pipeline`.
Configuration and manifests are YAML/JSON/CSV files with explicit schema versions
and SHA-256 lineage.
