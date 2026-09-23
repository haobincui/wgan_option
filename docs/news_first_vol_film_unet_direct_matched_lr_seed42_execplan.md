# Sweep FiLM learning rate without a parent stage

This ExecPlan is a living document. The sections Progress, Surprises &
Discoveries, Decision Log, and Outcomes & Retrospective must stay current as
work proceeds. Maintain it in accordance with `PLANS.md` at the repository
root.

## Purpose / Big Picture

Test whether the FiLM U-Net's weak text contribution is caused by an
under-sized learning rate on the FiLM projections. Five otherwise identical
matched-LP models start directly from the same seed-42 initialization. The CNN
backbone and NoLP Critic remain at `5e-7`, the text encoder remains at
`2.5e-6`, and only the FiLM projection learning rate changes across
`2.5e-7`, `5e-7`, `1e-6`, `2.5e-6`, and `5e-6`. There is no 30-epoch parent,
no no-text continuation, and no LR replay branch.

The frozen Pure-CNN + No-text experiment is not trained again. Its 500
pair-level predictions and four checkpoint hashes are used as an immutable
contextual reference. Completion is observable through 20 independently
trained checkpoints, 20 frozen prediction cells, 2,500 FiLM pair-metric rows,
an optimizer-contract record for each job, the SHA-bound Pure-CNN comparison,
and passing terminal QA.

The branch-local CLI is
`python -m scripts.rq3.news_first_vol_film_unet_direct_matched_lr_seed42`.
Its formal root is
`outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_seed42_exact_ttm_rolling_v1`.

## Progress

- [x] (2026-08-31 00:00Z) Freeze the five FiLM learning rates, shared group
  learning rates, four rolling folds, seed, architecture, 240-epoch budget,
  Pure-CNN comparator, and descriptive interpretation.
- [x] (2026-08-31 00:40Z) Add the branch-local YAML, thin orchestrator, grouped
  optimizer field forwarding, optimizer-contract training summary, and focused
  orchestrator tests.
- [x] (2026-08-31 00:43Z) Prepare a temporary real-data one-epoch root with
  exactly 20 jobs, 20 matched-LP overlays, split group LRs and zero test files.
- [x] (2026-08-31 00:44Z) Run 59 focused and regression tests, Ruff format/check,
  `py_compile`, and `git diff --check`. The mandatory verification helper was
  also executed and exited 2 solely because this repository has no
  `make format` target.
- [x] (2026-08-31 00:47Z) Run all 20 one-epoch cells concurrently on two A30s;
  all 20 completed with finite metrics and updated G/D states. Peak GPU memory
  was 3.804 GiB, peak host RAM fraction was 0.1415, so formal concurrency is
  frozen at 10 workers/GPU.
- [x] (2026-08-31 00:47Z) Launch detached supervisor PID 3745515 under its
  own session. The formal root registered 20 jobs and all 20 entered running
  state; both A30s reached 99% utilization at about 3.60 GiB each.
- [ ] Complete 20/20 training jobs, freeze all checkpoints, and produce 20/20
  MC64 predictions.
- [ ] Produce the FiLM-LR/Pure-CNN analysis, reports, terminal QA, and complete
  output SHA manifest.

## Surprises & Discoveries

- Observation: the repository has no `v*` release tag, so the optimizer
  profile fields and this CLI are branch-local interfaces.
  Evidence: release-tag discovery returned no matching local or remote tag.
- Observation: the completed Pure-CNN reference is already a direct,
  independent, full-G/D training run with no parent state. Every fold selected
  epoch 240 and used validation MC16 and test MC64.
  Evidence: the frozen root's training summary, resolved job configurations,
  checkpoint allowlist, and prediction manifests.
- Observation: the Pure-CNN and prior direct FiLM runs share the exact same
  fold/pair/session/persistence panels and deterministic per-fold MC64 noise
  bank hashes. They are not capacity matched: Pure CNN has 416,353 Generator
  parameters, while FiLM U-Net has 827,745.
  Evidence: the two frozen `rq12_pair_metrics.csv.gz` tables and model
  contracts.
- Observation: a real-data temporary prepare completed with 20 unique jobs,
  10/10 GPU assignment, 20 matched `lp_mean_l2` overlays, and no evaluation
  files. The first job config recorded backbone/text/FiLM LRs of
  `5e-7 / 2.5e-6 / 2.5e-7` and corresponding floors of
  `5e-8 / 2.5e-7 / 2.5e-8`.
  Evidence: `/tmp/film_lr_prepare.A9sjkf/root`.
- Observation: the project-native verification stack passes, while the
  repository-wide mandatory helper cannot advance past its first command.
  Evidence: 59/59 selected `unittest` cases passed; Ruff and `py_compile`
  passed; `bash .agents/skills/code-change-verification/scripts/run.sh`
  returned exit 2 with `No rule to make target 'format'`.
- Observation: the full-matrix benchmark stayed far below both resource gates
  and required no fallback wave.
  Evidence: `benchmark_result.json` records 20/20 jobs, 1 epoch, 3.8037 GiB
  peak GPU memory, 0.14150 peak RAM fraction, and 10 selected workers/GPU.

## Decision Log

- Decision: use `film_unet_split_lr_v1` with three disjoint Generator groups:
  backbone 416,353 parameters, text encoder 295,808, and FiLM projections
  115,584.
  Rationale: changing only the FiLM group's LR directly diagnoses whether the
  modulation projections are learning too slowly while holding architecture
  and all other optimizer targets fixed.
  Date/Author: 2026-08-31 / Codex.
- Decision: retain ReduceLROnPlateau and set every group's floor to 0.1 times
  its own initial LR.
  Rationale: one shared absolute floor would prematurely pin the low-LR arm or
  permit disproportionate decay in the high-LR arm.
  Date/Author: 2026-08-31 / Codex.
- Decision: materialize five separately hashed overlays whose records are
  byte-equivalent matched LP within each fold.
  Rationale: each arm needs an immutable, arm-addressed job input while text
  representation and pair coverage must not vary with LR.
  Date/Author: 2026-08-31 / Codex.
- Decision: do not retrain Pure CNN.
  Rationale: the existing run is direct, seed 42, four-fold, exact-TTM,
  best-learned, MC16/MC64, and its output manifest and checkpoint allowlist are
  terminal and SHA-addressed. Retraining would add Monte Carlo variation
  without improving the central FiLM-LR intervention. It remains a contextual
  architecture/capacity reference, not a capacity-matched ablation.
  Date/Author: 2026-08-31 / Codex.
- Decision: report the lowest test MAE only as a descriptive point leader.
  Rationale: the same rolling test folds have historical exposure, there is
  only one model seed, and selecting LR from these results would be another
  adaptive test-set choice.
  Date/Author: 2026-08-31 / Codex.

## Outcomes & Retrospective

Implementation, prelaunch validation, and the full benchmark are complete.
Formal training is active under supervisor PID 3745515 with 20/20 jobs
running. Benchmark peaks were 3.8037 GiB GPU memory and 0.14150 host-RAM
fraction, fixing concurrency at 10 workers/GPU. A live checkpoint audit after
startup confirmed that backbone, text-encoder, and FiLM-projection tensors all
changed in every LR arm inspected. Training/evaluation outcomes remain
pending. Update this section with per-group LR trajectories, selected epochs,
the five-arm ranking, all FiLM-vs-Pure paired results, internal LR contrasts,
terminal QA, and the output-manifest SHA. Do not infer completion from idle
GPUs or a quiet supervisor log.

## Context and Orientation

The Generator is `film_unet_mask_coords_v1`, c32. It consumes masked current
volatility, current-support mask, normalized strike coordinate, and normalized
maturity coordinate. LP1024 passes through `1024 -> 256 -> 128` and controls
six FiLM sites. The Generator has 827,745 parameters. The Critic is
`lp_disabled_same_shape_v1`, retains the same historical parameter shape,
zeros its encoded LP signal, and has 729,157 parameters.

The exact-TTM maturity axis is
`[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]`. The surface is 16 by 16, support
mode is `raw_joint`, current input is `current_support_masked`, output is the
identity residual construction, noise is Gaussian32, batch size is 16, and
there are five Critic steps per Generator step. Every market pair is one
training sample; articles are deduplicated and their LP1024 vectors are
averaged then L2 normalized.

The four test folds are 2023Q1 through Q4 and contain 110, 112, 135, and 143
pairs, or 500 pairs and 148 fold-session clusters per arm. Five LRs times four
folds gives exactly 20 jobs. Validation uses MC16. Test prediction uses MC64
with the same deterministic per-fold/pair noise bank as the frozen Pure-CNN
reference.

The core backward-compatible fields are:

    generator_optimizer_profile: film_unet_split_lr_v1
    generator_learning_rate: 5e-7
    generator_text_learning_rate: 2.5e-6
    generator_film_learning_rate: <arm value>
    reduce_lr_min_lr: 5e-8
    generator_text_min_learning_rate: 2.5e-7
    generator_film_min_learning_rate: <0.1 * arm value>

Historical specs that omit these fields retain the original uniform Adam
construction and do not gain new serialized keys.

## Plan of Work

Resolve and validate the YAML, including all Pure-CNN paths and SHA-256
digests. Materialize the four rolling pair universes without opening test rows.
Build the canonical matched LP development overlay once per fold, then write
five arm-addressed copies with identical records and distinct namespaces. Each
job starts from the common seed-42 Generator/Critic state, trains directly for
at most 240 epochs, and selects its own learned checkpoint using validation
`val_hybrid_score`, minimum epoch 30, patience 20.

The Generator Adam contains three named, complete, disjoint groups. The
Critic retains its own Adam. The two plateau schedulers remain enabled. Every
epoch records backbone, text, FiLM, and Critic LR. Postprocessing packages the
configured, initial, final, and full per-epoch trace plus parameter count for
all four groups into each row's `optimizer_contract_json`.

After all 20 checkpoints and hashes freeze, materialize four test panels and
20 matched-LP test overlays. Produce 20 MC64 predictions and exactly 2,500
FiLM pair rows. Join them only to the SHA-bound 500-row Pure-CNN evidence after
requiring identical fold, pair, session, effective origin, persistence error,
and MC-noise profile.

Report five paired FiLM-vs-Pure comparisons with fold-to-CME-session bootstrap
and Holm adjustment. Internal LR comparisons are development diagnostics and
cannot authorize an LR choice. All outputs carry
`retrospective_rolling_development_single_seed_descriptive`.

## Concrete Steps

Run from the repository root using the py312 environment:

    /home/haobin_cui/.conda/envs/py312/bin/python -m unittest tests.test_scripts.test_rq3_news_first_vol_film_unet_direct_matched_lr_seed42 -v
    /home/haobin_cui/.conda/envs/py312/bin/python -m unittest tests.test_scripts.test_grouped_generator_optimizer -v
    ruff format --check scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_lr_seed42.py
    ruff check scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_lr_seed42.py
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py scripts/rq123/news_first_vol_film_nolp_10seed.py
    git diff --check
    bash .agents/skills/code-change-verification/scripts/run.sh

Run the complete one-epoch matrix benchmark before creating the formal root:

    /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_lr_seed42 \
      benchmark \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_seed42_exact_ttm_rolling_v1

Use 10 workers/GPU only when both A30 peaks stay below 20 GiB, host RAM stays
below 85%, every G/D updates, and all metrics are finite. Otherwise rerun the
frozen six-workers/GPU fallback in two waves. A second gate failure prevents
formal-root creation.

After the benchmark and all gates pass, launch exactly one detached supervisor:

    nohup setsid env PYTHONPATH=src:. \
      /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_lr_seed42 \
      run-pipeline --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_seed42_exact_ttm_rolling_v1 \
      > outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_seed42_exact_ttm_rolling_v1_control/pipeline.log \
      2>&1 < /dev/null &

## Validation and Acceptance

The resolved config must produce exactly 20 unique jobs, with 10 assigned to
each physical GPU and all five arms in a fold on the same GPU. Every job must
have `pair_text_overlay_mode=lp_mean_l2`, the split optimizer profile, text LR
`2.5e-6`, and the arm's declared FiLM LR/floor. All epoch-zero Generator hashes
and all epoch-zero Critic hashes must agree across all arms and folds.

All five development and test overlay record arrays must match within each
fold. Before checkpoint freeze, there must be no test loader, test input, or
prediction. Every completed job must show changed G/D parameters, finite
metrics, and a complete optimizer contract containing counts
`416353 / 295808 / 115584 / 729157` plus a contiguous epoch-0-through-final LR
trace for backbone, text encoder, FiLM projection, and Critic.

The Pure-CNN output manifest, pair metrics, checkpoint allowlist, and four
Generator checkpoints must match their declared paths, sizes, and SHA-256
digests. The comparison must reject any pair/session/origin/persistence/noise
lineage mismatch.

Terminal acceptance requires 20 completed jobs, 20 checkpoint-allowlisted
prediction cells, exactly 2,500 new pair rows, the 500-row frozen Pure-CNN
reference, complete statistics/reports, passing `qa.json`, and a complete
output SHA manifest. The report must state that the result is single-seed,
retrospective, descriptive, not capacity matched, and not permission to select
an LR on these test folds.

## Idempotence and Recovery

The supervisor owns an exclusive lock, PID file, and stage journal. Resume may
skip a job only when source, config, dataset, support, overlay, optimizer
fields, job spec, and every artifact size/SHA match. Any Pure-CNN reference
drift fails validation before training. A terminal-complete root is strictly
read-only. No command mutates the frozen Pure-CNN root or any previous FiLM
experiment.

## Artifacts and Notes

Expected artifacts include the resolved config, source/code/config hash
manifests, pair-universe manifest, 20 matched-LP development overlays, 20-job
registry, 20 selected checkpoint pairs, evaluation checkpoint allowlist, four
test panels, 20 test overlays, 20 prediction manifests, 2,500-row
`analysis/rq12_pair_metrics.csv.gz`, 20-row training summary with optimizer
contracts, FiLM-LR ranking/comparison/bootstrap tables, group LR trace, Markdown
and self-contained HTML reports, resource summary, `qa.json`, and
`output_hashes.csv`.

## Interfaces and Dependencies

The CLI actions are `benchmark`, `prepare`, `dry-run`, `worker`, `launch`,
`freeze-evaluation`, `predict`, `postprocess`, `qa`, `status`, and
`run-pipeline`. The new grouped optimizer fields are opt-in; omitted historical
specs preserve `uniform_v1`. The implementation introduces no new third-party
dependency.
