# Implement the F4 FiLM-CNN / Pure-CNN capacity robustness experiment

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

After this change, a researcher can run one isolated rolling-F4 experiment that compares FiLM-CNN and Pure-CNN at capacities c08, c12, c16, c24, c32, and c48. Every fit trains only on observations before 2023-07-01, selects its checkpoint on the 2023Q3 validation window, and evaluates once on the common 143-pair 2023Q4 test panel. The experiment produces auditable pair-level losses, same-architecture c32-relative capacity improvements, and the single-panel LaTeX table used by Chapter 3.

The observable entry point is `python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed <action>`. A completed run has a frozen checkpoint allowlist, 5,148 pair-metric rows, a six-row summary, a generated LaTeX fragment, terminal QA, and an output hash manifest.

## Progress

- [x] (2026-09-24 00:00Z) Audited the existing rolling-F4, FiLM-CNN, Pure-CNN, capacity, and chapter-table contracts.
- [x] (2026-09-24 00:00Z) Locked the six capacity profiles, c48 parameterization, three seeds, F4 time windows, and same-architecture c32 improvement definition with the user.
- [x] (2026-09-24 00:00Z) Confirmed there is no release tag; the new CLI/config/artifact schema is branch-local and needs no compatibility shim.
- [x] (2026-09-24 13:07Z) Implemented the versioned config, independent F4-only training lifecycle, CLI registration, source-bound registry, and pre-Q4 loader gate.
- [x] (2026-09-24 13:05Z) Implemented pair-metric validation, aggregation, summary artifacts, and the single-panel LaTeX renderer.
- [x] (2026-09-24 13:05Z) Added focused CLI/matrix/fold/c48 and aggregation/formatting/tamper tests; 13 focused tests passed.
- [x] (2026-09-24 13:07Z) Ran the full 36-job one-epoch resource benchmark at 18 workers/GPU: 36/36 passed, GPU peaks 6.242/6.236 GiB, host-RAM peak 28.71 percent, finite metrics, and both networks updated.
- [x] (2026-09-24 15:34Z) Completed all 36 formal fits, froze 72 Generator/Critic checkpoint rows, evaluated all 36 cells on F4, postprocessed 5,148 pair rows, and passed terminal QA.
- [x] (2026-09-24 15:34Z) Replaced the old Chapter 3 capacity diagnostics with the generated F4 table and refreshed bindings and audit notes.
- [x] (2026-09-24 15:34Z) Ran the repository verification stack and recorded the final artifacts and limitations below.

## Surprises & Discoveries

- Observation: the old six-profile U-Net sweep cannot supply this table because its Text128 arm uses an LP-enabled Critic, runs for at most 60 epochs, and has no F4 test predictions.
  Evidence: `configs/rq3/news_first_vol_film_unet_capacity_epoch60_3seed.yaml` and its execution plan.

- Observation: the standard F4 fold is not a pre-Q4 refit. It trains before 2023-07-01, validates on 2023Q3, and tests on 2023Q4.
  Evidence: `configs/rq3/news_first_vol_film_unet_text_10seed_lr2p5e5.yaml` and `configs/rq3/news_first_vol_cnn_unet_pure_no_text_10seed.yaml`.

- Observation: a shared capacity name means shared spatial/Critic widths, not equal total parameters. FiLM-CNN retains its text encoder and FiLM projections.
  Evidence: executable construction gives c32 totals 1,556,902 versus 1,145,510 and c48 totals 2,776,184 versus 2,307,000 for FiLM-CNN and Pure-CNN respectively.

- Observation: evaluating all remaining cells concurrently at MC64 exhausted activation memory for 10 FiLM-CNN cells even though the one-epoch training benchmark was comfortably below the GPU threshold.
  Evidence: 24 cells completed in the first evaluation wave; resuming the unchanged frozen allowlist at five workers per GPU completed the remaining 10 cells without recomputing successful cells.

- Observation: the new c32 runs reproduce the immutable c32 MAE anchors within the frozen `5e-5` relative tolerance and use the identical 143-pair / 45-session panel, but their checkpoint and noise-profile hashes are not bitwise identical.
  Evidence: `analysis/f4_c32_regression_control.json` records both numerical checks as passed and `exact_determinism_claimed=false`.

## Decision Log

- Decision: use capacities c08, c12, c16, c24, c32, and c48; remove c04.
  Rationale: c48 scales all c32 width-related dimensions by 1.5 and continues the existing capacity ladder.
  Date/Author: 2026-09-24 / user and Codex.

- Decision: use the standard direct rolling F4 contract and remove staged refit/LR replay.
  Rationale: the requested dataset is the chapter's fold-level F4; a pre-Q4 refit estimates a different procedure.
  Date/Author: 2026-09-24 / user and Codex.

- Decision: compute improvement against c32 separately inside each architecture.
  Rationale: a common FiLM-c32 denominator would mix capacity and architecture effects for Pure-CNN.
  Date/Author: 2026-09-24 / user and Codex.

- Decision: rerun all 36 cells, including c32, and use the immutable completed c32 roots only as regression controls.
  Rationale: the final table should have one experiment lineage and one frozen evaluation contract.
  Date/Author: 2026-09-24 / Codex.

- Decision: implement a new versioned runner and output root rather than modify completed capacity runners.
  Rationale: existing roots are thesis audit evidence and contain experiment-specific lifecycle/recovery assumptions.
  Date/Author: 2026-09-24 / Codex.

- Decision: freeze formal concurrency at 18 workers per GPU and launch all 36
  jobs in one wave.
  Rationale: the experiment-specific 36-job benchmark passed every integrity
  gate with only 6.242 GiB peak GPU memory and 28.71 percent peak host-RAM use;
  this is also the maximum useful concurrency because only 18 jobs are assigned
  to each of the two GPUs.
  Date/Author: 2026-09-24 / Codex.

- Decision: recover the MC64 evaluation from its frozen checkpoint allowlist at five workers per GPU after the maximum-concurrency evaluation wave produced FiLM activation-memory failures.
  Rationale: evaluation has a different memory profile from training; resume validated every source and checkpoint hash, preserved the 24 completed cells, and filled only the 10 missing cells.
  Date/Author: 2026-09-24 / Codex.

## Outcomes & Retrospective

The complete F4 experiment is frozen under
`outputs/experiments/rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1`.
All 36 fits completed, the 72-row checkpoint allowlist was frozen before the Q4
loader was created, and all 36 architecture-capacity-seed cells were evaluated
on the shared 143-pair / 45-session panel. The final analysis contains 5,148
pair rows and six capacity rows. Both architectures attain their lowest
three-seed MAE at c32: `0.0016690756` for FiLM-CNN and `0.0016692538` for
Pure-CNN; consequently all non-reference capacity improvements are negative.

The generated LaTeX fragment is bound into Chapter 3 under
`tab:ch3:f4_film_pure_capacity_robustness`; the old historical table is marked
as superseded. Terminal QA reports `status=passed`. The immutable c32 controls
also pass their numerical tolerance and panel-identity checks, while correctly
making no claim of bitwise checkpoint determinism. The repository-native full
test suite, focused tests, Ruff, Python compilation, binding builder/verifier,
and `git diff --check` pass. No TeX engine is installed in this environment, so
the seven-column table received static structure/width checks but not a final
rendered-PDF inspection.

## Context and Orientation

The working directory is `/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone`. `src/wgan_option` contains the executable WGAN models and training stack. `scripts/rq3` contains research experiment supervisors. `configs/rq3` freezes experimental contracts. `docs/chapter3.tex` is the thesis chapter consuming the final table.

FiLM-CNN is `film_unet_mask_coords_v1` with a matched 1,024-dimensional LP embedding compressed through 256 and 128 dimensions. Pure-CNN is `cnn_unet_mask_coords_v1` and consumes no text. Both use `lp_disabled_same_shape_v1` for the Critic. F4 means training timestamps before 2023-07-01, validation timestamps in `[2023-07-01, 2023-10-01)`, and test timestamps in `[2023-10-01, 2024-01-01)`, all in UTC and keyed by `effective_origin_utc`.

The three seeds are 42, 202, and 404. The six capacity profiles contain 36 training jobs. Every architecture-capacity cell has three selected checkpoints and 429 test loss rows. The complete evaluation has 5,148 rows.

## Plan of Work

Add `configs/rq3/news_first_vol_f4_film_pure_capacity_3seed.yaml` with the exact data, fold, architecture, capacity, optimizer, loss, seed, runtime, count, and hash contracts. Add `scripts/rq3/news_first_vol_f4_film_pure_capacity_3seed.py` as a clean lifecycle supervisor. It may reuse stable low-level data/training/evaluation helpers but must own its experiment kind, registry, output root, and recovery rules. Its actions are benchmark, prepare, dry-run, launch, freeze-checkpoints, evaluate-f4, postprocess, worker, qa, and status.

Preparation may materialize train and validation windows but not the Q4 test dataset. Launch runs 36 independent fits and validation-selects each best learned checkpoint. Freeze-checkpoints accepts only a complete 36-cell registry and writes a SHA-bound allowlist. Evaluate-f4 validates that allowlist before materializing the common Q4 panel and must use one deterministic MC64 noise-bank contract across all capacities and both architectures.

Add `scripts/rq3/news_first_vol_f4_film_pure_capacity_3seed_analysis.py`. It validates exactly 5,148 rows, identical seed-specific pair panels, and the required source/checkpoint/noise hashes. It averages pair losses within seed and then gives the three seeds equal weight. It computes `100 * (1 - candidate_mae / same_architecture_c32_mae)` from unrounded values. It emits pair metrics, summary CSV/JSON, and a complete single-panel LaTeX fragment without persistence or inferential columns.

Replace the obsolete Chapter 3 capacity-development subsection with an F4-only robustness subsection and generated final values. Update the chapter binding builder, external-table verification, and audit notes so source paths, table label, hashes, filters, and formula are recoverable. Preserve old experiment artifacts as superseded provenance.

## Concrete Steps

From the repository root:

    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed benchmark
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed prepare
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed dry-run --resume
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed launch --resume
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed freeze-checkpoints --resume
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed evaluate-f4 --resume
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed postprocess --resume
    python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed qa --resume

The benchmark must report both Generator and Critic updates, finite metrics, peak GPU memory below 20 GiB per GPU, and host RAM below 85 percent before formal concurrency is frozen.

## Validation and Acceptance

Focused tests must prove exact parameter counts, c48 finite forward shapes, same-seed shared-backbone and Critic equality, the 36-job F4-only matrix, split boundaries/counts, Q4 no-read-before-allowlist behavior, common test panel/noise bank, exact aggregation, signed formatting, and tamper failure.

Run:

    python -m unittest tests.test_scripts.test_rq3_news_first_vol_f4_film_pure_capacity_3seed -v
    python -m unittest tests.test_scripts.test_rq3_news_first_vol_f4_film_pure_capacity_3seed_analysis -v
    bash .agents/skills/code-change-verification/scripts/run.sh

If this research repository lacks the Make targets assumed by the generic verification wrapper, record that limitation and run the repository-native Ruff, `py_compile`, focused/unit suites, binding checks, and `git diff --check` instead. The work is accepted only when the runner produces the expected artifacts and terminal QA verifies their hashes and row counts. If a TeX engine is installed, compile Chapter 3 and inspect the seven-column table visually.

## Idempotence and Recovery

Every lifecycle action must be safe with `--resume`: it either validates and reuses an identical completed artifact or refuses drift. Benchmark outputs use a separate root. Formal roots are never rewritten after terminal QA. If the resource benchmark fails at the proposed concurrency, create a new suffixed output root with the lower validated concurrency rather than mutate a prepared formal root. Q4 evaluation is retryable only when the checkpoint allowlist and all source hashes remain unchanged.

## Artifacts and Notes

Expected capacity totals are:

    c08: FiLM 544,738; Pure 220,034
    c12: FiLM 644,066; Pure 304,914
    c16: FiLM 770,370; Pure 416,770
    c24: FiLM 1,103,906; Pure 721,410
    c32: FiLM 1,556,902; Pure 1,145,510
    c48: FiLM 2,776,184; Pure 2,307,000

Existing c32 three-seed F4 regression anchors are FiLM MAE `0.0016690598968354838` and Pure MAE `0.0016693063748754267`. They are controls, not final table inputs.

## Interfaces and Dependencies

The runner exposes `run_news_first_vol_f4_film_pure_capacity_3seed(config_path, output_dir, *, action, job_id="", resume=False, reuse=False, worker_dry_run=False) -> Path`. The analysis module exposes a postprocess function callable by the runner and a pure aggregation function usable by unit tests. Output summary rows contain fold id, architecture, capacity id, current-capacity flag, Generator/Critic/total parameter counts, observed MAE, same-architecture c32 improvement, seed count, pair count, seed-pair row count, and provenance hashes.

Revision note (2026-09-24): initial ExecPlan created after the user replaced the earlier two-panel staged-refit proposal with a single standard-F4 FiLM/Pure capacity comparison.
