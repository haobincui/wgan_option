# Validate the two selected FiLM learning rates across five fresh seeds

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. Maintain this document in accordance with `PLANS.md` in the repository root.

## Purpose / Big Picture

This experiment tests whether the apparent seed-42 sweet spot at FiLM projection learning rates `1e-5` and `2.5e-5` persists across independent random initialization. It trains the same c32 FiLM U-Net with matched pair-level LP text and the same NoLP Critic on four rolling 5-minute-alignment folds. The five validation seeds are `202`, `404`, `382624741`, `1607127774`, and `1662128673`; seed `42`, which motivated these two candidates, is excluded.

After completion, a reader can inspect one fixed primary comparison, `2.5e-5` versus `1e-5`, across five seeds, four folds, and paired CME sessions. Each arm is also compared with the pair-matched persistence forecast as one Holm-2 secondary family. The output is a robustness check after learning-rate selection, not a new learning-rate search and not a confirmatory holdout. Every result is labelled `retrospective_rolling_development_post_selection_seed_robustness`.

## Progress

- [x] (2026-08-31 10:38Z) Froze the two learning-rate candidates, five fresh seeds, four rolling folds, analysis estimands, discovery provenance, and output-root contract in a new configuration.
- [x] (2026-08-31 10:38Z) Added a strict five-seed analysis module and synthetic tests for 5,000 pair rows, grouped optimizer traces, hierarchical bootstrap, and immutable output bundles.
- [x] (2026-08-31 10:44Z) Added and passed eight orchestration contract tests covering the 40-job matrix, GPU/seed isolation, initialization pairing, grouped LR payload, and selection-provenance rejection.
- [ ] Complete and verify the thin five-seed orchestration profile against its lifecycle and regression tests.
- [ ] Run project-native validation and the mandatory verification wrapper.
- [ ] Pass the 40-cell one-epoch two-A30 resource benchmark without opening test data.
- [ ] Launch the detached formal supervisor, then freeze 40 checkpoints and 40 MC64 prediction cells.
- [ ] Produce the statistical report, terminal QA, and complete output SHA manifest.

## Surprises & Discoveries

- Observation: the seed-42 high-LR experiment is already terminal and binds the source files it used by SHA.
  Evidence: the frozen selection provenance records report SHA `2e72d868b7e3a86c48928fe5ce57c9d019cc0b82c9dade73cd45777e4398c8d2` and output-manifest SHA `80a1565db290eb459e2d31f61a6938fe5ab57a5871cb1947b304f621670a28d6`.

- Observation: the shared single-seed direct orchestrator embeds seed `42` in some run and prediction paths.
  Evidence: `scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py::_run_directory` and `_prediction_path` use the module-level `SEED`; the new wrapper must route through seed-aware paths without modifying this frozen source.

- Observation: the shared ten-seed core already provides seed-aware run directories, prediction paths, and deterministic MC-noise profiles.
  Evidence: `scripts/rq123/news_first_vol_film_nolp_10seed.py::_run_directory`, `_prediction_path`, and `_noise_bank_profile_sha256` all read `job["seed"]`.

## Decision Log

- Decision: use five fresh seeds and exclude discovery seed `42`.
  Rationale: this separates the evidence used to choose the two rates from the evidence used to assess seed robustness while keeping compute at 40 training jobs.
  Date/Author: 2026-08-31 / Codex.

- Decision: train only `film_lr_1e5` and `film_lr_2p5e5`; do not retrain Pure CNN or add another learning-rate candidate.
  Rationale: the user's question is whether these two already-selected FiLM rates are robust. Adding candidates would turn this run into another adaptive sweep. Persistence remains available pair by pair without a new model job.
  Date/Author: 2026-08-31 / Codex.

- Decision: freeze `2.5e-5` versus `1e-5` as the only primary contrast and compare both arms with persistence in one Holm-2 secondary family.
  Rationale: a predeclared contrast prevents the five-seed test panels from being used to choose the estimand after results are visible.
  Date/Author: 2026-08-31 / Codex.

- Decision: hold the Generator backbone LR at `5e-7`, text-encoder LR at `2.5e-6`, Critic LR at `5e-7`, and each FiLM floor at 10% of its arm's initial LR.
  Rationale: changing only the FiLM projection schedule preserves a clean causal interpretation of the optimization ablation.
  Date/Author: 2026-08-31 / Codex.

- Decision: require identical G/D initialization hashes within a seed and distinct hashes across all five seeds.
  Rationale: both learning-rate arms within a seed must be paired fairly, while the five seeds must represent genuinely different initial conditions.
  Date/Author: 2026-08-31 / Codex.

## Outcomes & Retrospective

The configuration, analysis implementation, living plan, and eight focused orchestration contract tests are complete; all eight focused tests pass. The thin orchestration lifecycle, prelaunch resource evidence, formal training, frozen predictions, and terminal inference remain pending. This section must be updated with benchmark telemetry, supervisor PID, completion counts, primary effect estimate, confidence interval, Holm-adjusted secondary results, and terminal QA evidence once they exist.

## Context and Orientation

The working directory is `/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone`. The new configuration is `configs/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.yaml`. The new CLI is `scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.py`, and its statistical implementation is `scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed_analysis.py`. The formal root is `outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_1e5_2p5e5_5seed_exact_ttm_rolling_v1`.

The Generator is `film_unet_mask_coords_v1`: a c32, 827,745-parameter FiLM U-Net with LP1024 transformed through a 256-to-128 text encoder and six FiLM projection sites. Its split optimizer contains 416,353 backbone parameters at `5e-7`, 295,808 text-encoder parameters at `2.5e-6`, and 115,584 FiLM-projection parameters at the arm-specific rate. The Critic is `lp_disabled_same_shape_v1`, has 729,157 parameters, ignores its LP input exactly, and trains at `5e-7`.

The four folds test 2023Q1 through 2023Q4. Their respective test pair/session counts are 110/34, 112/36, 135/33, and 143/45. Thus each seed-arm combination produces 500 pair rows, each arm produces 2,500 rows across five seeds, and the full experiment produces 5,000 rows. A 5-minute alignment window describes news-to-market matching tolerance; every target remains the volatility surface exactly five minutes after the current surface.

The prior seed-42 root is read-only. Its results selected the two rates and are bound into the new configuration by report and output-manifest SHA. The new wrapper must be additive and temporarily profile shared code with exception-safe restoration. It must never edit the old single-seed implementation or its completed artifacts.

## Plan of Work

Implement the wrapper as a thin profile over `scripts/rq3/news_first_vol_film_unet_direct_5arm_seed42.py` and the shared seed-aware functions in `scripts/rq123/news_first_vol_film_nolp_10seed.py`. Configuration validation must require the exact seed, fold, arm, LR, model, parameter-count, interpretation, bootstrap, and source-provenance contracts. It must verify both frozen discovery files against their configured SHA before a formal root can be created.

Generate exactly 40 specifications in the Cartesian product of five seeds, four folds, and two arms. Both arms in one seed-fold block run on the same physical GPU. Folds 1 and 3 use GPU 0; folds 2 and 4 use GPU 1, yielding 20 jobs per GPU. Run and prediction paths include `seed_<value>` so checkpoints, logs, and predictions cannot collide. Build one deterministic G/D initialization per seed, share its hashes across that seed's eight jobs, and reject any repeated initial hash across different seeds.

Every job trains independently from its seed-specific random initialization, with no parent, no continuation, no warmup, at most 240 epochs, minimum 30 epochs, and validation patience 20. It uses matched pair-level LP text, pair-collapsed data, the exact 16-by-16 TTM/strike grid, batch size 16, five Critic steps, MC16 validation, and `best_learned` checkpoint selection. Test loaders remain forbidden until every checkpoint and its allowlist are frozen.

Before formal prepare, run the same complete 40-job matrix for one epoch in an independent benchmark root. Prefer ten workers per GPU; if GPU peak is not below 20 GiB or host RAM is not below 85%, rerun in a fresh benchmark root with six workers per GPU and two waves. If the fallback also fails, do not create the formal root. Once formal prepare records concurrency and source hashes, do not mutate them.

After 40 checkpoints freeze, create four test panels and eight arm overlays, then evaluate every seed-fold-arm cell using MC64. Within each seed-fold, both arms must share an identical deterministic noise-bank profile. Across seeds, noise banks must remain seed-specific. Postprocessing validates all checkpoint, prediction, panel, overlay, origin, persistence, LR-trace, and source hashes before running 10,000 seed-then-fold-then-paired-CME-session bootstrap replicates.

## Concrete Steps

Run all commands from the repository root. First execute focused unit tests with the project Python environment:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m unittest \
      tests.test_scripts.test_rq3_news_first_vol_film_unet_direct_matched_lr_5seed \
      tests.test_scripts.test_rq3_news_first_vol_film_unet_direct_matched_lr_5seed_analysis

Run formatting, lint, compile, and whitespace checks on the new files:

    /home/haobin_cui/.local/bin/ruff format --check \
      scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.py \
      scripts/rq3/news_first_vol_film_unet_direct_matched_lr_5seed_analysis.py \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_lr_5seed.py \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_lr_5seed_analysis.py
    /home/haobin_cui/.local/bin/ruff check <the same four files>
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile <the same four files>
    git diff --check

Run the mandatory repository verification wrapper. If it stops solely because this repository has no `make format` target, record that infrastructure limitation alongside all passing project-native checks.

    bash .agents/skills/code-change-verification/scripts/run.sh

Run the full one-epoch benchmark without creating the formal root:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_lr_5seed \
      benchmark --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_1e5_2p5e5_5seed_exact_ttm_rolling_v1

After the benchmark passes, launch exactly one detached supervisor:

    nohup setsid env PYTHONPATH=src:. \
      /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_lr_5seed \
      run-pipeline --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_lr_5seed.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_1e5_2p5e5_5seed_exact_ttm_rolling_v1 \
      > outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_lr_1e5_2p5e5_5seed_exact_ttm_rolling_v1_control/pipeline.log \
      2>&1 < /dev/null &

## Validation and Acceptance

Configuration and prepare must yield exactly 40 unique jobs: five seeds by four folds by two arms. Each seed has eight jobs, each fold ten, each arm twenty, and the two GPUs twenty each. All jobs use matched LP and split optimization. Payloads preserve the job seed, keep backbone and Critic at `5e-7`, keep the text encoder at `2.5e-6`, and set FiLM to exactly `1e-5` or `2.5e-5` with a 10% floor. No payload may request a test loader, refit, parent, continuation, or branch recipe.

For each seed, its eight specifications and realized checkpoints must report the same initial Generator SHA and the same initial Critic SHA. Generator and Critic SHA sets must each have cardinality five across seeds. Run directories, checkpoints, logs, prediction files, manifests, and MC-noise profiles must contain or bind the correct seed, and no two cells may share an output path.

Formal evaluation is accepted only with 40 valid checkpoint allowlist entries, 40 prediction manifests, and exactly 5,000 pair rows. Both arms within each seed-fold must match on pair ID, session ID, effective origin, persistence error, and MC-noise SHA. The market lineage must also match across seeds, while each seed retains a distinct MC bank. Grouped LR traces must contain all four parameter groups at every realized epoch.

The analysis report must contain exactly one fixed primary row for `film_lr_2p5e5` versus `film_lr_1e5`, plus two persistence comparisons in one Holm-2 family. It must report seed and fold consistency without selecting or renaming a winner from test results. The report, terminal `qa.json`, and `output_hashes.csv` must all identify the interpretation as `retrospective_rolling_development_post_selection_seed_robustness`.

## Idempotence and Recovery

The benchmark, formal root, and completed seed-42 discovery root are independent. Re-running a fully completed formal root is strictly read-only. `--resume` may skip a training or prediction cell only after its job spec, source, configuration, inputs, output size, and output SHA all match the registry. Any drift in the discovery report or manifest, seed-specific initialization, overlay, checkpoint, prediction, or grouped LR trace fails closed.

A failed benchmark never authorizes formal-root creation. A partially trained formal root keeps completed cells and resumes only valid pending cells. Do not repair a SHA mismatch by editing a manifest or historical artifact; audit the cause and, when lineage has changed, create a new versioned output root.

## Artifacts and Notes

Expected formal artifacts include `registry/task_registry.json`, `registry/initial_state_fairness.json`, `registry/evaluation_checkpoint_allowlist.csv`, `inputs/pair_text_overlay_hashes.csv`, `evaluation/prediction_manifest.csv`, `analysis/rq12_pair_metrics.csv.gz`, `analysis/training_summary.csv`, primary and persistence comparison tables, Markdown and self-contained HTML reports, `qa.json`, and `output_hashes.csv`.

Prelaunch evidence is stored under the sibling control root. Record the final worker count, benchmark GPU/RAM peaks, supervisor PID, start time, task counts, and test-data-opened flag here after launch.

## Interfaces and Dependencies

The module `scripts.rq3.news_first_vol_film_unet_direct_matched_lr_5seed` exposes `benchmark`, `prepare`, `worker`, `launch`, `freeze_evaluation`, `predict`, `postprocess`, `qa`, `status`, and `run_pipeline`. It also exposes the pure helpers `load_config`, `validate_config`, `planned_specs`, `validate_gpu_balance`, `_run_directory`, `_prediction_path`, and `_training_payload` for contract tests. Its exception-safe `multiseed_profile()` temporarily adapts shared orchestration code and restores all module globals after exit.

The analysis module exposes `analyze_experiment(output_root, config)` and writes deterministic, SHA-bound analysis artifacts. No new third-party dependency is introduced; training and analysis use the existing PyTorch, pandas, NumPy, PyYAML, and project py312 environment.
