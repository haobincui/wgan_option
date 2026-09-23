# Validate higher FiLM learning rates without a parent stage

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. Maintain this document in accordance with `PLANS.md` in the repository root.

## Purpose / Big Picture

This experiment tests whether the six FiLM projection layers in the c32 FiLM U-Net learn a more useful matched-text signal when their optimizer learning rate is raised above the earlier maximum of `5e-6`. It trains directly from the common seed-42 random initialization and never loads a parent or continuation state. The five rates are `5e-6`, `1e-5`, `2.5e-5`, `5e-5`, and `1e-4`; the first is a bridge to the earlier sweep and the last is a stability stress endpoint.

After completion, a reader can compare every FiLM arm with the frozen Pure CNN + No-text reference on exactly the same four rolling test panels and MC64 noise banks. Because the architecture and 2023 test periods have already been examined and only one seed is used, every result is labelled `retrospective_rolling_development_single_seed_descriptive`. Test results must not be used to claim confirmatory evidence or an unbiased learning-rate selection.

## Progress

- [x] (2026-08-31 02:10Z) Audited the completed low-LR experiment, prediction routing, frozen source hashes, resource contract, and reusable orchestration components.
- [x] (2026-08-31 02:24Z) Added the independent high-LR configuration and thin profile wrapper without modifying any old hashed source/config file.
- [x] (2026-08-31 02:24Z) Added an independent four-fold, five-epoch `1e-4` stability canary ahead of formal prepare.
- [x] (2026-08-31 02:59Z) Added and verified the high-LR analysis adapter and synthetic analysis tests.
- [x] (2026-08-31 02:59Z) Passed targeted formatting, Ruff, `py_compile`, 32 focused/regression unit tests, and `git diff --check`; ran the mandatory verification script, which stopped only because this repository has no `make format` target.
- [x] (2026-08-31 03:08Z) Passed the full 20-cell one-epoch resource benchmark and the four-cell five-epoch `1e-4` stress canary after correcting the canary worker-count gate.
- [x] (2026-08-31 03:12Z) Created the formal root and launched the detached supervisor as PID `3872572`; all 20 workers entered training with a balanced 10/10 GPU assignment.
- [ ] Freeze predictions, analyze all five high rates against Pure CNN, and complete terminal QA.

## Surprises & Discoveries

- Observation: the failed low-LR root froze live-source hashes for both the shared direct orchestrator and the low-LR wrapper. Editing either file would make that historical root fail its own source-integrity check.
  Evidence: its `code_hashes.csv` records the direct orchestrator SHA `ab77a97525f159aa8ebe2e6bb3437ca9ddc495f5efe72b525330ccb6ff0251f7` and low-LR wrapper SHA `1288d2e0c265577cf4c25b654b715690dab666328c3b193370c740c9f8487d4e`.

- Observation: the shared prediction core historically resolves overlay mode from a closed arm-name mapping even though each new job already carries `pair_text_overlay_mode=lp_mean_l2`.
  Evidence: `scripts/rq123/news_first_vol_film_nolp_10seed.py::_panel_with_overlay` calls its module-local `_overlay_mode(arm)`, which does not know arbitrary FiLM-LR arm names.

- Observation: validation improved monotonically but only slightly over the earlier `2.5e-7` through `5e-6` range.
  Evidence: equal-fold mean best `val_hybrid_score` changed from `0.00204040349` to `0.00203897898`, approximately `0.0698%`; this motivates a higher-rate probe but does not establish test improvement.

- Observation: the py312 environment does not install Ruff as a Python module, while `/home/haobin_cui/.local/bin/ruff` is available and passes the new files.
  Evidence: `python -m ruff` returned `No module named ruff`; the standalone Ruff binary reported all four new Python/test files formatted and `All checks passed!`.

- Observation: the mandatory repository verification wrapper cannot advance beyond its first step in this repository.
  Evidence: `bash .agents/skills/code-change-verification/scripts/run.sh` ran `make format`, which returned `No rule to make target 'format'`; project-native targeted validation passed independently.

- Observation: the first stress-canary prepare attempt requested two workers per GPU, but the frozen runtime contract permits only ten or six.
  Evidence: formal-root creation never began; the attempt failed closed with `Workers/GPU must be one of [6, 10]`. The incomplete evidence was archived, the canary now uses the configured six-worker fallback, and all source-bound gates were rerun from scratch.

## Decision Log

- Decision: create only new high-LR files and use nested, exception-safe context managers to patch the low-LR wrapper, direct experiment profile, and shared prediction overlay resolver.
  Rationale: this keeps all historical source hashes valid while reusing the audited lifecycle and restoring global state after every call.
  Date/Author: 2026-08-31 / Codex.

- Decision: test `5e-6`, `1e-5`, `2.5e-5`, `5e-5`, and `1e-4`.
  Rationale: five arms retain the existing Holm-5 and five-choose-two Holm-10 analysis shape. `5e-6` is an overlap bridge and `1e-4` is explicitly a stress endpoint, not a preferred setting.
  Date/Author: 2026-08-31 / Codex.

- Decision: keep backbone and Critic LR at `5e-7`, text-encoder LR at `2.5e-6`, no warmup, and each FiLM scheduler floor at 10% of its arm's initial LR.
  Rationale: changing only the FiLM LR schedule preserves interpretability relative to the earlier direct matched-text experiment.
  Date/Author: 2026-08-31 / Codex.

- Decision: require both the existing full-matrix one-epoch resource benchmark and a separate four-fold `1e-4` five-epoch stability canary before formal prepare.
  Rationale: one epoch is enough for capacity telemetry but is too weak to assess immediate high-LR divergence. The extra canary checks finite metrics, parameter updates, LR identity, GPU memory, and host RAM at the most aggressive rate.
  Date/Author: 2026-08-31 / Codex.

- Decision: prepare the stress root with the frozen fallback setting of six workers per GPU, while executing only the four selected `1e-4` jobs.
  Rationale: six is an allowed registry value; the selected jobs still run as two balanced fold waves and no unselected cell is trained.
  Date/Author: 2026-08-31 / Codex.

## Outcomes & Retrospective

The additive configuration, thin orchestration profile, high-LR analysis module, living plan, and synthetic contract tests are complete. Seventy focused and regression tests pass, including twelve high-LR tests. Ruff format/check, `py_compile`, and `git diff --check` pass. All previously frozen low-LR source/config SHA values remain byte-for-byte unchanged. The source-consistent 20-cell benchmark passed at 10 workers per GPU with a `3.804 GiB/GPU` peak and `14.21%` host-RAM peak. The four-cell, five-epoch `1e-4` canary passed with a `0.396 GiB/GPU` peak and `8.29%` host-RAM peak. The formal supervisor is running as PID `3872572`; 20/20 workers entered epoch training, both A30s reached 99% utilization at approximately `3.51 GiB` each, and the test-data-opened flag remains false. Terminal results remain pending.

## Context and Orientation

The working directory is `/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone`. The earlier low-rate implementation is `scripts/rq3/news_first_vol_film_unet_direct_matched_lr_seed42.py`; it must remain byte-for-byte unchanged because a completed experiment root binds its SHA. The new configuration is `configs/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.yaml`, and the new CLI is `scripts/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.py`.

The Generator is `film_unet_mask_coords_v1` with 827,745 parameters. Its optimizer has three disjoint parameter groups: 416,353 backbone parameters at LR `5e-7`; 295,808 text-encoder parameters at LR `2.5e-6`; and 115,584 FiLM-projection parameters at the arm-specific LR. The NoLP Critic has 729,157 parameters and LR `5e-7`. All arms use the same matched pair-level LP1024 vectors and the same seed-42 initial G/D tensor states.

The four rolling folds use 5-minute news/market alignment. Each predicts the future five-minute volatility surface. Their test pair/session counts are 110/34, 112/36, 135/33, and 143/45, so each arm produces 500 test-pair rows and all five arms produce 2,500 rows. No test loader may be created until all 20 best-learned checkpoints and their allowlist are frozen.

The formal output root is `outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_seed42_exact_ttm_rolling_v1`. The resource benchmark and stress-canary roots are siblings, not children of the formal root. Formal prepare records the SHA-bound stress result in the task registry.

## Plan of Work

The thin high-LR wrapper temporarily replaces the experiment constants on the low-LR module. It then enters the low module's own profile, which installs those constants and functions on the shared direct orchestrator. Finally it temporarily replaces the shared RQ1–RQ3 core `_overlay_mode` with a resolver that accepts exactly the five high-LR arms and returns `lp_mean_l2`. `finally` blocks restore core, direct, and low module state even if validation or execution fails.

The full resource benchmark creates 20 one-epoch jobs, ten per A30. The additional stress canary prepares a separate five-epoch registry but executes only the four `film_lr_1e4` fold jobs, two per GPU. It fails closed for NaN/Inf, an incorrect initial FiLM LR, missing epochs, unchanged model parameters, GPU memory at or above 20 GiB, or host RAM at or above 85%.

Only after both gates pass does formal prepare create the 20-job root. Every formal job starts independently from the same seed-42 initialization, trains for at most 240 epochs with validation early stopping, and freezes `best_learned`. Prediction uses MC64 and matched-LP overlays. Postprocessing performs 10,000 fold-then-session paired bootstrap replicates, Holm-5 comparisons against frozen Pure CNN, and Holm-10 comparisons among all five FiLM rates.

## Concrete Steps

Run commands from the repository root with the py312 interpreter.

    /home/haobin_cui/.conda/envs/py312/bin/python -m pytest \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_high_lr_seed42.py \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis.py

Run static checks on new Python files and confirm the patch has no whitespace errors.

    /home/haobin_cui/.conda/envs/py312/bin/python -m ruff format --check \
      scripts/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.py \
      scripts/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis.py \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_high_lr_seed42.py \
      tests/test_scripts/test_rq3_news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis.py
    /home/haobin_cui/.conda/envs/py312/bin/python -m ruff check <the same files>
    git diff --check

Run mandatory repository verification. If it fails solely because this repository has no `make format` target, record that infrastructure error and retain the project-native evidence above.

    bash .agents/skills/code-change-verification/scripts/run.sh

Run the two prelaunch gates. The `benchmark` action runs both the one-epoch 20-cell benchmark and the five-epoch stress canary.

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42 \
      benchmark --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_seed42_exact_ttm_rolling_v1

After both gates pass, launch exactly one detached supervisor.

    nohup setsid env PYTHONPATH=src:. \
      /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42 \
      run-pipeline --resume \
      --config configs/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_seed42_exact_ttm_rolling_v1 \
      > outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_seed42_exact_ttm_rolling_v1_control/pipeline.log \
      2>&1 < /dev/null &

## Validation and Acceptance

Configuration validation must produce exactly five ordered rates and exactly 20 unique jobs. GPU assignments must be balanced 10/10. All jobs must have empty parent/continuation/recipe inputs, matched-LP overlay mode, the same initial G/D state SHA, backbone LR `5e-7`, text LR `2.5e-6`, and FiLM floor equal to 10% of initial FiLM LR.

An exception raised inside the high-LR profile must leave every patched low/direct/core global identical by object identity to its pre-context value. Inside the context, both the direct orchestrator and shared prediction core must resolve every high-LR arm to `lp_mean_l2`, including `film_lr_1e4`.

The full resource benchmark must complete 20/20 jobs without NaN, Inf, OOM, or unchanged G/D parameters. The stress canary must complete exactly four `1e-4` jobs with epoch trace `0..5`, finite numeric metrics, initial `g_lr_film_projection=1e-4`, GPU memory below 20 GiB, and RAM below 85%. Formal prepare must refuse to create the root if the stress result is absent or SHA-invalid.

Terminal acceptance requires 20/20 formal training jobs, 20/20 prediction cells, 2,500 FiLM pair rows, 500 frozen Pure CNN pair rows, strict fold/pair/session/origin/persistence/MC-noise equality, complete grouped-LR traces, Markdown and self-contained HTML reports, terminal QA, and a full output SHA manifest.

## Idempotence and Recovery

All roots are independent. Re-running a completed terminal formal root is read-only. `--resume` skips a benchmark, canary job, formal job, or prediction only after registry/spec/artifact size and SHA validation. A partial canary root retains all 20 planned cells but only the four stress jobs are eligible to run; the other 16 remain pending and are never mistaken for formal output.

Do not delete or rewrite the earlier low-LR or Pure CNN roots. If a source, config, dataset, overlay, checkpoint, canary-result, or registry hash changes, resume fails closed. Recovery consists of auditing the changed artifact and starting a new versioned root, not mutating a frozen manifest.

## Artifacts and Notes

Expected formal artifacts include `registry/task_registry.json`, `registry/evaluation_checkpoint_allowlist.csv`, `registry/initial_state_fairness.json`, `analysis/rq12_pair_metrics.csv.gz`, grouped LR diagnostics, bootstrap/Holm tables, Markdown/HTML reports, `qa.json`, and `output_hashes.csv`. Prelaunch evidence is stored under the sibling control directory as `benchmark_result.json` and `stress_canary_result.json`.

Implementation verification before launch:

    70 focused high-LR/low-LR/Pure-CNN/optimizer regressions: OK
    Ruff format/check: OK
    py_compile: OK
    git diff --check: OK
    mandatory wrapper: infrastructure stop at missing `make format`

## Interfaces and Dependencies

The module `scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42` exposes `benchmark`, `stress_canary`, `prepare`, `worker`, `launch`, `freeze_evaluation`, `predict`, `postprocess`, `qa`, `status`, and `run_pipeline`. It imports `scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis`, which must expose an `analyze_experiment(output_root, config)` function compatible with the low-LR analysis manifest schema.

No new third-party dependency is introduced. Training continues to use PyTorch, pandas, NumPy, PyYAML, and the project's existing py312 environment.
