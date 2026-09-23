# Measure FiLM text effects after a freshly trained Pure-CNN backbone

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. Maintain this document in accordance with `PLANS.md` in the repository root.

## Purpose / Big Picture

This experiment answers a narrower question than the completed direct-training runs: after a strong text-free convolutional U-Net has learned the quantitative surface transition, does correctly matched text add predictive information? It first trains one fresh `cnn_unet_mask_coords_v1` parent for every seed and rolling fold. It then starts six equal-budget continuations from that selected parent: a Pure-CNN no-text continuation, a structurally matched FiLM zero-text control, and FiLM models using matched LP, shuffled LP, BoW, or sentiment.

The experiment uses only the 5-minute news-alignment dataset, ten fixed seeds, and four rolling folds. It contains 40 backbone parents and 240 continuations, for 280 training jobs. Once all selected checkpoints are frozen, it creates 280 standard MC64 prediction cells and 80 input-intervention cells. The primary Holm-2 family compares matched LP with the FiLM zero-text control and with the independently trained shuffled-LP placebo. Results are retrospective rolling-development evidence, not a confirmatory holdout or a causal news-effect estimate.

All public artifacts live under one formal root:

    outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1

Within that root, `stages/backbones/` holds the 40 Pure-CNN parents and `stages/continuations/` holds the 240 continuation runs. This internal separation preserves lineage without exposing multiple experimental roots.

## Progress

- [x] (2026-09-01 15:06Z) Audited the completed direct Pure-CNN, direct FiLM-text, legacy parent/continuation, and U-Net parent/continuation experiments.
- [x] (2026-09-01 15:06Z) Froze the five-minute panel, ten seeds, four folds, 40-plus-240 training matrix, split learning rates, snapshot schedule, prediction counts, primary contrasts, and source-evidence hashes in a new configuration.
- [ ] Implement and test the Pure-CNN-to-FiLM graft contract and the single-root orchestration profile.
- [ ] Pass recovery, one-epoch 280-job resource, data-lineage, and prediction-routing gates without opening test data.
- [ ] Launch the detached formal supervisor and freeze all 280 selected checkpoints.
- [ ] Produce 280 standard and 80 intervention prediction cells, 10,000-draw inference, reports, terminal QA, and the complete output SHA manifest.

## Surprises & Discoveries

- Observation: the earlier U-Net parent experiment did not train a literal Pure-CNN parent. It trained the full FiLM U-Net with a zero 1024-vector, so text-encoder biases and FiLM projections could learn sample-invariant offsets.
  Evidence: `film_unet_mask_coords_v1` registers 827,745 Generator parameters, while `cnn_unet_mask_coords_v1` registers 416,353 and has no text encoder or FiLM projection modules.

- Observation: the earlier U-Net parent budget was commonly exhausted. Sixty-nine of 80 parent checkpoints selected epoch 30, and 45 of 80 no-text continuations selected the 240-epoch ceiling.
  Evidence: the frozen `best_learned_checkpoint.json` files and `report/branch_epoch_summary.json` under `rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240_exact_ttm_rolling_v1`.

- Observation: the prior pretrained text branches used one `5e-7` Generator rate, whereas the best recent direct configuration used `5e-7` for the convolutional backbone, `2.5e-6` for the text encoder, and `2.5e-5` for the six FiLM projections.
  Evidence: the terminal direct FiLM config and grouped-LR traces bound under `source_evidence.film_text_direct` in the new configuration.

- Observation: the completed direct roots are valid discovery evidence but cannot be fair parents for the new matrix.
  Evidence: they were independently trained arms, do not provide one selected Pure-CNN parent shared by all six continuations, and are terminal SHA-bound roots.

## Decision Log

- Decision: train 40 new literal Pure-CNN parents rather than feeding zero text through a FiLM parent.
  Rationale: a literal Pure-CNN isolates the quantitative backbone and prevents constant FiLM offsets from entering Stage A.
  Date/Author: 2026-09-01 / User and Codex.

- Decision: graft only the shared U-Net convolution tensors and the complete NoLP Critic tensors into each FiLM continuation.
  Rationale: the two Generator modes share `surface_encoder`, `bottleneck_conv`, `decoder_convs`, and `residual_head`; text and FiLM tensors have no Pure-CNN source and therefore require deterministic new initialization.
  Date/Author: 2026-09-01 / User and Codex.

- Decision: reset all Stage-B optimizer and scheduler states, including the Pure-CNN continuation, while preserving identical selected parent weights within each seed-fold block.
  Rationale: FiLM adds new optimizer groups that have no parent moments. A common fresh optimizer makes all six continuation budgets comparable and avoids an optimizer-history advantage for the Pure-CNN control.
  Date/Author: 2026-09-01 / Codex.

- Decision: allow every continuation to select its own validation-best checkpoint from at most 240 epochs.
  Rationale: the earlier common-epoch rule answered an equal-duration question but forced text arms to inherit a no-text stopping decision. Independent validation selection tests each representation under the same maximum budget without reading test errors.
  Date/Author: 2026-09-01 / User and Codex.

- Decision: use `5e-7`, `2.5e-6`, and `2.5e-5` for the FiLM backbone, text encoder, and FiLM projections, respectively; keep the NoLP Critic at `5e-7` and all floors at 10% of their initial rates.
  Rationale: this carries the strongest direct-training FiLM schedule into the pretrained-backbone experiment while leaving the quantitative path unchanged.
  Date/Author: 2026-09-01 / User and Codex.

- Decision: freeze `LP matched versus FiLM zero-text` and `LP matched versus shuffled LP` as one Holm-2 primary family.
  Rationale: the first contrast isolates sample-varying text in an equal-parameter architecture; the second tests whether correct pairing matters. Pure-CNN and representation comparisons remain secondary or diagnostic.
  Date/Author: 2026-09-01 / User and Codex.

- Decision: produce two input interventions for every matched-LP checkpoint: zero text and an independently shuffled LP input.
  Rationale: 80 extra predictions measure whether a fixed trained checkpoint reacts to text at inference, separately from differences caused by training two models.
  Date/Author: 2026-09-01 / Codex.

## Outcomes & Retrospective

The configuration contract is frozen. Runtime implementation, benchmarks, formal training, prediction, and inference remain pending. Update this section with the realized parent and continuation epoch distributions, snapshot trajectories, graft equivalence evidence, resource peaks, primary effect estimates and confidence intervals, intervention sensitivity, supervisor timing, and terminal QA after execution.

## Context and Orientation

The repository root is `/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone`. The formal configuration is `configs/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.yaml`. The public CLI is `python -m scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed`. The implementation should remain a thin profile over the audited ten-seed direct orchestration and model stack rather than copying either pipeline.

The surface input is a 16-by-16 exact-TTM grid. Its maturity days are `[1,2,3,6,7,8,9,10,14,15,17,21,26,30,35,38]`, and its moneyness values span 0.970 through 1.030. Four Generator input channels contain the current supported IV surface, the current support mask, normalized moneyness, and normalized log maturity. `raw_joint` support is used in losses and metrics. The forecast target is always the surface five minutes after the current surface; “5m alignment” describes only the permitted news-to-market timestamp distance.

The Pure-CNN Generator has 416,353 parameters. The FiLM Generator adds a 295,808-parameter LP1024-to-256-to-128 text encoder and 115,584 parameters across six zero-initialized FiLM projections, for 827,745 Generator parameters. Both use the 729,157-parameter `lp_disabled_same_shape_v1` Critic, whose encoded text feature is exactly zero and whose output is invariant to its text argument.

For each fold and seed, Stage A starts a new Pure-CNN Generator and Critic from the canonical seeded initialization and trains at most 240 epochs. The validation-best learned state becomes the only parent for that block. Stage B creates six independent continuations from the selected parent weights. `pure_cnn_continue_no_text` keeps the Pure-CNN architecture. The five FiLM arms receive an identical deterministic graft: every shared convolution tensor equals the parent exactly, the Critic equals the parent exactly, the text encoder has the same initial state across FiLM arms, and all FiLM projections start at zero. Before training, `G_film(current, zero_text, noise)` must match the Pure-CNN parent to maximum absolute error at most `1e-7`.

The completed direct roots listed in `source_evidence` are read-only discovery provenance. Formal preparation verifies every configured SHA but never loads one of their checkpoints as a parent. Any drift fails before the new formal root is created.

## Plan of Work

Implement a versioned graft utility with state kind `pure_cnn_to_film_graft_state_v1` and manifest kind `pure_cnn_to_film_graft_manifest_v1`. It must copy exactly the Generator prefixes `surface_encoder.`, `bottleneck_conv.`, `decoder_convs.`, and `residual_head.`. The only new target keys may begin with `text_encoder.`, `encoder_film_layers.`, `bottleneck_film_layer.`, or `decoder_film_layers.`. It must reject missing, extra, shape-mismatched, nonfinite, or renamed tensors; record the parent and graft artifact paths, sizes, and SHA-256 values; and verify the zero-text forward identity using a fixed finite fixture.

Build one configuration-driven orchestrator that exposes two internal stages beneath the same root. Parent runs write to `stages/backbones/fold_<fold>/seed_<seed>/pure_cnn_parent/`. Continuation runs write to `stages/continuations/fold_<fold>/seed_<seed>/<arm>/`. A task registry must contain 40 unique backbone jobs and 240 unique continuation jobs. It must also bind the selected parent and graft manifest used by every continuation. No path, checkpoint, optimizer state, RNG stream, overlay, or validation snapshot may collide across cells.

Stage A trains at most 240 epochs, minimum 30, with patience 20 and `val_hybrid_score` selection. Save MC16 validation trajectory artifacts at epochs 0, 1, 5, 10, 20, 30, and the selected best epoch. Epoch 0 is the state before any update. If the selected best duplicates a numbered snapshot, record one physical artifact with both roles. These snapshots are validation diagnostics only and may never be chosen using test results.

After all 40 parents pass full-state and SHA validation, freeze a parent allowlist and materialize six Stage-B job specifications per parent. Every Stage-B optimizer and scheduler begins fresh. The Pure-CNN continuation uses one Generator group at `5e-7`. Each FiLM arm uses disjoint backbone, text, and FiLM groups at `5e-7`, `2.5e-6`, and `2.5e-5`; the Critic uses `5e-7`. All groups use ReduceLROnPlateau with factor 0.5, patience 3, and a floor equal to 10% of the initial group rate. All six arms use the same batch order and Gaussian training-noise bank within a seed-fold block and independently select validation-best checkpoints under the same 240/minimum-30/patience-20 rule.

Do not materialize a test loader until all 280 selected checkpoints, parent/graft manifests, validation-snapshot manifests, and the evaluation allowlist are frozen. Then generate 280 standard MC64 predictions: the parent plus all six continuations in every seed-fold block. Freeze one deterministic noise bank per seed-fold and use it for every standard and intervention prediction in that block. Standard predictions must share pair IDs, session IDs, origins, targets, persistence errors, and support masks.

For each of the 40 `film_lp_matched` checkpoints, also predict with a zero vector and with an independently shuffled pair-level LP vector, producing 80 intervention cells. The intervention derangement is frozen before prediction, stays within each test split, has no fixed point, and differs from the training `film_lp_shuffle` mapping. It may not use targets, errors, or text similarity.

Analyze the 5-minute standard panel with 10,000 paired `seed -> fold -> CME-session` bootstrap draws. The estimand is the equal-seed-fold mean of `log(MAE_focal / MAE_reference)`; negative values favor the focal model. Apply Holm correction only to the two frozen primary comparisons. Statistical support additionally requires a negative point estimate, a 95% confidence-interval upper bound below zero, Holm-adjusted `p < 0.05`, and non-worse direction in at least seven seeds and three folds. Report the other trained-model and persistence comparisons as secondary descriptive evidence. Report the two fixed-checkpoint interventions as diagnostic input sensitivity, not as a substitute for the trained-arm primary tests.

Postprocessing produces a single Markdown report, a self-contained HTML report, pair/seed/fold summaries, all 10,000 bootstrap draws, validation trajectory tables, graft diagnostics, resource telemetry, terminal `qa.json`, and a complete output SHA manifest. Every output must carry the interpretation `retrospective_rolling_development_text_effectiveness`.

## Concrete Steps

Run commands from the repository root with the project Python environment. First run focused graft, orchestration, prediction, and statistical tests:

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python -m unittest \
      tests.test_models.test_pure_cnn_to_film_graft \
      tests.test_scripts.test_rq3_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed

Run formatting, lint, compile, and whitespace checks on the new runtime and test files:

    /home/haobin_cui/.local/bin/ruff format --check <new Python files>
    /home/haobin_cui/.local/bin/ruff check <new Python files>
    /home/haobin_cui/.conda/envs/py312/bin/python -m py_compile <new Python files>
    git diff --check

Run the mandatory verification wrapper. If it fails solely because this repository has no `make format` target, record that infrastructure limitation together with the passing native checks.

    bash .agents/skills/code-change-verification/scripts/run.sh

Before formal prepare, run the full 40-parent-plus-240-continuation matrix for one epoch in an independent benchmark root. The benchmark may use one-epoch parents only to create temporary graft inputs; none of its artifacts may enter the formal root.

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed \
      benchmark --resume \
      --config configs/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.yaml

If the benchmark passes, prepare the single formal root. Preparation creates its `control/`, `stages/backbones/`, and `stages/continuations/` directories and freezes the concurrency decision.

    PYTHONPATH=src:. /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed \
      prepare --resume \
      --config configs/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1

Launch exactly one detached supervisor after prepare succeeds:

    nohup setsid env PYTHONPATH=src:. \
      /home/haobin_cui/.conda/envs/py312/bin/python \
      -m scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed \
      run-pipeline --resume \
      --config configs/rq3/news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed.yaml \
      --output-dir outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1 \
      > outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1_control/pipeline.log \
      2>&1 < /dev/null &

The public actions are `benchmark`, `prepare`, `dry-run`, `worker`, `launch-backbones`, `freeze-backbones`, `launch-continuations`, `freeze-evaluation`, `predict`, `predict-interventions`, `analyze`, `report`, `qa`, `status`, and `run-pipeline`.

## Validation and Acceptance

Configuration validation must recover the exact 5-minute fold counts: train/validation/test pairs of 382/144/110, 526/110/112, 636/112/135, and 748/135/143. It must reject any seed, fold, grid, support, loss, source-evidence SHA, arm, LR, model count, or prediction-count drift. Formal preparation must create exactly 280 unique training jobs, balanced as 20 parent blocks and 120 continuation jobs per GPU.

Every parent must have a selected learned epoch from 1 through 240 and a complete validation trajectory manifest. Every continuation must point to exactly one allowlisted parent. Within a seed-fold block, all five initial FiLM states must have the same complete Generator SHA; every shared Generator tensor and Critic tensor must equal the parent; only the declared text/FiLM keys may be new. Different seed-fold blocks must not accidentally share parent or new-module initialization hashes.

The zero-text graft test must pass at `max_abs <= 1e-7` in both evaluation and training mode with an explicitly supplied common Gaussian noise tensor. NoLP Critic outputs must be identical for arbitrary LP inputs and its text gradients must be zero. The first matched-text update must produce finite nonzero text-encoder and FiLM-projection gradients, while every optimizer parameter appears in exactly one group at the configured rate.

Validation snapshots must exist at 0, 1, 5, 10, 20, 30, and selected-best for all 280 runs. Snapshot rows must contain validation pair/session counts, MC profile SHA, MAE, persistence MAE, constraint diagnostics, grouped LR, and checkpoint SHA. No snapshot or validation selector may contain a test pair ID or test metric.

Evaluation is accepted only with 280 standard cells, 80 intervention cells, 35,000 standard pair rows, and 10,000 intervention rows. All comparison cells must align exactly on market lineage and use the same MC64 noise profile within seed-fold. The intervention manifest must prove valid no-fixed-point split-local derangements and must differ from the training shuffle manifest.

The final report must contain exactly two primary rows in one Holm-2 family, each based on 10,000 nested paired bootstrap draws. It must show point estimates, 95% intervals, raw and adjusted p-values, seed/fold consistency, and the full support gate. Secondary model, persistence, trajectory, and intervention tables must be explicitly labelled non-primary. Terminal QA passes only after all source, input, parent, graft, checkpoint, prediction, bootstrap, report, and output-manifest hashes revalidate.

## Idempotence and Recovery

The benchmark and formal root are independent. A benchmark failure never authorizes formal prepare. Prefer ten workers per GPU; if either GPU reaches 20 GiB or host RAM reaches 85%, rerun a fresh benchmark at six workers per GPU and two waves. A second failure stops the experiment. Formal prepare freezes the successful concurrency profile and no later stage may change it.

The supervisor uses one exclusive lock, PID file, stage journal, and fail-closed registry. `--resume` skips a cell only when its source hashes, job specification, parent/graft inputs, output sizes, and output hashes all match. A partial or corrupted attempt is retained for audit and retrained in a new attempt directory. Re-running a terminal root is strictly read-only. Never repair a mismatch by editing a frozen manifest or a completed direct-evidence root.

## Artifacts and Notes

Expected artifacts include the 280-job registry, 40 selected-parent manifests, 40 graft manifests, 280 validation-trajectory manifests, a 280-entry standard checkpoint allowlist, a 280-row standard prediction manifest, an 80-row intervention manifest, 45,000 pair-metric rows, raw bootstrap draws, grouped-LR and resource summaries, Markdown and HTML reports, terminal QA, and a complete output SHA manifest.

The completed direct roots are evidence for why this experiment exists: matched LP had the best point estimate in the direct ten-seed comparison, but its independent text increment over Pure-CNN remained small and statistically unresolved. Their configured SHAs must be rechecked before formal preparation, but their model weights, optimizer states, predictions, and test-selected observations are never imported.

## Interfaces and Dependencies

The module `scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed` owns the public CLI and the single-root lifecycle. Its pure planning helpers must expose deterministic job IDs, stage paths, GPU assignments, graft bindings, grouped LR payloads, snapshot paths, standard prediction paths, intervention paths, and bootstrap contrast definitions for unit testing.

The graft utility exposes a pure construction/validation operation that accepts an allowlisted Pure-CNN parent checkpoint plus the target FiLM model contract and returns a `pure_cnn_to_film_graft_state_v1` payload and `pure_cnn_to_film_graft_manifest_v1`. No new third-party dependency is required; the implementation uses the existing PyTorch, pandas, NumPy, PyYAML, and project py312 environment.
