# RQ2 Continuation-Based Representation Validation Implementation

Implementation timestamp (UTC): `2026-07-24 15:05:46`

## Research Question

在相同 raw-vol surface、Stage-A parent、residual FiLM-WGAN、训练预算、
rolling folds 和 seeds 下，LP semantic embedding 是否优于 BoW log-count
与 ChatGPT sentiment scores？

Primary differences:

```text
Delta_LP_BoW       = BoW surface MAE - LP surface MAE
Delta_LP_sentiment = sentiment surface MAE - LP surface MAE

positive => LP has lower MAE
```

本实验是 `2023 rolling development evidence`，不能表述为 2024+ untouched
confirmation。

## Worktree And Source Experiment

Implementation branch/worktree:

```text
branch: wgan_rq2_pair_continuation
worktree:
/home/haobin_cui/research_files_space_2/wgan_option-rq2-pair-continuation
```

Continuation source:

```text
/home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation/
outputs/experiments/rq1_pair_text_raw_vol_continuation_20260723-143511
```

The source RQ1 validation status is `ok`. RQ2 imports:

```text
12 Stage-A no-text parent checkpoints
12 Stage-B continued no-text checkpoints and test results
12 Stage-B LP checkpoints and test results
4 rolling split manifests
4 LP PCA artifacts
```

Imports use hard links when source and target are on the same filesystem, with
copy fallback. Source path, target path, import method, size and SHA256 are
written to `inputs/rq1_import_manifest.csv`.

## Controlled Model Design

Each fold/seed uses this common Stage-A parent:

```text
pair_pca_no_text_residual
```

The comparison models are:

```text
continued_no_text = imported RQ1 Stage-B continuation control
lp                = imported RQ1 Stage-B LP residual branch
bow               = newly trained RQ2 Stage-B BoW residual branch
llm_sentiment     = newly trained RQ2 Stage-B sentiment residual branch
```

Only BoW and sentiment are newly trained:

```text
4 folds x 3 seeds x 2 representations = 24 new runs
```

Both new branches load the same fold/seed Stage-A generator checkpoint. They
do not initialize from the Stage-B continued no-text checkpoint. Both use a
fresh critic initialized under the same seed. The trainer writes
`metrics/initialization_audit.json` and checkpoint collection fails unless the
BoW/sentiment generator and critic initialization hashes match.

The imported RQ1 runs predate runtime initialization hashes. Their evidence is
therefore labeled separately: exact parent SHA, resolved architecture/seed,
fresh-critic code path and the imported RQ1 paired-stage audit are checked,
while only the new BoW/sentiment runs are marked as having directly observed
runtime state hashes. The combined record is written to
`checkpoint_selection/four_branch_initialization_audit.csv`.

Parent transform validation uses `dimension_only`: the Stage-A LP PCA artifact
and a new representation artifact may have different SHA256, but both must
produce exactly 128 input dimensions. Model shape and parent checkpoint SHA
remain strict.

## Pair-Level Representation Construction

The pair feature interface is keyed by `surface_pair_id`. Every artifact
records all source sample IDs and the deduplicated article IDs/source files.
Articles are deduplicated first by `ArticleID`, then by LP embedding SHA256.

### LP

The imported RQ1 path remains:

```text
mean(L2(unique article LP embeddings))
-> final L2 normalization
-> fold-train-only PCA-128
```

### BoW

For each fold:

```text
training corpus = unique articles from train surface pairs only
features        = unigram + bigram raw counts
vocabulary      = top 1024 deterministic frequency-ranked terms
pair vector     = L2(log1p(sum unique-article term counts))
transform       = fold-train-only PCA-128, no whitening
```

Validation and test text cannot influence vocabulary or PCA. The historical
full-sample BoW workbook/vocabulary is copied under
`inputs/audit/full_sample_bow/` and is explicitly forbidden as a training
input.

### ChatGPT Sentiment

Frozen source:

```text
model: gpt-5.4-mini
prompt: sun2026_zero_shot_chatgpt_v1
dimensions:
  macroeconomic_uncertainty
  institutional_action
  risk_off_intensity
```

For each pair, the three scores are averaged across unique articles. Mean and
standard deviation are fitted only on fold training pairs. The standardized
3-vector is zero-padded to 128 dimensions. The single usable `empty_text`
article retains a zero score and is recorded in `pair_feature_audit.csv`.

## Rolling Folds

```text
fold     train  validation  test
2023Q1   1621       425      521
2023Q2   2046       521      333
2023Q3   2567       333      365
2023Q4   2900       365      378
```

Seeds:

```text
42, 202, 404
```

Checkpoint selection:

```text
split  = validation
metric = lowest val_mae
epoch  > 10
```

## Statistical Outputs

The comparison builder matches rows by:

```text
fold, seed, surface_pair_id
```

It checks timestamps and persistence errors across all four models with
tolerance `1e-8`. Statistics first average paired differences across seeds,
then use trading-day cluster bootstrap:

```text
iterations = 10000
seed = 20260722
```

It also reports DM-style daily HAC tests with maximum lag 5, seed-level paired
t-tests, exact Wilcoxon tests, and fold/seed direction counts. The two primary
surface-MAE contrasts form one Holm family. Secondary metrics use separate
Holm families.

Full H2 support requires both primary mean differences and both CI lower bounds
to be positive, and both Holm-adjusted two-sided p-values below 0.05.

## Implemented Files

Core:

```text
src/bow/features.py
src/bow/__init__.py
src/film_wgan/config.py
src/film_wgan/data.py
src/film_wgan/text_transform.py
src/film_wgan/trainer.py
```

RQ2 workflow:

```text
configs/film_wgan/train_rq2_pair_textbase.yaml
scripts/rq2_pair/pair_features.py
scripts/rq2_pair/rq2_pair_experiment.py
scripts/rq2_pair/prepare_experiment.sh
scripts/rq2_pair/run_training_matrix.sh
scripts/rq2_pair/resume_training_matrix.sh
scripts/rq2_pair/start_training_matrix_background.sh
scripts/rq2_pair/monitor_training.sh
scripts/rq2_pair/collect_checkpoints.sh
scripts/rq2_pair/run_generate_test_matrix.sh
scripts/rq2_pair/build_comparison_archive.sh
scripts/rq2_pair/run_results_pipeline.sh
scripts/rq2_pair/start_results_pipeline_background.sh
scripts/rq2_pair/monitor_results_pipeline.sh
scripts/rq2_pair/package_experiment.sh
```

Tests:

```text
tests/test_scripts/test_rq2_pair_experiment.py
```

## Prepared Experiment

The implementation was validated against the real source artifacts and
prepared:

```text
outputs/experiments/
rq2_pair_representation_raw_vol_continuation_20260724-145528/
```

Preparation produced all four fold feature sets with the expected counts. A
real Q1 loader smoke check for both BoW and sentiment returned:

```text
train=1621, validation=425, test=521, embedding_dim=128
```

A CPU dry-run using the real Q1 Stage-A parent confirmed:

```text
BoW initial generator SHA       = sentiment initial generator SHA
BoW initial critic SHA          = sentiment initial critic SHA
surface shape                   = 16 x 16
embedding dimension             = 128
```

No RQ2 GPU training was started during implementation.

## Verification

Commands completed successfully:

```bash
conda run -n py312 python -m unittest \
  tests.test_text_features.test_bow_features \
  tests.test_scripts.test_rq1_pair_experiment \
  tests.test_scripts.test_rq2_pair_experiment \
  tests.test_standalone_wgan.test_film_wgan_module

# 48 tests passed; 1 skipped.

bash run_all_tests.sh

# 46 discovered tests passed.

conda run -n py312 python -m compileall -q \
  src/bow src/film_wgan scripts/rq2_pair \
  tests/test_scripts/test_rq2_pair_experiment.py

for file in scripts/rq2_pair/*.sh; do bash -n "${file}"; done
git diff --check
```

The RQ2 tests cover train-only vocabulary fitting, exact pair BoW pooling,
train-only sentiment statistics, 3-to-128 zero padding, hard-link/SHA import,
pair matching, difference direction, Holm correction, cluster bootstrap, HAC,
and an end-to-end synthetic four-model comparison archive.

## Commands

Prepare a new experiment:

```bash
conda activate py312
bash scripts/rq2_pair/prepare_experiment.sh
```

Start the 24-run sequential GPU matrix in the background:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq2_pair/start_training_matrix_background.sh
```

Monitor:

```bash
bash scripts/rq2_pair/monitor_training.sh
```

After all runs finish, start checkpoint collection, test generation and
comparison in the background:

```bash
bash scripts/rq2_pair/start_results_pipeline_background.sh
```

Monitor the result pipeline:

```bash
bash scripts/rq2_pair/monitor_results_pipeline.sh
```

Package:

```bash
bash scripts/rq2_pair/package_experiment.sh
```

## Expected Final Files

```text
comparisons/development_rq2_test_sample_metrics.csv
comparisons/development_rq2_pairwise_differences.csv
comparisons/development_rq2_cluster_bootstrap_ci.csv
comparisons/development_rq2_dm_hac_tests.csv
comparisons/development_rq2_seed_level_tests.csv

final_tables/development_rq2_primary_lp_vs_baselines.csv
final_tables/development_rq2_incremental_value_vs_no_text.csv
final_tables/development_rq2_vs_persistence.csv
final_tables/development_rq2_model_overall_metrics.csv
final_tables/development_rq2_result_summary.json
```
