# RQ1 No-Text Continuation Control Implementation

Implementation timestamp (UTC): `2026-07-23 14:24:44`

## Worktree Isolation

This change is developed in a separate Git worktree so that the active
72-run RQ1 training matrix continues to read its original source tree.

```text
source worktree:
  /home/haobin_cui/research_files_space_2/wgan_option

development worktree:
  /home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation

development branch:
  wgan_rq1_no_text_continuation

base commit:
  75b9176ecb3644405296bb50d9e9ae9589536ce7
```

No file in the active training worktree is modified by this implementation.

## Research Motivation

The existing primary comparison is:

```text
Stage-A no-text parent error - Stage-B text residual error
```

This comparison mixes the text effect with a second optimization stage,
including a fresh critic, five frozen-backbone epochs, and subsequent
low-learning-rate backbone fine-tuning.

The revised experiment adds a Stage-B no-text continuation control:

```text
Stage A:
  pair_pca_no_text_residual

Stage B, same parent checkpoint:
  pair_pca_no_text_continued
  pair_pca_text_residual_pretrained
  pair_pca_shuffled_residual_pretrained
```

The primary difference becomes:

```text
difference =
  MAE(pair_pca_no_text_continued)
  - MAE(pair_pca_text_residual_pretrained)

difference > 0 => matched LP text has lower MAE
```

This controls for the additional Stage-B training budget. The shuffled-text
branch remains the semantic-alignment placebo.

## Planned Implementation

### Variant matrix

Add `pair_pca_no_text_continued` immediately after the Stage-A no-text parent.
The rolling development matrix changes from:

```text
4 folds x 3 seeds x 6 variants = 72 runs
```

to:

```text
4 folds x 3 seeds x 7 variants = 84 runs
```

### Stage-B no-text configuration

The continuation control will use:

```yaml
text_embedding_mode: zero_lp
text_alignment_mode: matched
text_preprocessing_mode: pca
normalize_text_embedding: false
conditioning_mode: residual_film
critic_conditioning_mode: projection
initial_generator_checkpoint_path: <paired Stage-A checkpoint>
freeze_backbone_epochs: 5
lambda_film: 0.0
lambda_mismatch: 0.0
```

It shares the same fold, seed, raw-vol workbook, train-only PCA transform,
parent generator checkpoint, epoch budget, learning-rate schedule, checkpoint
rule, and fresh-critic behavior as the matched-text Stage-B branch.

### Statistical contrasts

The revised comparisons will include:

```text
incremental_text:
  no_text_continued - text_residual_pretrained

text_vs_parent:
  no_text_parent - text_residual_pretrained

continuation_effect:
  no_text_parent - no_text_continued

matched_vs_shuffled:
  shuffled_text - matched_text
```

The primary development table will select only `incremental_text`.

### Audit controls

The registry and checkpoint collection will record:

```text
training_stage
parent_variant
parent_checkpoint_path
parent_checkpoint_sha256
```

For every fold and seed, validation will require the no-text continuation,
matched-text residual, and shuffled-text residual branches to share the exact
same parent checkpoint SHA256 and text-transform SHA256.

## Planned Tests

- The continuation variant requires a parent checkpoint.
- It uses `zero_lp`, disables FiLM/mismatch penalties, and freezes the
  backbone for five epochs.
- All three Stage-B branches point to the same parent checkpoint.
- The primary contrast uses continued no-text rather than the Stage-A parent.
- Comparison output includes the continuation-effect diagnostic.
- The expected matrix size is 84 runs.
- Existing residual-generator initialization and freeze/unfreeze tests remain
  green.

## Implementation Record

Status: `implemented and targeted tests passed`

### Implemented files

```text
scripts/rq1_pair/rq1_pair_experiment.py
scripts/rq1_pair/start_training_matrix_background.sh
tests/test_scripts/test_rq1_pair_experiment.py
docs/summary/20260723-142444/rq1_no_text_continuation_control_implementation.md
```

Implemented behavior:

- Added `pair_pca_no_text_continued` as the second variant, after its Stage-A
  parent.
- Changed the experiment prefix to
  `rq1_pair_text_raw_vol_continuation_<run_ts>`.
- Changed the primary baseline from the Stage-A parent to the Stage-B
  continued no-text model.
- Preserved `text_vs_parent` and added `continuation_effect` diagnostics.
- Added training-stage and parent-checkpoint provenance to the launch and
  checkpoint registries.
- Added `checkpoint_selection/paired_stage_validation.csv`.
- Delayed writing `selected_checkpoints.csv` until all config, parent SHA, and
  text-transform SHA validation passes.
- Added explicit controlled-primary and parent/continuation output tables.
- Updated the background launcher to select only continuation experiment
  directories.

### Initialization fairness verified

The tests instantiate matched-text and continued no-text Stage-B trainers with
the same seed and parent checkpoint. At setup they have:

```text
identical generator state
identical freshly initialized critic state
identical parent checkpoint SHA256
```

After setup, their effective difference is the matched text input and the
associated residual/projection/mismatch losses.

### Verification

Commands:

```bash
conda run -n py312 python -m unittest \
  tests.test_scripts.test_rq1_pair_experiment

conda run -n py312 python -m unittest \
  tests.test_standalone_wgan.test_film_wgan_module \
  tests.test_scripts.test_rq1_experiment \
  tests.test_scripts.test_rq1_pair_experiment

conda run -n py312 python -m compileall -q \
  scripts/rq1_pair \
  tests/test_scripts/test_rq1_pair_experiment.py

bash -n scripts/rq1_pair/*.sh
git diff --check
```

Results:

```text
RQ1 pair targeted suite: 12 passed
Combined adjacent suite: 45 passed, 1 skipped
compileall: passed
shell syntax: passed
git diff --check: passed
```

The full repository test suite was not run because this change is isolated to
the RQ1 pair orchestration and its adjacent runtime contracts.

### Local input links

Git-ignored input files are not copied automatically when a worktree is
created. Two local symbolic links were created in the development worktree:

```text
data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

They point to the original worktree inputs. `prepare_experiment.sh` resolves
the links, copies the actual files into the new experiment archive, and records
their SHA256 values.

### Run commands

The new 84-run matrix has not been started. Run it from the development
worktree:

```bash
cd /home/haobin_cui/research_files_space_2/wgan_option-rq1-no-text-continuation
conda activate py312

bash scripts/rq1_pair/prepare_experiment.sh

CUDA_VISIBLE_DEVICES=1 \
bash scripts/rq1_pair/start_training_matrix_background.sh
```

Monitor:

```bash
bash scripts/rq1_pair/monitor_training.sh
```

After all 84 runs complete:

```bash
bash scripts/rq1_pair/collect_checkpoints.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/rq1_pair/run_generate_test_matrix.sh
bash scripts/rq1_pair/build_comparison_archive.sh
```

### Known limitation

The 2022-2023 rolling experiment remains development evidence. It does not
replace a future untouched 2024+ confirmation sample. No best-seed selection
is allowed in the controlled primary comparison.
