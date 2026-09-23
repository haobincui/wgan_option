# wgan_option

Research code for building U.S. Treasury option implied-volatility representations and testing whether news text improves short-horizon surface forecasts.

This repository supports the full thesis workflow: raw option processing, surface construction, news alignment, model training, controlled RQ1/RQ2 comparisons, RQ3 robustness analysis, and reproducible experiment packaging. The current thesis-facing path uses corrected raw TY option implied-volatility observations and a five-minute `current -> future` target.

> **Current status (August 2026):** the FiLM-WGAN v3 engineering and audit pipeline is complete at the validation-pilot stage. The latest 12-run pilot did **not** pass scientific admission: matched news text did not outperform the no-text continuation, shuffled-text placebo, or persistence baselines. The outer test remains untouched, so there is no confirmatory RQ1 result yet.

## Start here

### 1. Create the tested environment

The research orchestrators expect a conda environment named `py312`. The current workspace was verified with Python 3.12.13 and PyTorch 2.6.0+cu124.

```bash
conda create -n py312 python=3.12 -y
conda activate py312
python -m pip install --upgrade pip

# Install the PyTorch build appropriate for your machine first.
# This is the CUDA 12.4 command used by the current GPU environment:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements.txt
python -m pip install -e .
```

Check that the intended Python and PyTorch installation are active:

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('cuda=', torch.cuda.is_available())"
python scripts/film_wgan/main.py --help
python scripts/rq1_pair/rq1_pair_experiment.py --help
```

CPU-only execution is sufficient for parsing, audits, and tests. Training launchers assume CUDA unless their configuration explicitly says otherwise.

### 2. Supply the research inputs

Large and licensed inputs are intentionally excluded by `.gitignore`. A fresh clone therefore does not contain a runnable thesis dataset.

| Required path | Purpose |
| --- | --- |
| `data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz` | Daily raw TY option trades |
| `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx` | Factiva news metadata and frozen HD/LP embeddings |
| `data/reference/` | Rates, CME expiries/session closures, and scheduled-event calendars; the small reference files are tracked |
| `data/processed/text_features/rq2/<run>/` | Frozen BoW and sentiment artifacts required by the full RQ1-RQ3 driver |

The current v3 pilot additionally expects the frozen session-aligned workbook:

```text
data/processed/raw-excel-session/
└── rq123_cme_session_20260729-131219/
    └── merged_vol_rq2_text.xlsx
```

The filename contains `rq2_text` because the workbook also carries RQ2 feature columns; it is still the canonical input used by the current RQ1 v3 pilot.

### 3. Run smoke tests before using data or GPUs

```bash
conda activate py312

python -m unittest discover \
  -s tests/test_standalone_wgan \
  -p 'test_film_wgan*.py'

python -m unittest tests.test_scripts.test_rq1_pair_experiment
```

Run the whole suite with:

```bash
bash run_all_tests.sh
```

## Choose the right workflow

| Goal | Use this entrypoint | Status |
| --- | --- | --- |
| Reproduce or extend the latest RQ1 FiLM-WGAN protocol | `scripts/rq1_pair/start_v3_pilot_gpu0.sh` | Current v3 diagnostic protocol |
| Run the broad corrected RQ1-RQ3 workflow | `scripts/rq123/start_corrected_pipeline_background.sh` | Current project-wide driver, but not yet migrated to the isolated v3 protocol |
| Build and validate one corrected raw-vol workbook | `scripts/raw_vol/prepare_raw_vol_dataset.sh` | Current data-construction path |
| Train the reusable merged-workbook models | `scripts/train/main.py` | Supported general workflow |
| Compare standalone conditioning architectures | `scripts/{volgan,cnn_wgan,transformer_wgan,film_wgan,stylemod_wgan,crossattn_wgan}/main.py` | Research baselines |
| Inspect older code or results | `scripts/archive/` and historical output folders | Legacy/diagnostic only |

Do not use the generic `configs/film_wgan/train_default.yaml` as a substitute for the thesis protocol. It retains legacy row/full-grid defaults for compatibility. The v3 experiment must be prepared through the RQ1 orchestrator so that splits, support, PCA, text alignment, negative plans, hashes, and run fingerprints are frozen together.

## Research design

The central question is:

> Does correctly aligned news text improve a five-minute bond-option volatility-surface forecast beyond the same model trained for the same additional budget without text?

The three thesis workstreams are:

- **RQ1 — incremental LP-text value:** matched LP text versus a shared-parent no-text continuation and a shuffled-text placebo.
- **RQ2 — text representation:** LP embeddings versus train-fold-only BoW and LLM-sentiment representations. RQ2 is implemented, but has not yet been migrated to the v3 symmetric-negative protocol.
- **RQ3 — conditional robustness:** frozen-model behavior around scheduled news and market jumps. This is predictive robustness analysis, not a causal event study.

### Semantics that must remain explicit

- In older generated workbooks, `backward = current` and `forward = future`.
- In the corrected raw workflow, availability time `t` defines non-overlapping half-open quote windows: current `[t-5m, t)` and target `[t, t+5m)`.
- News timestamps are interpreted in `Europe/London`; a configured publication-availability lag may shift `t`.
- Raw IV uses CME TY option expiry, Black76, frozen rates and prior futures, with OTM-preferred filtering.
- `merged_vol.xlsx` is already pair-based: one usable row contains `current_surface -> target_surface`.
- `merged_svi.xlsx` is direction-level: SVI training pairs backward and forward rows later via `news_row_id`.
- Surface-pair splits are grouped and chronological; a pair must never cross train, validation, and test.
- Normalization, support masks, text transforms, and hard-negative plans are fitted or frozen from the permitted development data only.
- Seven-day ATM is not reported for the irregular raw-support experiment because it lies below observed maturity support.

## Current FiLM-WGAN v3 protocol

The current model is **residual FiLM**, not the older design that applied FiLM after every convolution.

```text
current raw-IV surface + support mask
        │
        ├── no-text CNN backbone ───────────────► base log-IV delta
        │
LP text ─► deduplicate/pool ─► train-only PCA-128
                                      │
                                      └── bottleneck FiLM + text adapter
                                                       │
base delta + tanh(text gate) * text delta ─────────────┘
        │
        └── current + delivered delta ─► future surface
```

The text gate is initialized at zero, so the text model starts exactly nested inside the no-text parent. Stage B then compares equal-budget arms from the same Stage-A checkpoint:

| Arm | Purpose |
| --- | --- |
| `pair_pca_no_text_residual` | Stage-A no-text parent |
| `pair_pca_no_text_continued` | Equal-budget no-text continuation control |
| `pair_pca_text_residual_pretrained` | Correctly matched LP text |
| `pair_pca_shuffled_residual_pretrained` | Frozen shuffled-text placebo |

Key v3 safeguards:

- the WGAN realism scalar is separate from the bounded transition-text matching head;
- the matching head sees the standardized log-IV transition, not the current IV level;
- symmetric hard negatives protect both the native and placebo positives;
- identity, lineage overlap, and near-duplicate embeddings are excluded;
- raw support is fitted on the train fold only;
- gradient penalty is support-aware and audits unsupported gradients;
- checkpoint selection is fixed to validation MAE; matcher diagnostics cannot select checkpoints;
- protocol versions, plans, inputs, resolved configs, and checkpoints carry hashes/fingerprints;
- admission is fail-closed, and validation failure prevents outer-test generation.

Implementation entry points:

- `src/film_wgan/models.py` — residual-FiLM generator and transition-matching critic
- `src/film_wgan/data.py` — pair construction, lineage, splits, transforms, and alignment
- `src/film_wgan/matching.py` — symmetric hard-negative assignment
- `src/film_wgan/losses.py` — matching losses and support-aware gradient penalty
- `src/film_wgan/trainer.py` — Stage-B initialization, training, diagnostics, and checkpoints
- `src/film_wgan/inference.py` — deterministic Monte Carlo evaluation and exports
- `src/film_wgan/protocol.py` — protocol schemas and hashes

### Run a fresh v3 validation pilot

The launcher is deliberately strict:

- it runs physical GPU 0;
- it fixes fold `2023Q1`, seeds `42 202 404`, and the four arms above;
- it refuses a dirty Git worktree;
- it refuses an existing root whose frozen commit or input manifest does not match;
- it runs validation only and never consumes the outer test.

Always provide a new experiment root. The launcher's no-argument default points at an older development root and should not be reused.

```bash
git status --short
# Continue only when the command above prints nothing.

FILM_WGAN_CONDA_ENV=py312 \
FILM_WGAN_MAX_PARALLEL=5 \
bash scripts/rq1_pair/start_v3_pilot_gpu0.sh \
  outputs/experiments/rq1_film_wgan_v3_pilot_$(date -u +%Y%m%d-%H%M%S)
```

To reproduce the exact `r4` protocol, use clean commit `4770605`, the frozen input workbook and manifests, and a new output root. GPU-level bitwise identity is not guaranteed. Do not reuse v2 checkpoints or any earlier v3 debug root.

The authoritative result of a completed pilot is:

```text
<experiment-root>/final_tables/validation_pilot_summary.json
```

Supporting tables live in `final_tables/`, per-pair comparisons in `comparisons/`, run/checkpoint audits in `checkpoint_selection/`, and frozen provenance in `inputs/`.

## Current RQ1 v3 validation result

The latest complete artifact in the research workspace is:

```text
outputs/experiments/rq1_film_wgan_v3_symmetric_negative_pilot_r4/
```

It was produced from clean film commit `4770605` and contains 12/12 completed runs: one 2023Q1 validation fold, three seeds, four arms, and 133 validation surface pairs per arm. Q2-Q4 were prepared but not run. Generated outputs are ignored by Git and therefore are not distributed with a fresh clone.

Mean validation metrics across the three seeds:

| Forecast | Supported-surface MAE | Short-ATM MAE | Shortest-supported ATM absolute error |
| --- | ---: | ---: | ---: |
| Current-surface persistence | 0.001719529 | 0.001640669 | 0.001515169 |
| Stage-A no-text parent | 0.001818275 | 0.001642036 | 0.001523841 |
| No-text continuation | 0.001815002 | 0.001635068 | 0.001522682 |
| Matched LP text | 0.001815018 | 0.001634996 | 0.001522479 |
| Shuffled text | **0.001814768** | **0.001634838** | **0.001521924** |

Lower is better. These are descriptive pilot values, not confirmatory estimates.

- Matched text versus no-text continuation changed surface MAE by `-1.60e-8` in the preregistered `baseline minus matched` direction; matched won in only 1/3 seeds.
- Matched text versus shuffled text changed surface MAE by `-2.50e-7`; matched won in 0/3 seeds.
- Matched text was worse than persistence by `9.55e-5` in mean supported-surface MAE.
- Replacing native text with its placebo alignment changed the generated surface by only `4.16e-8` MAE on average, indicating that the generator was effectively insensitive to text.
- The fail-closed admission check passed 4 of 10 gates. Matching quality, matched-text MAE, duplicate-free direction, matching-gradient bounds, and cross-seed GP stability were among the failed checks.
- Transition clipping, numerical round-trip, shuffled-at-chance behavior, and zero unsupported GP gradients passed.

**Interpretation:** the implementation and audit chain are operational, but this frozen configuration has not demonstrated useful incremental text signal. This does not prove that news has no signal under every alignment, representation, horizon, or architecture; it does mean the project must not unlock the outer test or report a positive RQ1 finding from this pilot.

## Full corrected RQ1-RQ3 driver

The broad project-wide pipeline snapshots external inputs, builds and validates the corrected raw-vol workbook, runs RQ1/RQ2 matrices, executes RQ3 robustness analysis, records provenance, and can optionally package the experiment:

```bash
GPU_IDS="0 1" RUNS_PER_GPU=2 \
bash scripts/rq123/start_corrected_pipeline_background.sh
```

Monitor the most recent run:

```bash
bash scripts/rq123/monitor_corrected_pipeline.sh
```

Important: this broad driver currently uses `train_rq1_pair_textbase.yaml` and `train_rq2_pair_textbase.yaml`. It is not the same protocol as the isolated v3 symmetric-negative pilot. Treat its outputs as their own frozen development protocol; do not combine estimates across the two.

To rebuild only the CME-session-aligned workbook:

```bash
ENV_NAME=py312 \
bash scripts/rq123/build_session_aligned_dataset.sh
```

## Chapter 3 shared-market bootstrap correction

The analysis-only v2 entrypoint recomputes RQ1–RQ3 and architecture/alignment
inference from frozen pair-level errors. Every bootstrap draw shares its
fold/session sampling weights across model seeds and paired conditions. The
quarter resampling and the original equal-cell or pooled-pair estimands are
retained; RQ4 remains outside this correction.

```bash
python -m scripts.rq123.chapter3_shared_panel_bootstrap_v2
python -m scripts.rq123.chapter3_shared_panel_bootstrap_v2 --verify-only
python -m scripts.rq123.verify_chapter3_bootstrap_bindings
```

Results, replayable schedules, input/output hashes, and the old/new inference
comparison are written to
`outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2`.
An existing output directory is never overwritten. Verification replays the
saved weights against the archived canonical panels and checks frozen inputs;
neither command trains models or regenerates predictions. The historical
nested-bootstrap scripts and experiment artifacts remain available for audit.
The final command separately checks the manuscript's numerical bindings in
`docs/chapter3_bootstrap_bindings.json`; it does not edit the manuscript or
replace a LaTeX compilation.
The manuscript preserves the approved reporting conventions: the main summary
tables display v2 bootstrap-draw mean MAEs, paired RQ1–RQ3 inference retains
original-sample equal-cell log ratios, and architecture/alignment robustness
retains original-sample pooled-pair estimates. Validation-trajectory MAEs remain
observed equal-cell summaries. These quantities are explicitly distinguished;
one must not reconstruct a paired log ratio from displayed bootstrap means.
The explicit source mappings can also be checked for completeness and drift:

```bash
python -m scripts.rq123.build_chapter3_bootstrap_bindings --check
```

See [the integration notes](docs/chapter3_shared_panel_bootstrap_v2_integration_notes.md)
for the authorized resolution of concurrent edits, inferential limitations,
verification results, and the remaining full-thesis rendering checks.

## Reusable data and model workflows

### Build a corrected raw-vol workbook

```bash
ENV_NAME=py312 \
DEVICE=cpu \
RUN_TS=raw_vol_$(date -u +%Y%m%d-%H%M%S) \
bash scripts/raw_vol/prepare_raw_vol_dataset.sh
```

This generates and validates:

```text
data/processed/raw-excel/<run-id>/
├── merged_vol.xlsx
├── raw_vol_dataset_validation.json
└── generation and lineage artifacts
```

Add the frozen RQ2 text features when needed:

```bash
FEATURE_DIR=data/processed/text_features/rq2/<feature-run> \
bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh \
  data/processed/raw-excel/<run-id>
```

### Generate and merge an SVI-based workbook

```bash
python scripts/generate_surface/main.py generate_surface \
  --device gpu \
  --model svi \
  --data_range excel \
  --config configs/surface_builder/svi/generate_surface-svi-excel.yaml

python scripts/merge_file/merge_svi.py \
  --input-dir data/processed/svi-excel/<run-id>

python scripts/merge_file/merge_vol.py \
  --input-dir data/processed/svi-excel/<run-id>
```

### Train the general merged-workbook models

Review and copy a config before changing paths; thesis configs often contain frozen workspace-specific inputs.

```bash
python scripts/train/main.py vol-xlsx \
  --config configs/wgan/train_vol_xlsx.yaml

python scripts/train/main.py vol-regression-xlsx \
  --config configs/wgan/train_vol_regression_xlsx.yaml

python scripts/train/main.py svi-xlsx \
  --config configs/wgan/train_svi_xlsx.yaml
```

Standalone model CLIs follow the same pattern:

```bash
python scripts/cnn_wgan/main.py train \
  --config configs/cnn_wgan/train_lp_gen128_disc128.yaml

python scripts/transformer_wgan/main.py train \
  --config configs/transformer_wgan/train_lp.yaml

python scripts/film_wgan/main.py train \
  --config configs/film_wgan/train_lp_gen128_disc128.yaml
```

For these standalone CLIs, `train` normally performs training followed by result generation, while `sample` is an alias for `generate-result`. These general configurations are architectural baselines, not the frozen RQ1 v3 experiment.

## Data flow and repository map

```text
raw TY option trades + news workbook + market references
        │
        ├── scripts/raw_vol / scripts/generate_surface
        │       └── data/processed/<dataset>/<run-id>/
        │
        ├── scripts/merge_file / scripts/rq2 enrichment
        │       └── merged_vol.xlsx / merged_svi.xlsx / merged_params.xlsx
        │
        ├── scripts/train or scripts/rq1_pair / scripts/rq2_pair
        │       └── checkpoints, metrics, manifests, hashes
        │
        └── scripts/generate_result / scripts/analyze_error / scripts/rq3
                └── comparisons, figures, final tables, packaged evidence
```

| Path | Responsibility |
| --- | --- |
| `src/quantlib/` | Calendars, pricing, implied volatility, SVI/SABR/raw surfaces, and interpolation |
| `src/market_data/` | Contract parsing and market-data DTOs |
| `src/wgan_option/` | General merged-workbook WGAN/regression/SVI training |
| `src/film_wgan/` | Current residual-FiLM research model and protocol support |
| `src/{volgan,cnn_wgan,transformer_wgan,stylemod_wgan,crossattn_wgan}/` | Standalone architectural baselines |
| `scripts/raw_vol/` | Corrected raw-IV construction and audits |
| `scripts/rq1_pair/` | RQ1 Stage-A/Stage-B orchestration |
| `scripts/rq2_pair/` | RQ2 representation experiments |
| `scripts/rq3/` | Scheduled-news and market-jump robustness |
| `configs/` | Executable experiment and pipeline configurations |
| `tests/` | Unit, CLI, semantic, and protocol-tamper tests |
| `docs/` | Thesis design, methodology, architecture reviews, and historical result notes |
| `outputs/experiments/` | Generated experiment artifacts; ignored by Git |

## Reproducibility rules

- Start formal experiments from a clean worktree and preserve the exact commit.
- Never overwrite a frozen experiment root; create a new timestamped root.
- Keep input manifests, resolved configs, split/support/text plans, hashes, and Git state with every run.
- Do not reuse v2 FiLM checkpoints in v3. The v2 shuffled arm had asymmetric negative contamination and is superseded.
- Do not pool old SVI, pre-corrected raw-vol, v2, and v3 estimates as if they were one protocol.
- Do not read the outer test until every validation admission gate passes.
- Do not claim static-arbitrage or seven-day ATM results on irregular raw support where those metrics are explicitly unavailable.

## Known limitations

- The current v3 result is one validation fold with three seeds, not the planned multi-fold confirmatory experiment.
- The generator is nearly insensitive to text and its Monte Carlo scenarios are severely under-dispersed.
- The no-text model does not beat persistence on the full supported surface.
- Support-aware GP is numerically correct on unsupported cells but unstable across v3 seeds.
- Calendar/butterfly/smoothness penalties are disabled for irregular raw support until support-aware edge penalties are implemented.
- RQ2 has not yet adopted the v3 transition-matching and symmetric-negative protocol.
- The frozen embedding vectors are available locally, but their upstream API request/preprocessing provenance is incomplete.
- FiLM checkpoints do not contain optimizer/scheduler state, so interrupted jobs restart rather than resume at the exact epoch.
- Dependency files constrain version ranges but do not lock the complete environment; preserve the working conda/PyTorch/CUDA stack with formal artifacts.
- Generated datasets and outputs are ignored; sharing a result requires packaging its manifests and tables explicitly.

## Documentation

Read these next:

- [Current executable workflows](docs/current_executable_workflows.md)
- [Raw-vol interpolation for RQ1/RQ2](docs/raw_vol_interpolation_rq1_rq2.md)
- [Vol-surface input design](docs/input_vol.md)
- [SVI input design](docs/input_svi.md)
- [Vol-surface GAN architecture](docs/vol_surface_gan_architecture.md)
- [FiLM-WGAN architecture review](docs/summary/20260809-005100/film_wgan_architecture_review.md)
- [RQ2 BoW design](docs/rq2_bow_film_wgan.md)
- [RQ2 sentiment design](docs/rq2_llm_sentiment_film_wgan.md)
- [RQ3 calculation logic](docs/rq3_event_study_calculation_logic.md)
- [Thesis experiment status](docs/thesis_experiment_plan_status.md)
- [Script index](scripts/README.md)

Some historical documents describe older SVI grids, projection critics, or per-layer FiLM. When a document conflicts with `src/`, `scripts/`, a resolved config, or a frozen experiment manifest, treat the executable artifact as the implementation record and call out the mismatch rather than silently merging protocols.

## Research-use note

This is a thesis research repository, not a production trading system. Results are experimental, data access may be licensed, and no output should be interpreted as investment advice.
