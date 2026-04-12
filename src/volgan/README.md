# Standalone VolGAN Module

This package is a fully independent implementation of a VolGAN-style research workflow for this repository.

It is intentionally separate from `wgan_option`:

- no imports from `wgan_option`
- no shared config dataclasses
- no shared trainer or inference helpers
- no shared CLI entrypoints

The only shared contract is the external workbook format: `merged_vol.xlsx`.

## Goal

This module adapts the VolGAN paper's core ideas to the repo's current thesis-facing merged-vol workflow:

`current_surface + text_embedding + noise -> future log-IV increment -> future surface`

In contrast with the existing `wgan_option` WGAN path, this module uses:

- an MLP conditional GAN
- BCE adversarial training
- smoothness penalties in maturity and strike directions
- post-sampling arbitrage reweighting

## Expected Input

The module reads `merged_vol.xlsx`, typically sheet `gan_input_ready`.

Each usable row is expected to contain:

- `sample_id`
- `news_timestamp_utc`
- `current_snapshot_time_utc`
- `target_snapshot_time_utc`
- `strike_grid`
- `maturity_days_grid`
- `current_surface_flat`
- `target_surface_flat`
- `hd_embedding`
- `lp_embedding`

If present, `training_candidate_flag == 1` rows are preferred.

The repo semantics remain:

- `current = backward`
- `future = forward`

## Configs

Configs live in `configs/volgan/`.

Training configs:

- `train_default.yaml`
- `train_lp.yaml`
- `train_concat.yaml`

Sampling config:

- `sample_default.yaml`

## CLI

Train:

```bash
python scripts/volgan/main.py train --config configs/volgan/train_default.yaml
```

Sample:

```bash
python scripts/volgan/main.py sample --config configs/volgan/sample_default.yaml
```

## Training Logic

The generator outputs a future log-IV increment on the fixed grid.

Reconstruction is:

`future_surface = exp(log(current_surface_clamped) + delta)`

The discriminator sees:

- current surface
- text embedding
- candidate future increment

The generator loss combines:

- BCE adversarial loss
- strike smoothness penalty
- maturity smoothness penalty

Optional gradient-norm matching estimates `alpha_m` and `alpha_tau` before the main training loop.

## Sampling and Reweighting

Sampling generates multiple future surfaces per row using different noise draws.

For each generated surface, the module computes:

- calendar arbitrage penalty
- butterfly arbitrage penalty

These are combined into one scalar penalty and converted into scenario weights:

`w_i propto exp(-beta * penalty_i)`

The sampler then writes:

- weighted mean surface
- weighted quantile surfaces
- per-scenario penalties and weights
- summary CSV metrics

## Python Usage

```python
from volgan.config import load_train_config, load_sample_config
from volgan.trainer import VolGANTrainer
from volgan.inference import VolGANSampler

train_config = load_train_config("configs/volgan/train_default.yaml")
run_dir = VolGANTrainer(train_config).train()

sample_config = load_sample_config("configs/volgan/sample_default.yaml")
sample_config.checkpoint_path = str(run_dir / "checkpoints" / "volgan_best.pt")
sample_run_dir = VolGANSampler(sample_config).sample()
```

## Output Layout

Training outputs are stored under a standalone root such as:

- `outputs/volgan/train/<run_ts>/checkpoints`
- `outputs/volgan/train/<run_ts>/metrics`

Sampling outputs are stored under:

- `outputs/volgan/sample/<run_ts>/samples`
- `outputs/volgan/sample/<run_ts>/summary.csv`
