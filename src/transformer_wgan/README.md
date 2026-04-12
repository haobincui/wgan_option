# Transformer WGAN

Standalone Transformer-based WGAN for `merged_vol.xlsx`.

This package is intentionally independent from `wgan_option`, `volgan`, and `cnn_wgan`.

Core task:

- input: current surface + text embedding + noise
- generator output: future surface delta on the fixed grid
- reconstruction: `future_surface = softplus(current_surface + delta) + 1e-4`
- critic input: current surface + candidate future surface + text embedding
- training: WGAN-GP adversarial loss with configurable objective mode

Objective modes:

- `pure_adversarial: false`
  - `adv + lambda_recon * recon + optional calendar / butterfly / smooth + optional delta_shrink`
- `pure_adversarial: true`
  - `adv + optional calendar / butterfly / smooth`
  - no direct target-matching loss
  - no `delta_shrink`

CLI:

```bash
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp_pure_adv.yaml
python scripts/transformer_wgan/main.py generate-result --config configs/transformer_wgan/generate_result_default.yaml
```
