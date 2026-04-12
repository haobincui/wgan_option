# CNN WGAN

Standalone pure-adversarial CNN WGAN for `merged_vol.xlsx`.

This package is intentionally independent from both `wgan_option` and `volgan`.

Core task:

- input: current surface + text embedding + noise
- generator output: future `delta(log iv)` on the fixed grid
- reconstruction: `future_surface = exp(log(current_surface) + delta)`
- training: WGAN-GP adversarial loss plus optional calendar / butterfly / smoothness constraints
- no supervised `L1/MSE` reconstruction term

CLI:

```bash
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp.yaml
python scripts/cnn_wgan/main.py generate-result --config configs/cnn_wgan/generate_result_default.yaml
```
