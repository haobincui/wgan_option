# CNN WGAN — CNN-Based Wasserstein GAN for Volatility Surface Generation

## Directory Structure

```
cnn_wgan/
├── __init__.py          # Public API exports
├── config.py            # Configuration dataclasses and YAML loaders
├── models.py            # Neural network architectures (Generator, Critic)
├── losses.py            # Loss functions (WGAN loss, gradient penalty, smoothness)
├── data.py              # Data loading, normalization, Dataset definition
├── trainer.py           # Training loop and checkpoint management
├── inference.py         # Scenario sampling, arbitrage reweighting, result generation
├── arbitrage.py         # Arbitrage penalty computation and scenario reweighting
├── io.py                # Checkpoint / JSON / CSV I/O utilities
├── plotting.py          # Volatility surface visualization
└── training_plots.py    # Training curve visualization
```

## Model Architecture

This module uses a **CNN + WGAN-GP** architecture to conditionally generate future volatility surface log-IV increments from the current surface and text embeddings.

### Generator (`CnnWGANGenerator`)

Takes the current volatility surface, text embedding, and random noise as input; outputs a normalized log-IV delta.

```
                ┌──────────────┐
current_surface │  Surface CNN │──→ surface_features
  (1, H, W)     │  Encoder     │     (flattened)
                └──────────────┘
                                    ┌───────────┐
                ┌──────────────┐    │  Fusion   │
text_embedding  │  Text MLP    │──→ │  MLP      │──→ delta (surface_dim)
                │  Encoder     │    │           │
                └──────────────┘    └───────────┘
                                         ↑
                noise (noise_dim) ───────┘
```

**Surface CNN Encoder:**
- `Conv2d(1, C, 3×3, stride=1)` → LeakyReLU(0.2)
- `Conv2d(C, 2C, 3×3, stride=2)` → LeakyReLU(0.2)
- `Conv2d(2C, 4C, 3×3, stride=2)` → LeakyReLU(0.2)
- Configurable number of `_ResidualConvBlock(4C)` residual blocks
- Output flattened to a 1D vector

**Text MLP Encoder:**
- `Linear(embedding_dim, text_hidden_dim)` → LayerNorm → LeakyReLU(0.2)
- `Linear(text_hidden_dim, text_out_dim)` → LeakyReLU(0.2)

**Fusion MLP:**
- Input: concatenation of `[surface_features, text_features, noise]`
- Three fully-connected layers → output `surface_dim`-dimensional delta

### Critic (`CnnWGANCritic`)

Evaluates the realism of (current surface, future surface, text embedding) tuples, outputting a scalar score.

```
current_surface ┐
                ├─ stack(dim=1) → (2, H, W) → Surface CNN Encoder → surface_features
future_surface  ┘                                                        │
                                                                    ┌────┴────┐
text_embedding ───→ Text MLP Encoder ──→ text_features ──→          │Classifier│→ score
                                                                    └─────────┘
```

**Surface CNN Encoder:**
- 2-channel input (current + future surfaces stacked)
- Three stride-2 convolutional layers + residual blocks
- One additional stride-2 downsampling compared to the Generator

**Classifier:**
- `Linear(surface_feat_dim + text_out_dim, fusion_hidden_dim)` → LeakyReLU(0.2)
- `Linear(fusion_hidden_dim, 1)` → scalar output

### Residual Block (`_ResidualConvBlock`)

- `Conv2d(C, C, 3×3)` → LeakyReLU(0.2) → `Conv2d(C, C, 3×3)` + skip connection
- Output activated with LeakyReLU(0.2)

### Surface Reconstruction

```python
future_surface = exp(log(clamp(current, min=1e-4)) + delta)
```

Operates in log-IV space to ensure positive output values.

## Implementation Details

### Training Pipeline (`CnnWGANTrainer`)

Per epoch:

1. **Critic update** (repeated `critic_iter=5` times per batch):
   - Generate fake delta → reconstruct fake future surface
   - Compute Critic scores on real/fake surfaces
   - Critic loss: `E[fake_score] - E[real_score] + λ_gp × GP`
   - Gradient penalty (WGAN-GP): `E[(||∇_x critic(interpolated)||₂ - 1)²]`

2. **Generator update** (once per batch):
   - Adversarial loss: `-E[fake_score]`
   - Calendar arbitrage penalty: ensures call prices increase with maturity
   - Butterfly arbitrage penalty: ensures convexity in the strike dimension
   - Smoothness regularization: penalizes weighted second-order differences along maturity and strike dimensions
   - Total loss: `adv + λ_cal × calendar + λ_but × butterfly + λ_smooth × smooth`
   - Pure adversarial training: no supervised L1/MSE reconstruction term

### Loss Functions

| Loss | Formula | Default Weight |
|------|---------|----------------|
| WGAN Critic | `E[fake] - E[real] + λ_gp × GP` | `λ_gp = 10.0` |
| WGAN Generator | `-E[fake]` | — |
| Calendar Arbitrage | `mean(relu(C(τ₁,K) - C(τ₂,K)))`, τ₁ < τ₂ | `λ_cal = 2.0` |
| Butterfly Arbitrage | `mean(relu(-d²C/dK²))` | `λ_but = 2.0` |
| Maturity Smoothness | Weighted second differences (weights = 1/Δτ²) | `λ_smooth = 0.1` |
| Strike Smoothness | Weighted second differences (weights = 1/ΔK²) | (included in smooth) |

### Data Processing

- Input data from `merged_vol.xlsx` workbook (sheet: `gan_input_ready`)
- Surfaces stored in log-IV form: `log(clamp(surface, min=1e-4))`
- Delta normalization: `(delta - mean) / std` (statistics computed on training set only)
- Text embedding modes: `'hd'` (high-dim), `'lp'` (low-dim), `'concat'` (both)
- Train/val split chronologically ordered (`train_ratio=0.8`)

### Inference & Scenario Generation (`CnnWGANSampler`)

1. Load checkpoint and normalization statistics
2. Generate `mc_samples=64` scenarios per sample:
   - Deterministic seed-based noise sampling (reproducible)
   - Generator forward → delta → denormalize → reconstruct future surface
3. Compute arbitrage penalties per scenario
4. Softmax reweighting: `w_i = exp(-β × penalty_i) / Σ exp(-β × penalty_j)`
5. Weighted aggregation → summary surface, quantiles (0.05, 0.50, 0.95), evaluation metrics

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_dim` | 32 | Generator noise dimensionality |
| `gen_base_channels` | 32 | Generator CNN base channel count |
| `disc_base_channels` | 32 | Critic CNN base channel count |
| `gen_res_blocks` | 1 | Number of residual blocks in Generator |
| `text_hidden_dim` | 256 | Text encoder hidden dimension |
| `text_out_dim` | 128 | Text feature output dimension |
| `fusion_hidden_dim` | 512 | Fusion MLP hidden dimension |
| `critic_iter` | 5 | Critic updates per Generator update |
| `batch_size` | 32 | Batch size |
| `num_epochs` | 100 | Training epochs |
| `learning_rate` | 1e-4 | Learning rate (Adam, β₁=0.5, β₂=0.9) |
| `checkpoint_metric` | `val_mae_gap_vs_current` | Best checkpoint selection metric |

### CLI

```bash
python scripts/cnn_wgan/main.py train --config configs/cnn_wgan/train_lp.yaml
python scripts/cnn_wgan/main.py generate-result --config configs/cnn_wgan/train_lp.yaml
bash run_cnn_wgan_svi_excel.sh configs/cnn_wgan/train_lp_gen128_disc128.yaml
```

### Checkpoint Structure

Saved contents: training config, surface shape, grid coordinates, normalization statistics, Generator and Critic state_dicts.

Output files: `cnn_wgan_best.pt`, `cnn_wgan_final.pt`, `training_metrics.json`, `loss_curves.png`, etc.
