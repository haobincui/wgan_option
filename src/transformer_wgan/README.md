# Transformer WGAN — Transformer-Based Wasserstein GAN for Volatility Surface Generation

## Directory Structure

```
transformer_wgan/
├── __init__.py          # Public API exports
├── config.py            # Configuration dataclasses and YAML loaders
├── models.py            # Neural network architectures (Generator, Critic)
├── losses.py            # Loss functions (WGAN loss, gradient penalty, arbitrage, smoothness)
├── data.py              # Data loading, normalization, Dataset definition
├── trainer.py           # Training loop and checkpoint management
├── inference.py         # Scenario sampling, arbitrage reweighting, result generation
├── io.py                # Checkpoint / JSON / CSV I/O utilities
├── plotting.py          # Volatility surface visualization
└── training_plots.py    # Training curve visualization
```

## Model Architecture

This module uses a **Transformer Encoder + WGAN-GP** architecture, leveraging self-attention to capture long-range dependencies within the volatility grid. Compared to CNN WGAN, the Transformer can directly model correlations between any two points in the grid.

### Generator (`TransformerWGANGenerator`)

Treats grid points as a token sequence, prepends special text and noise tokens, passes through a Transformer encoder, and outputs a delta value for each grid point.

```
                   ┌────────────────────────────────────────────┐
                   │         Transformer Encoder                │
                   │  [text_token, noise_token, surface_tokens] │
                   └──────────────────┬─────────────────────────┘
                                      │
                              surface_tokens (skip first 2)
                                      │
                               Output Head → delta (H×W)
```

**Input processing:**
- **Surface tokens**: Each grid point value projected via `Linear(1, model_dim)`, plus learnable 2D positional embeddings (row + col)
- **Text token**: Text embedding projected through a 5-layer MLP (LayerNorm + GELU) to a `model_dim`-dimensional token
- **Noise token**: Noise projected through a 3-layer MLP (GELU) to a `model_dim`-dimensional token

**Transformer Encoder:**
- Pre-norm architecture (`norm_first=True`)
- GELU activation, `batch_first=True`
- Defaults: `gen_layers=4`, `num_heads=8`, `ffn_dim=512`, `dropout=0.1`
- Input and output LayerNorm

**Output:**
- Extract the surface token portion of the encoded sequence (skip the first 2 special tokens)
- `Linear(model_dim, 1)` → squeeze → delta vector (H×W)

### Critic (`TransformerWGANCritic`)

Uses a CLS token to aggregate global information, evaluating the realism of (current surface, future surface, text) tuples.

```
                   ┌────────────────────────────────────────────────┐
                   │            Transformer Encoder                 │
                   │  [CLS_token, text_token, surface_pair_tokens]  │
                   └──────────────────┬─────────────────────────────┘
                                      │
                               CLS_token output
                                      │
                               Score Head → scalar
```

**Input processing:**
- **Surface pair tokens**: Current and future surfaces concatenated along the feature dimension → `Linear(2, model_dim)` + 2D positional embeddings
- **CLS token**: Learnable parameter `(1, 1, model_dim)`
- **Text token**: Same text encoder as the Generator

**Transformer Encoder:**
- Defaults: `disc_layers=3` (one fewer layer than Generator), other parameters shared

**Score Head:**
- `Linear(model_dim, model_dim)` → GELU → `Linear(model_dim, 1)`

### 2D Positional Embeddings

```python
positional = row_embedding[i] + col_embedding[j]  # Additive 2D encoding
```

- `row_embedding`: `(surface_height, model_dim)` — encodes the maturity dimension
- `col_embedding`: `(surface_width, model_dim)` — encodes the strike dimension
- Initialization: `randn * 0.02`

### Surface Reconstruction

```python
future_surface = softplus(current_surface + delta) + 1e-4
```

Unlike CNN WGAN, Transformer WGAN uses **level delta** (not log-IV delta), with `softplus` ensuring positive values.

## Implementation Details

### Training Pipeline (`TransformerWGANTrainer`)

Per epoch:

1. **Critic update** (repeated `critic_iter=5` times per batch):
   - Generate fake delta → `softplus(current + delta)` to reconstruct fake future surface
   - Critic loss: `E[fake_score] - E[real_score] + λ_gp × GP`
   - Gradient penalty requires double-backward; forces **Math SDPA** kernel (Flash Attention does not support second-order gradients)

2. **Generator update** (once per batch):
   - Adversarial loss: `-E[fake_score]`
   - Reconstruction loss: `λ_recon × L1(fake_future, target_future)` (disabled when `pure_adversarial=True`)
   - Calendar arbitrage penalty: based on Black-Scholes relative call pricing
   - Butterfly arbitrage penalty: convexity constraint in the strike dimension
   - Smoothness regularization: L2 differences between neighboring grid points
   - Delta shrink penalty: `mean(|fake_future - current|)` (encourages small changes, off by default)
   - **Constraint warmup**: arbitrage/smoothness penalties disabled for the first `constraint_warmup_epochs` epochs

### Loss Functions

| Loss | Formula | Default Weight |
|------|---------|----------------|
| WGAN Critic | `E[fake] - E[real] + λ_gp × GP` | `λ_gp = 10.0` |
| WGAN Generator | `-E[fake]` | — |
| Reconstruction (L1) | `mean(\|fake - target\|)` | `λ_recon = 10.0` |
| Calendar Arbitrage | `mean(relu(C(τ₁,K) - C(τ₂,K)))` | `λ_cal = 2.0` |
| Butterfly Arbitrage | `mean(relu(-(C[K-1] - 2C[K] + C[K+1])))` | `λ_but = 2.0` |
| Smoothness | `L2 differences (maturity + strike directions)` | `λ_smooth = 0.1` |
| Delta Shrink | `mean(\|fake_future - current\|)` | `λ_delta_shrink = 0.0` (off) |

Loss assembly is managed by `assemble_generator_loss()`, supporting `pure_adversarial` mode switching.

**Objective modes:**
- `pure_adversarial: false` — `adv + λ_recon × recon + optional calendar/butterfly/smooth + optional delta_shrink`
- `pure_adversarial: true` — `adv + optional calendar/butterfly/smooth` (no target-matching loss, no delta_shrink)

### Double-Backward Compatibility

WGAN-GP's gradient penalty requires differentiating through Critic gradients (second-order derivatives). Flash Attention and Memory-Efficient Attention do not support this. The `_math_sdpa_context()` context manager forces the Math SDPA backend during gradient penalty computation.

### Advanced Training Features

**Learning rate scheduling:**
- Optional `ReduceLROnPlateau` (`use_reduce_lr_on_plateau=False` by default)
- Parameters: `reduce_lr_factor=0.5`, `reduce_lr_patience=8`, `reduce_lr_min_lr=1e-5`

**Early stopping:**
- Optional (`use_early_stopping=False` by default)
- Parameters: `early_stopping_patience=10`, `early_stopping_min_delta=0.0`

**Baseline-aware evaluation metrics:**
- `val_recon`: MAE between generated and target surfaces
- `val_current_recon`: MAE between current and target surfaces (baseline)
- `val_baseline_gap`: `val_recon - val_current_recon`
- `val_hybrid_score`: `val_recon + baseline_penalty_weight × max(0, baseline_gap)`

### Data Processing

- Input data from `merged_vol.xlsx` workbook (sheet: `gan_input_ready`)
- Current surface reshaped to `(1, H, W)` for Transformer sequence processing
- Normalization: current surface / delta / text embedding independently controllable
- Default text embedding mode: `'lp'` (low-dimensional)
- Train/val split chronologically ordered (`train_ratio=0.8`)

### Inference & Scenario Generation (`TransformerWGANSampler`)

1. Load checkpoint → Generator + normalization statistics
2. Generate `mc_samples=64` scenarios per sample:
   - Deterministic seed: `seed + global_index + draw_idx × 1000003`
   - Generator forward → delta → `softplus(current + delta)` reconstruction
3. Compute arbitrage penalties per scenario (calendar + butterfly)
4. Softmax reweighting → weighted mean / quantile (0.05, 0.50, 0.95) aggregation
5. Output: JSON payload + heatmap PNG + term structure/smile plots PNG + summary.csv

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_dim` | 32 | Generator noise dimensionality |
| `model_dim` | 128 | Transformer hidden dimension |
| `gen_layers` | 4 | Generator Transformer layers |
| `disc_layers` | 3 | Critic Transformer layers |
| `num_heads` | 8 | Multi-head attention heads |
| `ffn_dim` | 512 | Feedforward network dimension |
| `dropout` | 0.1 | Dropout rate |
| `text_hidden_dim` | 256 | Text encoder hidden dimension |
| `text_token_dim` | 128 | Text token intermediate dimension |
| `noise_hidden_dim` | 128 | Noise encoder hidden dimension |
| `critic_iter` | 5 | Critic updates per Generator update |
| `pure_adversarial` | False | Whether to disable reconstruction loss |
| `lambda_recon` | 10.0 | Reconstruction loss weight |
| `constraint_warmup_epochs` | 0 | Constraint penalty warmup epochs |
| `best_checkpoint_metric` | `val_recon` | Best checkpoint selection metric |
| `baseline_penalty_weight` | 2.0 | Baseline gap weight in hybrid score |
| `batch_size` | 32 | Batch size |
| `num_epochs` | 100 | Training epochs |
| `learning_rate` | 1e-4 | Learning rate (Adam, β₁=0.5, β₂=0.9) |

### CLI

```bash
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp.yaml
python scripts/transformer_wgan/main.py train --config configs/transformer_wgan/train_lp_pure_adv.yaml
python scripts/transformer_wgan/main.py generate-result --config configs/transformer_wgan/train_lp.yaml
bash run_transformer_wgan_svi_excel.sh configs/transformer_wgan/train_lp.yaml
```

### Checkpoint Structure

Saved contents: training config, surface shape, grid coordinates, normalization statistics, Generator and Critic state_dicts.

Output files: `transformer_wgan_best.pt`, `transformer_wgan_final.pt`, `training_metrics.json`, `loss_curves.png`, etc.
