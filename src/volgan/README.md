# VolGAN — MLP-Based Conditional GAN for Volatility Surface Generation

## Directory Structure

```
volgan/
├── __init__.py          # Public API exports
├── config.py            # Configuration dataclasses and YAML loaders
├── models.py            # Neural network architectures (Generator, Discriminator)
├── losses.py            # Loss functions (BCE loss, smoothness penalties, gradient matching)
├── data.py              # Data loading, normalization, Dataset definition
├── trainer.py           # Training loop and checkpoint management
├── inference.py         # Scenario sampling, arbitrage reweighting, result generation
├── arbitrage.py         # Arbitrage penalty computation and scenario reweighting
├── io.py                # Checkpoint / JSON / CSV I/O utilities
├── plotting.py          # Volatility surface visualization
└── training_plots.py    # Training curve visualization
```

## Model Architecture

This module uses an **MLP + standard GAN (BCE loss)** architecture to conditionally generate future volatility surface log-IV increments. Unlike CNN WGAN and Transformer WGAN, VolGAN uses binary cross-entropy (BCE) adversarial training rather than Wasserstein distance.

### Generator (`VolGANGenerator`)

Concatenates the current surface, text embedding, and noise into a single vector, and passes it through an MLP to produce the delta.

```
current_surface_flat ──┐
text_embedding ────────┼─ concat ──→ MLP ──→ delta (surface_dim)
noise ─────────────────┘
```

**Network structure:**
- `Linear(surface_dim + embedding_dim + noise_dim, hidden_dim)` → Softplus
- `Linear(hidden_dim, hidden_dim × 2)` → Softplus
- `Linear(hidden_dim × 2, surface_dim)`

Uses **Softplus** activation (rather than LeakyReLU), providing smoother gradient characteristics.

### Discriminator (`VolGANDiscriminator`)

Determines whether a candidate delta under the given conditions comes from the real data distribution, outputting a probability ∈ [0, 1].

```
current_surface_flat ──┐
text_embedding ────────┼─ concat ──→ MLP ──→ probability ∈ [0, 1]
candidate_delta ───────┘
```

**Network structure:**
- `Linear(surface_dim × 2 + embedding_dim, hidden_dim)` → Softplus
- `Linear(hidden_dim, 1)` → **Sigmoid**

Note: Input dimension is `surface_dim × 2` because both the current surface and delta are `surface_dim`-dimensional.

### Surface Reconstruction

```python
future_surface = exp(log(clamp(current, min=1e-4)) + delta)
```

Shares the same log-IV space incremental reconstruction as CNN WGAN.

## Implementation Details

### Training Pipeline (`VolGANTrainer`)

Per epoch:

1. **Discriminator update** (repeated `disc_steps_per_batch=2` times per batch):
   - Score real deltas → target label `real_label_value=0.9` (label smoothing)
   - Score fake deltas → target label `fake_label_value=0.0`
   - Loss: `0.5 × (BCE(real, 0.9) + BCE(fake, 0.0))`
   - Gradient clipping: `discriminator_grad_clip=5.0`

2. **Generator update** (repeated `gen_steps_per_batch=1` times per batch):
   - Adversarial loss: `BCE(fake_scores, target=1.0)`
   - Strike smoothness: `α_m × strike_smoothness_penalty`
   - Maturity smoothness: `α_τ × maturity_smoothness_penalty`
   - Gradient clipping: `generator_grad_clip=5.0`

### Loss Functions

| Loss | Formula | Description |
|------|---------|-------------|
| Generator BCE | `BCE(fake_scores, 1.0)` | Fool the Discriminator |
| Discriminator BCE | `0.5 × (BCE(real, 0.9) + BCE(fake, 0.0))` | Label-smoothed classification |
| Strike Smoothness | `Σ (1/ΔK²) × (V[k+1] - V[k])²` | Weighted second differences |
| Maturity Smoothness | `Σ (1/Δτ²) × (V[τ+1] - V[τ])²` | Weighted second differences |

### Gradient Matching

A distinctive feature of VolGAN is **automatic balancing** of adversarial and smoothness penalty weights via gradient matching:

1. Run `gradient_match_epochs=5` pre-computation epochs before training
2. For each batch, compute gradient norms of BCE loss and smoothness loss w.r.t. Generator parameters separately
3. Compute ratio: `α = ||∇BCE|| / ||∇smooth||`
4. Use the median across batches as the final weight, clipped to `[alpha_clip_min, alpha_clip_max] = [1e-3, 10.0]`

This ensures the smoothness penalties are balanced with the adversarial loss in gradient magnitude.

### Data Processing

- Input data from `merged_vol.xlsx` workbook (sheet: `gan_input_ready`)
- Surfaces flattened to 1D vectors (MLP input)
- Normalization strategy consistent with CNN WGAN: log space, training-set statistics
- Default text embedding mode: `'hd'` (high-dimensional)
- Train/val split chronologically ordered (`train_ratio=0.8`)

### Inference & Scenario Generation

Inference follows the same pipeline as CNN WGAN:
1. Load checkpoint → generate multiple scenarios → compute arbitrage penalties → softmax reweighting → aggregate

**Arbitrage penalties (applied in post-processing, not in training loss):**
- Calendar arbitrage: checks that call prices increase with maturity
- Butterfly arbitrage: checks convexity in the strike dimension
- Penalties computed via Black-Scholes relative call prices

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_dim` | 32 | Generator noise dimensionality |
| `hidden_dim` | 128 | MLP hidden dimension |
| `disc_steps_per_batch` | 2 | Discriminator updates per Generator update |
| `gen_steps_per_batch` | 1 | Generator updates per batch |
| `real_label_value` | 0.9 | Real label value (label smoothing) |
| `use_gradient_matching` | True | Whether to enable gradient matching |
| `gradient_match_epochs` | 5 | Gradient matching pre-computation epochs |
| `alpha_m` | 1.0 | Strike smoothness weight (initial value before matching) |
| `alpha_tau` | 1.0 | Maturity smoothness weight (initial value before matching) |
| `generator_grad_clip` | 5.0 | Generator gradient clipping threshold |
| `discriminator_grad_clip` | 5.0 | Discriminator gradient clipping threshold |
| `batch_size` | 32 | Batch size |
| `num_epochs` | 100 | Training epochs |
| `learning_rate` | 1e-4 | Learning rate |
| `checkpoint_metric` | `val_mae_gap_vs_current` | Best checkpoint selection metric |

### Comparison with CNN WGAN / Transformer WGAN

| Feature | VolGAN | CNN WGAN | Transformer WGAN |
|---------|--------|----------|------------------|
| Network type | MLP | CNN | Transformer |
| Adversarial loss | BCE | WGAN-GP | WGAN-GP |
| Lipschitz constraint | None | Gradient penalty | Gradient penalty |
| Penalty weighting | Auto (gradient matching) | Manual | Manual |
| Arbitrage constraints | Post-processing only | Training + post-processing | Training + post-processing |
| Label smoothing | Yes (0.9/0.0) | No | No |
| Reconstruction loss | No | No | Optional (L1) |

### CLI

```bash
python scripts/volgan/main.py train --config configs/volgan/train_default.yaml
python scripts/volgan/main.py sample --config configs/volgan/sample_default.yaml
```

### Python API

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

### Checkpoint Structure

Saved contents: training config, `alpha_m`/`alpha_tau`, surface shape, grid coordinates, normalization statistics, Generator and Discriminator state_dicts.

Output files: `volgan_best.pt`, `volgan_final.pt`, `training_metrics.json`, `loss_curves.png`, etc.
