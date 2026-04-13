# Detailed GAN Model Architecture

This document explains the current WGAN-based vol-surface forecasting model in detail, including the internal structure of the generator, the discriminator, and the way both modules interact during training.

It is based on the current implementation in:

- `src/wgan_option/models/generator.py`
- `src/wgan_option/models/discriminator.py`
- `src/wgan_option/models/gan_model.py`

It complements the higher-level overview in:

- `docs/vol_surface_gan_architecture.md`

## 1. Problem Setting

The model is a conditional GAN for forecasting the future option volatility surface.

The mapping is:

\[
(\text{current surface},\ \text{text embedding},\ \text{noise}) \rightarrow \text{future surface}
\]

More precisely:

- the generator predicts a future surface conditioned on the current surface and the news embedding
- the discriminator, used as a WGAN critic, scores whether a candidate future surface looks realistic given the same current surface and text embedding

The current implementation is not an unconditional image generator.
It is a conditional transition model:

\[
(\text{surface}_t,\ \text{text}_t) \rightarrow \text{surface}_{t+h}
\]

## 2. Inputs, Outputs, and Tensor Shapes

### 2.1 Generator inputs

The generator receives:

- `current_surface`: shape `[B, 1, H, W]`
- `text_embedding`: shape `[B, E]`
- `noise`: shape `[B, Z]`

where:

- `B` is batch size
- `H` is the number of maturity bins
- `W` is the number of strike bins
- `E` is the text-embedding dimension
- `Z` is the latent noise dimension

If `noise` is not provided explicitly, the generator samples:

\[
\epsilon \sim \mathcal{N}(0, I)
\]

inside `forward()`.

### 2.2 Generator output

The generator returns:

- `next_surface`: shape `[B, 1, H, W]`

This is interpreted as the predicted future volatility surface.

### 2.3 Discriminator inputs

The discriminator receives:

- `next_surface`: shape `[B, 1, H, W]`
- `current_surface`: shape `[B, 1, H, W]`
- `text_embedding`: shape `[B, E]`

It then scores:

\[
D(\text{candidate future surface} \mid \text{current surface}, \text{text embedding})
\]

### 2.4 Discriminator output

The discriminator returns:

- shape `[B, 1]`

This is not a probability.
In WGAN-GP, it is a critic score.

Higher values mean:

- the candidate future surface looks more like a real sample under the given conditioning information

## 3. Generator Internals

The generator is implemented in `src/wgan_option/models/generator.py`.

Its internal structure has three branches:

1. surface encoder
2. text encoder
3. fusion head

### 3.1 Surface encoder

The current surface is passed through three convolutional layers:

1. `Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)`
2. `Conv2d(32, 64, kernel_size=3, stride=2, padding=1)`
3. `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)`

Each convolution is followed by:

- `LeakyReLU(0.2)`

#### Shape evolution

For an input of shape:

\[
[B, 1, H, W]
\]

the surface encoder produces approximately:

\[
[B, 32, H, W]
\rightarrow
[B, 64, H/2, W/2]
\rightarrow
[B, 128, H/4, W/4]
\]

with exact output sizes determined by the convolution formula used in `_conv2d_out_size(...)`.

#### Default-grid example

With the current common grid size:

- `H = 16`
- `W = 16`

the shape becomes:

\[
[B, 1, 16, 16]
\rightarrow
[B, 32, 16, 16]
\rightarrow
[B, 64, 8, 8]
\rightarrow
[B, 128, 4, 4]
\]

After flattening:

\[
128 \times 4 \times 4 = 2048
\]

So under the default 16x16 grid, the generator surface feature vector has dimension `2048`.

### 3.2 Text encoder

The text branch is a small MLP:

1. `Linear(embedding_dim, 256)`
2. `LayerNorm(256)`
3. `LeakyReLU(0.2)`
4. `Dropout(0.1)`
5. `Linear(256, 128)`
6. `LeakyReLU(0.2)`

So the text embedding is compressed into a 128-dimensional feature vector.

#### Why this matters

This branch does two things:

- reduces a potentially large text embedding into a compact conditioning vector
- normalizes the representation before fusion, which helps when embedding values come from external models

### 3.3 Noise branch

The generator includes a latent noise input of dimension `noise_dim`.

In the current default configuration:

- `noise_dim = 32`

If no noise tensor is passed into `forward()`, the code samples:

\[
\epsilon \in \mathbb{R}^{B \times Z}
\]

from a standard Gaussian.

This means the generator is technically stochastic.
However, because the training objective also contains a strong reconstruction loss, the model is encouraged to stay close to the true future surface, so the practical amount of stochastic diversity may be limited.

### 3.4 Fusion head

The three feature sources are concatenated:

- encoded surface features
- encoded text features
- latent noise

So the fusion input dimension is:

\[
\text{surface\_feat\_dim} + 128 + \text{noise\_dim}
\]

Under the default 16x16 setup:

\[
2048 + 128 + 32 = 2208
\]

The fusion MLP is:

1. `Linear(fusion_dim, hidden_dim)`
2. `LeakyReLU(0.2)`
3. `Linear(hidden_dim, hidden_dim)`
4. `LeakyReLU(0.2)`
5. `Linear(hidden_dim, H * W)`

With the default config:

- `hidden_dim = 512`

So the generator ultimately maps the fused latent state into `H * W` scalar outputs, then reshapes them into a surface delta of shape `[B, 1, H, W]`.

### 3.5 Residual output design

The final generator output is computed as:

\[
\text{next\_surface} = \text{softplus}(\text{current\_surface} + \Delta) + 10^{-4}
\]

where:

- `Delta` is the surface delta produced by the fusion MLP

This design has two important consequences.

#### Residual forecasting

The generator does not predict the future surface from scratch.
Instead, it predicts a change relative to the current surface.

That is often a better inductive bias for forecasting, because current and future surfaces are usually strongly related.

#### Positivity enforcement

`softplus(...) + 1e-4` guarantees strictly positive outputs.

That is important because volatility values should not be negative, and some downstream penalty calculations assume a positive surface.

## 4. Discriminator Internals

The discriminator is implemented in `src/wgan_option/models/discriminator.py`.

It is a conditional critic rather than a sigmoid classifier.

Its internal structure also has three parts:

1. stacked surface encoder
2. text encoder
3. classifier head

### 4.1 Surface stacking

The discriminator first concatenates the current and candidate future surfaces along the channel axis:

\[
\text{stacked} = \text{concat}(\text{current\_surface},\ \text{next\_surface})
\]

So the input shape becomes:

\[
[B, 2, H, W]
\]

This is a key conditional-design choice.
Instead of judging the future surface alone, the critic judges the pair:

- what the surface looks like now
- what the proposed next surface looks like

### 4.2 Surface encoder

The stacked surface tensor is passed through:

1. `Conv2d(channels * 2, 32, kernel_size=3, stride=2, padding=1)`
2. `LeakyReLU(0.2)`
3. `Conv2d(32, 64, kernel_size=3, stride=2, padding=1)`
4. `InstanceNorm2d(64, affine=True)`
5. `LeakyReLU(0.2)`
6. `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)`
7. `InstanceNorm2d(128, affine=True)`
8. `LeakyReLU(0.2)`

#### Shape evolution

For input `[B, 2, H, W]`, the feature map becomes approximately:

\[
[B, 32, H/2, W/2]
\rightarrow
[B, 64, H/4, W/4]
\rightarrow
[B, 128, H/8, W/8]
\]

#### Default-grid example

For the common 16x16 grid:

\[
[B, 2, 16, 16]
\rightarrow
[B, 32, 8, 8]
\rightarrow
[B, 64, 4, 4]
\rightarrow
[B, 128, 2, 2]
\]

After flattening:

\[
128 \times 2 \times 2 = 512
\]

So the default discriminator surface feature dimension is `512`.

### 4.3 Why InstanceNorm is used here

The discriminator uses `InstanceNorm2d` on the deeper convolutional blocks.

Practically, this helps:

- stabilize feature scales in the critic
- keep the critic less sensitive to absolute activation magnitude shifts

This is a common stabilization pattern in adversarial models, although exact normalization choices always involve trade-offs.

### 4.4 Text encoder

The discriminator text branch is simpler than the generator text branch:

1. `Linear(embedding_dim, 128)`
2. `LeakyReLU(0.2)`

So the critic also uses a 128-dimensional text conditioning vector.

### 4.5 Classifier head

The flattened surface features and text features are concatenated:

\[
\text{combined} = \text{concat}(\text{surface\_features},\ \text{text\_features})
\]

The head is:

1. `Linear(surface_feat_dim + 128, hidden_dim)`
2. `LeakyReLU(0.2)`
3. `Linear(hidden_dim, 1)`

With the default config:

- `hidden_dim = 256`

Under the default 16x16 grid, the combined input dimension is:

\[
512 + 128 = 640
\]

The output is one scalar critic score per sample.

## 5. Generator and Discriminator Together

The two modules interact through the `WGAN_GP` wrapper in `src/wgan_option/models/gan_model.py`.

### 5.1 Generator usage

Given:

- `current_surface`
- `text_embedding`

the training loop computes:

\[
\text{fake\_future} = G(\text{current\_surface},\ \text{text\_embedding})
\]

The fake future surface is then:

- compared with the true future surface through reconstruction and constraint losses
- fed into the discriminator to produce adversarial feedback

### 5.2 Discriminator usage

The discriminator sees both:

\[
D(\text{real\_future},\ \text{current\_surface},\ \text{text\_embedding})
\]

and:

\[
D(\text{fake\_future},\ \text{current\_surface},\ \text{text\_embedding})
\]

So the critic learns to judge whether the future surface is plausible under the same conditioning context.

## 6. Training Loop Mechanics

### 6.1 Optimizers

The code creates two Adam optimizers:

- one for the generator
- one for the discriminator

Both use:

- `learning_rate`
- `beta_1`
- `beta_2`

from the config.

### 6.2 Critic update schedule

Inside each batch, the discriminator is updated `critic_iter` times, then the generator is updated once.

So the per-batch structure is:

1. repeat discriminator step `critic_iter` times
2. run one generator step

This is standard WGAN practice and gives the critic more chances to approximate the Wasserstein geometry before updating the generator.

### 6.3 Generator loss

The generator step computes:

\[
\mathcal{L}_G
=
\mathcal{L}_{adv}
+ \lambda_{recon}\mathcal{L}_{recon}
+ \lambda_{cal}\mathcal{L}_{calendar}
+ \lambda_{bfly}\mathcal{L}_{butterfly}
+ \lambda_{smooth}\mathcal{L}_{smooth}
\]

where:

\[
\mathcal{L}_{adv} = -\mathbb{E}[D(fake)]
\]

\[
\mathcal{L}_{recon} = L1(fake,\ real)
\]

and the calendar, butterfly, and smoothness terms are optional according to config flags.

### 6.4 Discriminator loss

The critic step computes:

\[
\mathcal{L}_D = \mathbb{E}[D(fake)] - \mathbb{E}[D(real)] + \mathcal{L}_{gp}
\]

where:

\[
\mathcal{L}_{gp}
=
\lambda_{gp}
\cdot
\mathbb{E}\left[
\left(
\|\nabla_{\hat{x}}D(\hat{x})\|_2 - 1
\right)^2
\right]
\]

and:

\[
\hat{x} = \alpha \cdot real + (1-\alpha) \cdot fake
\]

with random interpolation coefficient `alpha`.

### 6.5 Validation phase

Validation uses only the generator.

For each validation batch, the code computes:

- `val_recon`
- `val_calendar`
- `val_butterfly`

No discriminator validation loss is used for model selection.

That is an important design choice:

- forecast quality is tracked by validation reconstruction and structural penalties
- adversarial losses are treated as training-dynamics diagnostics rather than the final selection rule

## 7. Constraint Modules

The current GAN wrapper includes three structural penalties on generated surfaces.

### 7.1 Calendar arbitrage penalty

The code computes total variance:

\[
w(\tau, k) = \sigma(\tau, k)^2 \tau
\]

and penalizes:

\[
\text{ReLU}(-(w_{\tau+1} - w_{\tau}))
\]

This discourages violations of maturity-direction monotonicity in total variance.

### 7.2 Butterfly arbitrage penalty

The code converts implied volatility into Black call prices and applies a second finite difference across strike:

\[
C_{i+1} - 2C_i + C_{i-1}
\]

Negative values are penalized through `ReLU(-second_diff)`.

This encourages convexity of call price with respect to strike.

### 7.3 Smoothness penalty

The code adds squared first differences along:

- maturity direction
- strike direction

This discourages locally jagged or noisy predicted surfaces.

## 8. Metrics and Selection Logic

At epoch level, the training loop records metrics such as:

- `g_total`
- `g_adv`
- `g_recon`
- `g_calendar`
- `g_butterfly`
- `g_smooth`
- `d_total`
- `d_real`
- `d_fake`
- `gp`
- `val_recon`
- `val_calendar`
- `val_butterfly`
- `g_lr`
- `d_lr`

The primary model-selection metric is:

- `val_recon`

When validation is available, the code can also:

- save best checkpoints
- apply early stopping
- apply `ReduceLROnPlateau`

All three currently monitor `val_recon`.

## 9. Architectural Rationale

The current model design reflects several practical choices.

### 9.1 Why use a residual generator

Forecasting future surfaces is often easier as:

\[
\text{future surface} = \text{current surface} + \text{change}
\]

than as unconditional decoding from text alone.

The generator therefore learns a delta, not a brand-new surface from scratch.

### 9.2 Why combine adversarial and reconstruction losses

Pure adversarial training might produce visually plausible surfaces that are not close to the actual future target.
Pure regression might minimize average error but produce overly smooth, less realistic shapes.

The current model combines both objectives:

- adversarial realism
- pointwise forecasting accuracy
- structural surface regularity

### 9.3 Why condition the discriminator on the current surface

A future surface should not be judged in isolation.
It should be judged relative to:

- where the surface started
- what text information is available

That is why the critic sees both current and candidate future surfaces together.

### 9.4 Why this is not a U-Net or decoder-heavy generator

The current grid sizes are small enough that a compact CNN encoder plus MLP fusion head is practical.
The model is therefore simpler than many image-to-image GANs:

- no transpose-convolution decoder
- no skip connections
- no attention blocks

This keeps the architecture easier to reason about while still supporting conditional surface forecasting.

## 10. Limitations of the Current Design

The architecture is workable and research-usable, but it has several limitations.

### 10.1 Noise may be partially suppressed

Because reconstruction loss is strong, the model may use the latent noise only weakly.
So even though the generator is stochastic in form, it may behave almost deterministically in practice.

### 10.2 Global MLP output head

The generator decodes the final surface through a fully connected head to `H * W`.
That is simple and effective for small grids, but it may scale less naturally to larger surfaces than a fully convolutional decoder.

### 10.3 Critic only sees static one-step transitions

The current critic evaluates one current/future pair at a time.
It does not model multi-step surface trajectories or temporal consistency across longer windows.

### 10.4 Constraint penalties are soft, not hard

The model encourages no-arbitrage-like behavior through penalties, but it does not guarantee strict arbitrage freedom.

## 11. Practical Reading Guide

If you are reading the code from top to bottom, the most useful order is:

1. `src/wgan_option/models/generator.py`
2. `src/wgan_option/models/discriminator.py`
3. `src/wgan_option/models/gan_model.py`
4. `src/wgan_option/train_vol_xlsx.py`
5. `src/wgan_option/utils/merged_xlsx.py`

If you are reading training outputs, start with:

1. `val_recon`
2. `val_calendar`
3. `val_butterfly`
4. `g_recon`
5. `gp`

and treat `g_total`, `d_total`, `d_real`, and `d_fake` mainly as training-dynamics diagnostics.
