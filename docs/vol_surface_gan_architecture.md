# Bond Option Vol Surface Forecasting with Text Embeddings and Arbitrage-Aware Conditional WGAN

## 1. Research Objective
The target mapping is:

\[
(\text{Surface}_t, \text{TextEmbedding}_t) \rightarrow \text{Surface}_{t+h}
\]

where `Surface_t` is the volatility surface at time `t` (in the current implementation, a vol-like proxy built from trade data), `TextEmbedding_t` is the daily news embedding (default: `HD_embedding`), and `h` is the forecast horizon (default: 1 day).

## 2. Literature Basis (including `ref/`)

### 2.1 Local papers in `ref/`
1. **Horvath, Muguruza, Tomas (2021)**, *Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models*, Quantitative Finance, 21(1), 11-27.  
   Link: https://doi.org/10.1080/14697688.2020.1817974
   - Key takeaway: model the full surface as a network output to learn a high-dimensional map offline and infer quickly online.
2. **Ge et al. (2025)**, *GAN-Enhanced Implied Volatility Surface Reconstruction for Option Pricing Error Mitigation*, IEEE Access.  
   Link: https://doi.org/10.1109/ACCESS.2025.3619553
   - Key takeaway: GAN framework with explicit calendar/butterfly constraints and composite objective design.

### 2.2 Additional related references
1. **Ackerer, Tagasovska, Vatter (2020)**, *Deep Smoothing of the Implied Volatility Surface*.  
   Link: https://arxiv.org/abs/1906.05065
2. **Cao, Liu, Zhai (2021)**, *Option valuation under no-arbitrage constraints with neural networks*, European Journal of Operational Research.  
   Link: https://doi.org/10.1016/j.ejor.2020.06.006
3. **Cohen, Reisinger, Wang (2021)**, *Arbitrage-free neural-SDE market models*, Applied Mathematical Finance.  
   Link: https://doi.org/10.1080/1350486X.2021.1910659
4. **Cuchiero et al. (2020)**, *A Generative Adversarial Network Approach to Calibration of Local Stochastic Volatility Models*.  
   Link: https://doi.org/10.3390/risks8040101
5. **Liu, Tang, Zhou (2024)**, *Applying text embedding for forecasting realized volatility in stock markets*.  
   Link: https://doi.org/10.1016/j.heliyon.2024.e26303
6. **Engle, Giglio, Kelly, Lee, Stroebel (2019)**, *A New Look at the News-Implied Volatility Index*.  
   Link: https://www.nber.org/papers/w26366

These references jointly support the current design:
- deep models for high-dimensional surface mapping;
- explicit no-arbitrage constraints in the loss function;
- text-based factors (embeddings) for volatility-related prediction tasks.

### 2.3 Local Volatility + Machine Learning (Surface Construction and Forecasting)

Local volatility (Dupire) is a pointwise transform of an arbitrage-free call price surface. In practice it is extremely sensitive to noise, missing quotes, and small violations of calendar/butterfly constraints, so ML methods usually combine (i) arbitrage-aware surface learning and (ii) additional regularization/PDE consistency to obtain a stable local-vol surface.

#### 2.3.1 Dupire-Consistent Local Volatility from ML
1. **Chataigner, Crépey, Dixon (2020)**, *Deep Local Volatility*, Risks / arXiv:2007.10462.  
   Link: https://doi.org/10.3390/risks8030082 (journal), https://arxiv.org/abs/2007.10462 (preprint)  
   - Summary: fits an option price surface with a neural network and explicitly enforces no-arbitrage (hard or soft) while using the Dupire formula to constrain/regularize the implied local volatility.
   - Practical takeaway: a Dupire-based regularizer (bounds + smoothness of local vol) is a natural complement to calendar/butterfly penalties in a GAN loss.

2. **Chataigner, Cousin, Crépey, Dixon, Gueye (SIAM J. Financial Math. 2021 / arXiv 2022)**, *Beyond Surrogate Modeling: Learning the Local Volatility Via Shape Constraints*, arXiv:2212.09957.  
   Link: https://arxiv.org/abs/2212.09957  
   - Summary: compares (a) a shape-constrained GP approach on prices (provably arbitrage-free) and (b) a neural-net approach with arbitrage penalties on implied vol; both jointly yield a local-vol surface and are benchmarked against SSVI.
   - Practical takeaway: a two-stage pipeline (arbitrage-free surface completion -> forecasting) can be competitive and reduces instability when later converting IV -> local vol.

3. **Wang, Shaa, Privault, Guet (arXiv 2021, revised 2025)**, *Deep self-consistent learning of local volatility*, arXiv:2201.07880.  
   Link: https://arxiv.org/abs/2201.07880  
   - Summary: parameterizes both option prices and local volatility with neural networks and enforces self-consistency via the Dupire PDE residual sampled over a continuous strike-maturity domain, with additional soft no-arbitrage penalties.
   - Practical takeaway: “PDE-residual-as-loss” provides a differentiable constraint alternative to pure finite-difference arbitrage penalties.

4. **Bae, Kang, Lee (2024)**, *Option Pricing and Local Volatility Surface by Physics-Informed Neural Network*, Computational Economics.  
   Link: https://doi.org/10.1007/s10614-024-10551-2  
   - Summary: uses a physics-informed neural network (PINN) to approximate option prices/Greeks and then constructs a local-vol surface via Dupire from the network outputs.
   - Practical takeaway: PINN-style derivative losses are relevant if we later want the GAN to output a differentiable surface that supports stable derivative-based constraints.

5. **Hakala (2019)**, *Applied Machine Learning for Stochastic Local Volatility Calibration*, Frontiers in Artificial Intelligence.  
   Link: https://doi.org/10.3389/frai.2019.00004  
   - Summary: in local stochastic volatility (LSV) calibration, the key bottleneck is estimating conditional variance given spot; the paper reframes this as supervised regression and proposes radial-basis-function estimators to improve efficiency/stability.
   - Practical takeaway: for interest-rate options (stochastic rates + vol), “local vol” often lives inside an LSV-type model, and ML can accelerate calibration loops.

#### 2.3.2 ML for Vol Surface Dynamics (Forecasting / Simulation)
1. **Chen, Zhang (2019)**, *Forecasting Implied Volatility Smile Surface via Deep Learning and Attention Mechanism*, arXiv:1912.11059.  
   Link: https://arxiv.org/abs/1912.11059  
   - Summary: attention-enhanced LSTM is used to forecast future implied-vol surfaces; the paper evaluates the predicted surfaces via downstream strategy performance.

2. **Choudhary, Jaimungal, Bergeron (2023)**, *FuNVol: A Multi-Asset Implied Volatility Market Simulator using Functional Principal Components and Neural SDEs*, arXiv:2303.00859.  
   Link: https://arxiv.org/abs/2303.00859  
   - Summary: combines functional PCA (to reduce surface dimension) with neural SDEs to generate sequences of implied-vol surfaces (and underlying prices) that match historical stylized facts and remain close to the static-arbitrage-free manifold.

3. **Ning, Jaimungal, Zhang, Bergeron (2022)**, *Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders*, arXiv:2108.04941.  
   Link: https://arxiv.org/abs/2108.04941  
   - Summary: a hybrid VAE + SDE-parameterization approach to sample arbitrage-free implied-vol surfaces, including conditional generation; useful as a scenario generator or as a strong prior on surface geometry.

4. **Hoshisashi, Phelan, Barucca (2024)**, *Whack-a-mole Online Learning: Physics-Informed Neural Network for Intraday Implied Volatility Surface*, arXiv:2411.02375.  
   Link: https://arxiv.org/abs/2411.02375  
   - Summary: proposes a PINN-style intraday calibration method with multiple objectives (fit + PDE + no-arbitrage inequalities) and adaptive loss reweighting for stable real-time updates.
   - Practical takeaway: adaptive balancing of multiple constraints is directly relevant when mixing adversarial loss, reconstruction, and several arbitrage penalties.

#### 2.3.3 Interest-Rate Vol Surfaces (SABR/LSV) + ML Surrogates
1. **Abatcheva, Dankwart, Renk, Ewald (2021)**, *Fast calibration of the SABR model using neural networks*, Neural Computing and Applications.  
   Link: https://doi.org/10.1007/s00521-021-06602-x  
   - Summary: trains neural networks to approximate the inverse calibration map from implied-vol smiles to SABR parameters, yielding substantial speed-ups versus iterative calibration.
   - Practical takeaway: for US rates options where SABR is a common benchmark, “forecast surface -> calibrate SABR -> derive local vol (if needed)” is a pragmatic hybrid baseline.

2. **Balzani, Witte (2020)**, *Deep neural network calibration of the SABR model*, arXiv:2011.13610.  
   Link: https://arxiv.org/abs/2011.13610  
   - Summary: builds a supervised DNN surrogate for SABR calibration, focusing on accuracy, numerical stability, and training-data generation.
   - Practical takeaway: SABR-parameter dynamics can be a low-dimensional target for conditional generative models, with an explicit mapping back to a full implied-vol surface.

## 3. Data and Feature Engineering (Current Implementation)

Code file: `src/wgan_option/utils/dataloader.py`

### 3.1 Option data processing
- Input: `data/raw/option_data/*.csv.gz`.
- Parse `#RIC` into approximate `strike / expiry month code / year digit` (rule-based parsing for TY-style option symbols).
- Build daily 2D grids:
  - x-axis: `moneyness` (normalized by daily median strike);
  - y-axis: `maturity_days` (approximate days from trade date to expiry date).
- Missing grid values are filled via row/column interpolation and global fallback.

### 3.2 Text embedding processing
- Input: `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx`.
- Streaming XML parser (no `openpyxl` dependency), then daily mean aggregation of `HD_embedding`.
- Date alignment with option surfaces:
  - if a same-day embedding exists: use it;
  - otherwise: use a zero vector fallback.

### 3.3 Supervised sample construction
Construct sample pairs:

\[
X_t = (\text{Surface}_t, \text{Embedding}_t), \quad Y_t = \text{Surface}_{t+h}
\]

## 4. Model Architecture

### 4.1 Generator (conditional)
Code: `src/wgan_option/models/generator.py`

Inputs:
- current surface: `[B, 1, H, W]`
- text embedding: `[B, E]`
- noise: `[B, Z]`

Structure:
1. `surface_encoder`: CNN for local shape features of the surface;
2. `text_encoder`: MLP for semantic text features;
3. fused representation is mapped to a `delta surface`;
4. `softplus(current + delta)` enforces strictly positive output as the predicted future surface.

### 4.2 Discriminator / Critic (conditional)
Code: `src/wgan_option/models/discriminator.py`

Inputs:
- candidate future surface (real or generated)
- current surface
- text embedding

Design:
- concatenate `(current, future)` along channels and encode with CNN;
- fuse with text features and output Wasserstein score.

## 5. Modified Loss: Adversarial + Reconstruction + No-Arbitrage

Code: `src/wgan_option/models/gan_model.py`

Overall objective:

\[
\mathcal{L}_G = \mathcal{L}_{adv}
+ \lambda_{recon}\mathcal{L}_{recon}
+ \lambda_{cal}\mathcal{L}_{calendar}
+ \lambda_{bfly}\mathcal{L}_{butterfly}
+ \lambda_{smooth}\mathcal{L}_{smooth}
\]

### 5.1 Adversarial term (WGAN-GP)
- Critic uses Wasserstein objective;
- gradient penalty enforces the 1-Lipschitz condition.

### 5.2 Reconstruction term
- `L1(fake_surface, real_future_surface)` to preserve predictability, not only realism.

### 5.3 Calendar arbitrage penalty
Constraint:

\[
\partial_\tau (\sigma^2 \tau) \ge 0
\]

Discrete implementation: penalize negative increments of total variance across adjacent maturities via `ReLU(-diff)`.

### 5.4 Butterfly arbitrage penalty
Constraint:

\[
\frac{\partial^2 C}{\partial K^2} \ge 0
\]

Implementation:
1. compute `C(K,\tau)` via Black formula using generated `\sigma(K,\tau)`;
2. apply second-order finite difference along strike;
3. penalize negative parts.

### 5.5 Smoothness penalty
- add first-order difference penalties along maturity and strike to suppress local oscillations.

## 6. Training Procedure

Entry point: `scripts/train.py` -> `src/wgan_option/train.py`

Workflow:
1. load and build dataloaders;
2. initialize conditional WGAN;
3. for each batch, update Critic `n_critic` times and Generator once;
4. periodically save checkpoints;
5. write metrics to `outputs/metrics/training_metrics.csv`.

## 7. Current Data Constraints and Recommendations

In the current sample dataset:
- option data covers only about 4 trading days;
- the earliest news embedding date is later than the option sample period (little to no overlap).

Therefore this implementation is fully runnable for architecture validation, but statistical forecasting performance is limited by sample size. Recommended next steps:
1. extend TY option history coverage;
2. provide news embeddings with stronger date overlap;
3. if underlying spot/rates are available, replace the current vol-like proxy with true implied volatility surfaces.

## 8. Key Code Locations
- config: `src/wgan_option/config.py`
- data pipeline: `src/wgan_option/utils/dataloader.py`
- generator: `src/wgan_option/models/generator.py`
- discriminator: `src/wgan_option/models/discriminator.py`
- training and losses: `src/wgan_option/models/gan_model.py`
