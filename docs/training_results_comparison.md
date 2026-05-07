# Training Results Comparison

This note summarizes the current local `outputs/training` snapshot on branch
`codex/standalone-wgan-suite-film-atm`.

It is based on saved artifacts only:

- `metrics/training_metrics.csv`
- `metrics/training_metrics.json`
- `metrics/best_checkpoint.json`
- `metrics/*resolved_config.yaml`
- `metrics/run_config_*.yaml`
- `run.log`
- `generate_result/*/samples/*.json` for post-hoc CNN short-ATM evaluation

No training jobs were rerun for this review. The CNN short-ATM figures below were computed post-hoc from saved generated-sample JSON artifacts.

## Scope

The current snapshot contains 53 discovered run directories:

| Status | Count |
| --- | ---: |
| `finished` | 30 |
| `partial/interrupted` | 13 |
| `early stopped` | 8 |
| `no metrics` | 2 |

There are 51 runs with `training_metrics.csv` and 2 setup/dry or aborted runs without epoch metrics.

The model-family coverage is:

| Family | Runs |
| --- | ---: |
| `film_wgan` | 18 |
| `cnn_wgan` | 15 |
| `volgan` | 6 |
| `transformer_wgan` | 5 |
| `legacy_wgan` | 5 |
| `stylemod_wgan` | 3 |
| `crossattn_wgan` | 1 |

## Metric Comparability

The runs do not all optimize or checkpoint the same metric, so they should not be collapsed into a single scoreboard without qualification.

- `val_recon` is the older reconstruction metric used by the legacy raw/svi WGAN and transformer-style runs.
- `val_hybrid_score` combines reconstruction with baseline-gap and regularization terms in some older WGAN variants.
- `val_mae_gap_vs_current` is the main paired `svi-excel` metric for judging whether the generated future surface beats the held-out `current_surface` baseline. More negative is better.
- `val_short_atm_mae_gap_vs_current` narrows the comparison to a weighted short-end / near-ATM region.
- `val_atm_short_pure_mae_gap_vs_current` is the later FiLM checkpoint metric focused on the pure short-ATM region.
- CNN short-ATM values in this note are post-hoc metrics from saved `samples/*.json`, not CNN checkpoint-selection metrics and not evidence that CNN was trained with a short-ATM objective.

For thesis-facing comparison, use `val_mae_gap_vs_current` for broad paired-surface ranking, use short-ATM metrics only for the targeted FiLM study or explicitly labeled post-hoc references, and keep legacy `val_recon` / `val_hybrid_score` results separate.

## High-Level Findings

- The strongest broad `svi-excel` results now come from the later short-ATM FiLM experiments. The best `val_mae_gap_vs_current` is `-0.002785` from `outputs/training/film_wgan/svi-excel/20260417_180233` at epoch `18`.
- The older CNN reference `outputs/training/cnn_wgan/svi-excel/20260414_123032` remains the best pre-short-ATM CNN baseline, with `val_mae_gap_vs_current = -0.001999` and `val_mae = 0.023857` at epoch `19`.
- Post-hoc short-ATM evaluation of that same CNN checkpoint shows that it still improves the weighted short-end band, but it does not beat the current-surface baseline on the stricter `film_180233` pure short-ATM mask (`+0.001236` gap). On the wider `film_131244` pure mask, it is effectively tied with persistence (`-0.000007` gap).
- `crossattn_wgan` is stable but not competitive with the best CNN or FiLM results. Its best gap is `-0.000700` at epoch `109`.
- `stylemod_wgan` is trainable after stabilization work, but the tuned run `20260414_233543` remains above the current-surface baseline on its selected checkpoint (`+0.000851`).
- `volgan` provides a simple normalized baseline. Its best gap run reaches `-0.000432`, useful as a lower-capacity reference but materially weaker than CNN/FiLM.
- `transformer_wgan` is a negative control in this snapshot: all measured reconstruction errors are much worse than the paired current-surface baseline.
- Legacy `raw-excel` and root `svi-excel` WGAN runs use different metrics and data semantics, so they are useful for data-lineage context but should not be directly ranked against the later gap-based standalone WGAN suite.

## Top Result Snapshots

Best broad paired-surface gaps:

| Rank | Run | Best `val_mae_gap_vs_current` | `val_mae` | Epoch |
| ---: | --- | ---: | ---: | ---: |
| 1 | `film_wgan/svi-excel/20260417_180233` | `-0.002785` | `0.023071` | 18 |
| 2 | `film_wgan/svi-excel/20260417_180227` | `-0.002778` | `0.023077` | 18 |
| 3 | `film_wgan/svi-excel/20260417_125450` | `-0.002777` | `0.023079` | 18 |
| 4 | `film_wgan/svi-excel/20260417_125442` | `-0.002705` | `0.023150` | 18 |
| 5 | `film_wgan/svi-excel/20260416_124456` | `-0.002588` | `0.023267` | 16 |
| 6 | `film_wgan/svi-excel/20260417_180230` | `-0.002547` | `0.023308` | 14 |
| 7 | `film_wgan/svi-excel/20260417_131244` | `-0.002405` | `0.023450` | 12 |
| 8 | `cnn_wgan/svi-excel/20260414_123032` | `-0.001999` | `0.023857` | 19 |

Best short-ATM checkpoint results:

| Rank | Run | Best `val_atm_short_pure_mae_gap_vs_current` | Overall gap at that epoch | `val_mae` | Epoch |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `film_wgan/svi-excel/20260417_131244` | `-0.001601` | `-0.001516` | `0.024339` | 9 |
| 2 | `film_wgan/svi-excel/20260416_124456` | `-0.001216` | `-0.002034` | `0.023821` | 15 |
| 3 | `film_wgan/svi-excel/20260417_180233` | `-0.001206` | `-0.002220` | `0.023635` | 15 |
| 4 | `film_wgan/svi-excel/20260417_180227` | `-0.001187` | `-0.002182` | `0.023673` | 15 |
| 5 | `film_wgan/svi-excel/20260417_125442` | `-0.001183` | `-0.002098` | `0.023757` | 15 |

Post-hoc CNN short-ATM reference:

| Reference mask | Run | Samples | `short_atm_mae_gap_vs_current` | `atm_short_pure_mae` | `current_atm_short_pure_mae` | `atm_short_pure_mae_gap_vs_current` | Pure win rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `film_180233` mask: pure `0.04/60`, weighted `0.06/150 x6` | `cnn_wgan/svi-excel/20260414_123032` | 743 | `-0.001388` | `0.017253` | `0.016017` | `+0.001236` | `0.342` |
| `film_131244` mask: pure `0.06/90`, weighted `0.06/90 x8` | `cnn_wgan/svi-excel/20260414_123032` | 743 | `-0.001393` | `0.020410` | `0.020416` | `-0.000007` | `0.455` |

The post-hoc CNN artifacts are:

- `outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/short_atm_posthoc_eval/film_180233_mask/summary.json`
- `outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/short_atm_posthoc_eval/film_131244_mask/summary.json`

Legacy reconstruction ranking:

| Rank | Run | Best `val_recon` | Epoch | Note |
| ---: | --- | ---: | ---: | --- |
| 1 | `raw-excel/20260410_215114` | `0.016845` | 102 | Best legacy raw-excel reconstruction run. |
| 2 | `raw-excel/20260410_215318` | `0.016952` | 79 | Close repeat of the best raw-excel setup. |
| 3 | `raw-excel/20260410_224839` | `0.017871` | 97 | Monitored by `val_hybrid_score`; weaker than the current-surface baseline in that run. |
| 4 | `svi-excel/20260410_225145` | `0.024071` | 12 | Root SVI legacy run that beats its `val_current_recon` baseline. |
| 5 | `svi-excel/20260410_194619` | `0.024986` | 12 | Older root SVI legacy run. |

## Model Family Analysis

### `raw-excel` Legacy WGAN

The three `raw-excel` runs are lineage/context runs rather than direct competitors to the standalone paired `svi-excel` experiments. The best reconstruction result is `20260410_215114` with `val_recon = 0.016845`. The hybrid-score run `20260410_224839` records a lower `val_current_recon` baseline (`0.010861`) than its generated `val_recon` (`0.017871`), so it should not be interpreted as an improvement over persistence.

### Root `svi-excel` Legacy WGAN

The two root `svi-excel` runs are partial/interrupted legacy WGAN runs. `20260410_225145` is the useful reference because it records `val_recon = 0.024071` and `val_baseline_gap = -0.002016`, but its metric schema differs from the later standalone CNN/FiLM gap runs.

### `volgan`

The first two `volgan` runs only expose `val_mae`, both reaching `0.026951`. The later gap-aware pair `20260412_172740` and `20260412_182057` both reach `val_mae_gap_vs_current = -0.000432` at epoch `3`, then drift to `+0.005931` by the final epoch. The tuned `20260412_183055` never beats the current-surface baseline.

### `transformer_wgan`

`transformer_wgan` remains a negative control in this snapshot. The best measured `val_recon` is `0.058147`, while the recorded current-surface baselines are around `0.026`. This family is not competitive under the saved configurations.

### `cnn_wgan`

`cnn_wgan` is the strongest pre-short-ATM model family. The best stable thesis-facing CNN run is `20260414_123032`, which reaches `val_mae_gap_vs_current = -0.001999` at epoch `19`. Several earlier CNN runs also briefly beat the baseline, but many are partial/interrupted or peak before later checkpoint-warmup conventions. The family shows the same recurring pattern as the FiLM runs: best validation often arrives early, and long training drifts back toward positive gaps.

For short-ATM comparison, the CNN values here are post-hoc evaluations of the saved `cnn_wgan_best` generated samples. They should be used as a reference baseline only: CNN was trained and checkpointed on broad `val_mae_gap_vs_current`, while the later FiLM runs explicitly add short-ATM metrics, extra checkpoints, and short-ATM loss terms.

### `crossattn_wgan`

The single `crossattn_wgan` run completes all 220 epochs and reaches `val_mae_gap_vs_current = -0.000700`. It is reasonably stable compared with some adversarial variants, but the gain is too small relative to its parameter budget (`G=38.27M`, `C=15.05M`) and it does not close the gap to CNN or later FiLM runs.

### `stylemod_wgan`

`stylemod_wgan` is now trainable, but it is not competitive. The tuned `20260414_233543` run selects a positive checkpoint gap (`+0.000851`) and finishes at `+0.004233`. Earlier StyleMod rows include negative raw epoch gaps before selection, but the official selected checkpoints do not justify keeping this as a main thesis model without a new architectural reason.

### `film_wgan`

The FiLM family has two regimes. The early 220-epoch FiLM run `20260414_233331` improves on the current-surface baseline (`-0.001347`) but drifts to a positive final gap. The later short-ATM FiLM experiments change the objective and checkpointing policy: `lambda_recon` increases to `20.0`, `lambda_adv` is reduced to `0.05` or `0.10`, checkpoint warmup drops to `5`, and early stopping becomes active. These later runs dominate the broad `val_mae_gap_vs_current` ranking as well as the short-ATM metrics, with `20260417_180233` the best overall gap run and `20260417_131244` the best pure short-ATM checkpoint run.

## Reproducing CNN Short-ATM Evaluation

`run_train.sh` wraps `python scripts/train/main.py` and is intended for `vol-xlsx`, `vol-regression-xlsx`, and `svi-xlsx`. It does not dispatch to standalone `cnn_wgan`; use `scripts/cnn_wgan/main.py` for CNN WGAN jobs.

Regenerate the existing CNN best checkpoint samples if needed:

```bash
python scripts/cnn_wgan/main.py generate-result \
  --config configs/cnn_wgan/train_lp_gen128_disc128.yaml \
  --checkpoint outputs/training/cnn_wgan/svi-excel/20260414_123032/checkpoints/cnn_wgan_best.pt \
  --output-dir generate_result/cnn_wgan_best_short_atm_eval \
  --split val \
  --selection-mode all \
  --selection-count 0 \
  --set mc_samples=64 \
  --set reweight_beta=25.0
```

Compute the CNN post-hoc metric using the `film_180233` short-ATM definition:

```bash
python scripts/cnn_wgan/main.py short-atm-eval \
  --samples-dir outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/samples \
  --output-dir outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/short_atm_posthoc_eval/film_180233_mask \
  --label film_180233_mask \
  --atm-short-range 0.04 \
  --atm-short-max-days 60 \
  --recon-atm-range 0.06 \
  --recon-atm-short-end-max-days 150 \
  --recon-atm-multiplier 6
```

Compute the CNN post-hoc metric using the `film_131244` short-ATM definition:

```bash
python scripts/cnn_wgan/main.py short-atm-eval \
  --samples-dir outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/samples \
  --output-dir outputs/training/cnn_wgan/svi-excel/20260414_123032/generate_result/cnn_wgan_best/short_atm_posthoc_eval/film_131244_mask \
  --label film_131244_mask \
  --atm-short-range 0.06 \
  --atm-short-max-days 90 \
  --recon-atm-range 0.06 \
  --recon-atm-short-end-max-days 90 \
  --recon-atm-multiplier 8
```

To rerun the closest CNN training baseline to `20260414_123032`:

```bash
python scripts/cnn_wgan/main.py train \
  --config configs/cnn_wgan/train_lp_gen128_disc128.yaml \
  --set disc_base_channels=112 \
  --set output_root=outputs/training/cnn_wgan/svi-excel
```

For a background training run:

```bash
mkdir -p logs
nohup python scripts/cnn_wgan/main.py train \
  --config configs/cnn_wgan/train_lp_gen128_disc128.yaml \
  --set disc_base_channels=112 \
  --set output_root=outputs/training/cnn_wgan/svi-excel \
  > logs/train_cnn_wgan_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

## Full Run Comparison

| Run | Model | Data | Status | Text | Epochs | Checkpoint metric | Best epoch | Best value | Best comparable validation | Final validation | Core hyperparameters | Params |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| `outputs/training/cnn_wgan/svi-excel/20260412_203136` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 103 | -0.001877 | mae 0.023978, gap -0.001877 @e103 | mae 0.026199, gap 0.000344 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260412_230253` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 2 | -0.001932 | mae 0.023924, gap -0.001932 @e2 | mae 0.024952, gap -0.000903 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=128<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260413_102855` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 28/220 | `val_mae_gap_vs_current` | 5 | -0.001472 | mae 0.024384, gap -0.001472 @e5 | mae 0.027887, gap 0.002032 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=128<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260413_112106` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 5 | -0.001718 | mae 0.024138, gap -0.001718 @e5 | mae 0.025346, gap -0.000509 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=128<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260413_150847` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 2 | -0.001773 | mae 0.024082, gap -0.001773 @e2 | mae 0.026084, gap 0.000229 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=128<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260413_151440` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 119 | -0.001198 | mae 0.024658, gap -0.001198 @e119 | mae 0.025854, gap -0.000001 | batch=128<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5 |  |
| `outputs/training/cnn_wgan/svi-excel/20260413_224626` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 95/220 | `val_mae_gap_vs_current` | 1 | -0.001049 | mae 0.024806, gap -0.001049 @e1 | mae 0.027366, gap 0.001511 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5<br>warmup=10 | G=37.12M, C=13.90M |
| `outputs/training/cnn_wgan/svi-excel/20260413_230516` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 74/220 | `val_mae_gap_vs_current` | 5 | -0.001572 | mae 0.024283, gap -0.001572 @e5 | mae 0.027944, gap 0.002089 | batch=96<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/cnn_wgan/svi-excel/20260413_231523` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 69/220 | `val_mae_gap_vs_current` | 4 | -0.001368 | mae 0.024487, gap -0.001368 @e4 | mae 0.028506, gap 0.002651 | batch=256<br>epochs=220<br>lr=5e-05<br>g_lr=5e-05<br>d_lr=5e-05<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/cnn_wgan/svi-excel/20260413_231654` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 68/220 | `val_mae_gap_vs_current` | 9 | -0.001190 | mae 0.024665, gap -0.001190 @e9 | mae 0.031773, gap 0.005918 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/cnn_wgan/svi-excel/20260413_233328` | `cnn_wgan` | `svi-excel` | partial/interrupted | `lp` | 64/220 | `val_mae_gap_vs_current` | 18 | -0.001828 | mae 0.024027, gap -0.001828 @e18 | mae 0.030352, gap 0.004496 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/cnn_wgan/svi-excel/20260414_123011` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 33 | 0.000198 | mae 0.024757, gap -0.001098 @e9 | mae 0.028664, gap 0.002809 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/cnn_wgan/svi-excel/20260414_123023` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 13 | -0.001309 | mae 0.024547, gap -0.001309 @e13 | mae 0.028934, gap 0.003079 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=37.12M, C=13.90M |
| `outputs/training/cnn_wgan/svi-excel/20260414_123032` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 19 | -0.001999 | mae 0.023857, gap -0.001999 @e19 | mae 0.028350, gap 0.002495 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=37.12M, C=13.90M |
| `outputs/training/cnn_wgan/svi-excel/20260414_123040` | `cnn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 13 | -0.000641 | mae 0.024703, gap -0.001152 @e9 | mae 0.028323, gap 0.002468 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=37.12M, C=16.98M |
| `outputs/training/crossattn_wgan/svi-excel/20260414_230946` | `crossattn_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 109 | -0.000700 | mae 0.025155, gap -0.000700 @e109 | mae 0.026952, gap 0.001097 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=38.27M, C=15.05M |
| `outputs/training/film_wgan/svi-excel/20260414_233331` | `film_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 12 | -0.001347 | mae 0.024508, gap -0.001347 @e12 | mae 0.028375, gap 0.002520 | batch=256<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=128<br>d_ch=128<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=47.83M, C=19.25M |
| `outputs/training/film_wgan/svi-excel/20260415_133750` | `film_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 12 | -0.000439 | mae 0.025416, gap -0.000439 @e12 | mae 0.028427, gap 0.002571 | batch=128<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260415_133827` | `film_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 19 | 0.000069 | mae 0.025123, gap -0.000732 @e5 | mae 0.029108, gap 0.003253 | batch=192<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260415_191342` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_mae_gap_vs_current` | 13 | -0.000447 | mae 0.025408, gap -0.000447 @e13 | mae 0.027948, gap 0.002093 | batch=128<br>epochs=60<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>critic_iter=5<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260415_191349` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_mae_gap_vs_current` | 13 | -0.000764 | mae 0.024084, gap -0.001771 @e2 | mae 0.025929, gap 0.000074 | batch=128<br>epochs=60<br>lr=1e-5<br>g_lr=1e-5<br>d_lr=2e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>critic_iter=3<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260415_191355` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_short_atm_mae_gap_vs_current` | 16 | -0.000710 | mae 0.024080, gap -0.001775 @e2 | mae 0.026373, gap 0.000518 | batch=128<br>epochs=60<br>lr=1e-5<br>g_lr=1e-5<br>d_lr=2e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>critic_iter=3<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260415_191359` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_short_atm_mae_gap_vs_current` | 19 | -0.001115 | mae 0.024074, gap -0.001782 @e19 | mae 0.027142, gap 0.001287 | batch=128<br>epochs=60<br>lr=1e-5<br>g_lr=1e-5<br>d_lr=2e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>critic_iter=3<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260416_124454` | `film_wgan` | `svi-excel` | early stopped | `lp` | 11/25 | `val_atm_short_pure_mae_gap_vs_current` | 6 | 0.000145 | mae 0.025237, gap -0.000618 @e7 | mae 0.026954, gap 0.001098 | batch=128<br>epochs=25<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=1.0<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260416_124456` | `film_wgan` | `svi-excel` | early stopped | `lp` | 20/25 | `val_atm_short_pure_mae_gap_vs_current` | 15 | -0.001216 | mae 0.023267, gap -0.002588 @e16 | mae 0.023448, gap -0.002407 | batch=128<br>epochs=25<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.05<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260416_124708` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_short_atm_mae_gap_vs_current` | 14 | -0.001172 | mae 0.024099, gap -0.001756 @e14 | mae 0.028890, gap 0.003034 | batch=128<br>epochs=60<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=1.0<br>critic_iter=5<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260416_124711` | `film_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_short_atm_mae_gap_vs_current` | 17 | -0.000020 | mae 0.025378, gap -0.000477 @e17 | mae 0.029053, gap 0.003198 | batch=128<br>epochs=60<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=1.0<br>critic_iter=5<br>warmup=8 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260416_124714` | `film_wgan` | `svi-excel` | finished | `lp` | 25/25 | `val_short_atm_mae_gap_vs_current` | 12 | 0.000690 | mae 0.025742, gap -0.000113 @e4 | mae 0.027398, gap 0.001543 | batch=128<br>epochs=25<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=1.0<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_125442` | `film_wgan` | `svi-excel` | early stopped | `lp` | 22/30 | `val_atm_short_pure_mae_gap_vs_current` | 15 | -0.001183 | mae 0.023150, gap -0.002705 @e18 | mae 0.023753, gap -0.002102 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.05<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_125450` | `film_wgan` | `svi-excel` | early stopped | `lp` | 22/30 | `val_atm_short_pure_mae_gap_vs_current` | 15 | -0.001118 | mae 0.023079, gap -0.002777 @e18 | mae 0.023756, gap -0.002099 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.05<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_131244` | `film_wgan` | `svi-excel` | early stopped | `lp` | 16/30 | `val_atm_short_pure_mae_gap_vs_current` | 9 | -0.001601 | mae 0.023450, gap -0.002405 @e12 | mae 0.024446, gap -0.001410 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.1<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_180227` | `film_wgan` | `svi-excel` | early stopped | `lp` | 22/30 | `val_atm_short_pure_mae_gap_vs_current` | 15 | -0.001187 | mae 0.023077, gap -0.002778 @e18 | mae 0.023770, gap -0.002085 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.05<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_180230` | `film_wgan` | `svi-excel` | early stopped | `lp` | 16/30 | `val_atm_short_pure_mae_gap_vs_current` | 9 | -0.001073 | mae 0.023308, gap -0.002547 @e14 | mae 0.024729, gap -0.001126 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.1<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/film_wgan/svi-excel/20260417_180233` | `film_wgan` | `svi-excel` | early stopped | `lp` | 22/30 | `val_atm_short_pure_mae_gap_vs_current` | 15 | -0.001206 | mae 0.023071, gap -0.002785 @e18 | mae 0.023676, gap -0.002179 | batch=128<br>epochs=30<br>lr=2e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=20.0<br>lambda_adv=0.05<br>critic_iter=5<br>warmup=5 | G=40.48M, C=15.89M |
| `outputs/training/raw-excel/20260410_215114` | `legacy_wgan` | `raw-excel` | partial/interrupted | `lp` | 122/200 | `val_recon` | 102 | 0.016845 | recon 0.016845 @e102 | recon 0.017475 | batch=128<br>epochs=200<br>lr=6e-05<br>g_ch=96<br>d_ch=96<br>g_blocks=4<br>d_blocks=2<br>g_hidden=4096<br>d_hidden=2048<br>lambda_recon=10.0<br>disc_iter=5 |  |
| `outputs/training/raw-excel/20260410_215318` | `legacy_wgan` | `raw-excel` | partial/interrupted | `lp` | 99/200 | `val_recon` | 79 | 0.016952 | recon 0.016952 @e79 | recon 0.017858 | batch=128<br>epochs=200<br>lr=6e-05<br>g_ch=96<br>d_ch=96<br>g_blocks=4<br>d_blocks=2<br>g_hidden=4096<br>d_hidden=2048<br>lambda_recon=10.0<br>disc_iter=5 |  |
| `outputs/training/raw-excel/20260410_224839` | `legacy_wgan` | `raw-excel` | partial/interrupted | `lp` | 117/200 | `val_hybrid_score` | 97 | 0.031891 | recon 0.017871 @e97 | recon 0.018271 | batch=128<br>epochs=200<br>lr=6e-05<br>g_ch=96<br>d_ch=96<br>g_blocks=4<br>d_blocks=2<br>g_hidden=4096<br>d_hidden=2048<br>lambda_recon=10.0<br>disc_iter=5 |  |
| `outputs/training/stylemod_wgan/svi-excel/20260414_121456` | `stylemod_wgan` | `svi-excel` | finished | `lp` | 220/220 | `val_mae_gap_vs_current` | 12 | -0.000365 | mae 0.024689, gap -0.001166 @e10 | mae 0.028150, gap 0.002295 | batch=128<br>epochs=220<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=4<br>d_blocks=2<br>lambda_recon=2.0<br>critic_iter=5<br>warmup=10 | G=44.69M, C=15.89M |
| `outputs/training/stylemod_wgan/svi-excel/20260414_223509` | `stylemod_wgan` | `svi-excel` | finished | `lp` | 60/60 | `val_mae_gap_vs_current` | 34 | 0.001072 | mae 0.024733, gap -0.001122 @e10 | mae 0.029534, gap 0.003679 | batch=128<br>epochs=60<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=64<br>d_ch=64<br>g_blocks=2<br>d_blocks=2<br>lambda_recon=0.5<br>critic_iter=3<br>warmup=15 | G=13.41M, C=6.39M |
| `outputs/training/stylemod_wgan/svi-excel/20260414_233543` | `stylemod_wgan` | `svi-excel` | finished | `lp` | 120/120 | `val_mae_gap_vs_current` | 40 | 0.000851 | mae 0.026706, gap 0.000851 @e40 | mae 0.030088, gap 0.004233 | batch=128<br>epochs=120<br>lr=1e-5<br>g_lr=2e-5<br>d_lr=1e-5<br>g_ch=112<br>d_ch=112<br>g_blocks=2<br>d_blocks=2<br>lambda_recon=0.5<br>critic_iter=3<br>warmup=15 | G=23.37M, C=13.66M |
| `outputs/training/svi-excel/20260410_194619` | `legacy_wgan` | `svi-excel` | partial/interrupted | `lp` | 32/200 | `val_recon` | 12 | 0.024986 | recon 0.024986 @e12 | recon 0.027553 | batch=128<br>epochs=200<br>lr=6e-05<br>g_ch=96<br>d_ch=96<br>g_blocks=4<br>d_blocks=2<br>g_hidden=4096<br>d_hidden=2048<br>lambda_recon=10.0<br>disc_iter=5 |  |
| `outputs/training/svi-excel/20260410_225145` | `legacy_wgan` | `svi-excel` | partial/interrupted | `lp` | 32/200 | `val_hybrid_score` | 12 | 0.024071 | recon 0.024071 @e12 | recon 0.027858 | batch=128<br>epochs=200<br>lr=6e-05<br>g_ch=96<br>d_ch=96<br>g_blocks=4<br>d_blocks=2<br>g_hidden=4096<br>d_hidden=2048<br>lambda_recon=10.0<br>disc_iter=5 |  |
| `outputs/training/transformer_wgan/svi-excel/20260412_201424` | `transformer_wgan` | `svi-excel` | no metrics | `lp` | 0/200 |  |  |  |  |  | batch=64<br>epochs=200<br>lr=6e-05<br>g_lr=6e-05<br>d_lr=6e-05<br>lambda_recon=10.0<br>critic_iter=5 |  |
| `outputs/training/transformer_wgan/svi-excel/20260412_201651` | `transformer_wgan` | `svi-excel` | partial/interrupted | `lp` | 168/200 | `val_hybrid_score` | 168 | 0.121156 | recon 0.058147 @e168 | recon 0.058147 | batch=64<br>epochs=200<br>lr=6e-05<br>g_lr=6e-05<br>d_lr=6e-05<br>lambda_recon=10.0<br>critic_iter=5 |  |
| `outputs/training/transformer_wgan/svi-excel/20260412_203554` | `transformer_wgan` | `svi-excel` | finished | `lp` | 120/120 | `val_hybrid_score` | 120 | 0.124167 | recon 0.058830 @e120 | recon 0.058830 | batch=32<br>epochs=120<br>lr=5e-05<br>g_lr=6e-05<br>d_lr=2.5e-05<br>lambda_recon=50.0<br>critic_iter=1 |  |
| `outputs/training/transformer_wgan/svi-excel/20260412_204017` | `transformer_wgan` | `svi-excel` | partial/interrupted | `lp` | 19/120 | `val_hybrid_score` | 19 | 0.800887 | recon 0.284403 @e19 | recon 0.284403 | batch=32<br>epochs=120<br>lr=5e-05<br>g_lr=6e-05<br>d_lr=2.5e-05<br>lambda_recon=50.0<br>critic_iter=1 |  |
| `outputs/training/transformer_wgan/svi-excel/20260412_204404` | `transformer_wgan` | `svi-excel` | finished | `lp` | 120/120 | `val_hybrid_score` | 120 | 0.139097 | recon 0.063806 @e120 | recon 0.063806 | batch=32<br>epochs=120<br>lr=4e-05<br>g_lr=5e-05<br>d_lr=2e-05<br>lambda_recon=0.0<br>critic_iter=1 |  |
| `outputs/training/volgan/svi-excel/20260412_135527` | `volgan` | `svi-excel` | finished | `lp` | 100/100 | `val_mae` | 18 | 0.026951 | mae 0.026951 @e18 | mae 0.028348 | batch=32<br>epochs=100<br>lr=0.0001 |  |
| `outputs/training/volgan/svi-excel/20260412_135951` | `volgan` | `svi-excel` | finished | `lp` | 100/100 | `val_mae` | 18 | 0.026951 | mae 0.026951 @e18 | mae 0.028348 | batch=32<br>epochs=100<br>lr=0.0001 |  |
| `outputs/training/volgan/svi-excel/20260412_172740` | `volgan` | `svi-excel` | finished | `lp` | 100/100 | `val_mae_gap_vs_current` | 3 | -0.000432 | mae 0.025423, gap -0.000432 @e3 | mae 0.031786, gap 0.005931 | batch=32<br>epochs=100<br>lr=0.0001<br>g_lr=0.0001<br>d_lr=0.0001 |  |
| `outputs/training/volgan/svi-excel/20260412_182057` | `volgan` | `svi-excel` | finished | `lp` | 100/100 | `val_mae_gap_vs_current` | 3 | -0.000432 | mae 0.025423, gap -0.000432 @e3 | mae 0.031786, gap 0.005931 | batch=32<br>epochs=100<br>lr=0.0001<br>g_lr=0.0001<br>d_lr=0.0001 |  |
| `outputs/training/volgan/svi-excel/20260412_183027` | `volgan` | `svi-excel` | no metrics | `lp` | 0/100 |  |  |  |  |  | batch=32<br>epochs=100<br>lr=0.0001<br>g_lr=0.0001<br>d_lr=0.0001 |  |
| `outputs/training/volgan/svi-excel/20260412_183055` | `volgan` | `svi-excel` | finished | `lp` | 100/100 | `val_mae_gap_vs_current` | 52 | 0.002665 | mae 0.028520, gap 0.002665 @e52 | mae 0.033037, gap 0.007182 | batch=32<br>epochs=100<br>lr=0.0001<br>g_lr=0.0001<br>d_lr=0.0001 |  |
