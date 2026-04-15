# Training Testing Plan

## Executive Summary

This document records the current decision boundary for thesis-facing training work. For `svi-excel` experiments, the key question is whether a text-conditioned generator improves on the held-out `current_surface` baseline on paired `current_surface -> target_surface` examples from `merged_vol.xlsx`. Older `raw-excel` results are kept for data-lineage context, but their `val_recon` metrics are not directly comparable to gap-based `svi-excel` runs.

- `cnn_wgan` run `20260414_123032` remains the primary `svi-excel` reference: best `val_mae_gap_vs_current = -0.001999` and best `val_mae = 0.023857` at epoch `19`.
- None of the 2026-04-14 23xx standalone architectures surpasses that baseline.
- `film_wgan` is the strongest new architecture, but only when checkpointed very early: it reaches `-0.001347` at epoch `12` and then drifts to `+0.002520` by epoch `220`.
- `crossattn_wgan` is more stable than `film_wgan`, but its best result (`-0.000700`) is weaker than the CNN baseline despite a large parameter budget.
- `stylemod_wgan` is now trainable after stabilization changes, but the tuned run never beats the `current_surface` baseline; it should be deprioritized.

All values below were rechecked from saved artifacts. For the 2026-04-14 23xx runs, the numbers were verified against `metrics/best_checkpoint.json`, `metrics/training_metrics.csv`, and `run.log`. Older legacy runs without the newer `run.log` layout were verified from `metrics/best_checkpoint.json` and `metrics/training_metrics.csv`.

## Reference Baselines

| Model | Best Run | Data / Embedding | Primary Metric | Best Value | Best Epoch | Decision Note |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn_wgan` | `20260414_123032` | `svi-excel / lp` | `val_mae_gap_vs_current` | `-0.001999` (`val_mae = 0.023857`, peak win rate `0.606`) | `19` | Main thesis-facing baseline for paired `current_surface -> target_surface` forecasting. |
| `cnn_wgan` | `20260410_215114` | `raw-excel / lp` | `val_recon` | `0.016845` | `102` | Best legacy raw-excel run; useful for lineage, not directly comparable to gap-based `svi-excel` scores. |
| `volgan` | `20260412_172740` | `svi-excel / lp` | `val_mae_gap_vs_current` | `-0.000432` (`val_mae = 0.025423`) | `3` | Simple normalized baseline; competitive only with very early checkpointing. |
| `transformer_wgan` | `20260412_203554` | `svi-excel / lp` | `val_hybrid_score` | `0.124167` (`val_recon = 0.058830`) | `120` | Useful negative control; materially weaker than CNN-style models and scored on a different metric. |

## 2026-04-14 23xx Architecture Sprint

All three runs below use the same `svi-excel` workbook (`data/processed/svi-excel/20260410-174929/merged_vol.xlsx`), `gan_input_ready`, LP embeddings, normalized current surfaces, normalized target deltas, normalized text embeddings, and the same train/validation split size (`2968 / 743`). The comparison is therefore meaningful for architecture ranking, with one caveat noted after the table.

| Model | Run Timestamp | Data / Embedding Mode | Core Config Delta | Generator / Critic Params | Best Epoch | Best `val_mae_gap_vs_current` | Best `val_mae` | Best Win Rate | Final-Epoch Gap | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `crossattn_wgan` | `20260414_230946` | `svi-excel / lp` | `112/112` channels, `4/2` blocks, adds `8` heads / `8` text tokens / `attn_dim=256`, batch `256` | `38.27M / 15.05M` | `109` | `-0.000700` | `0.025155` | `0.548` | `+0.001097` | Stable and recoverable, but too heavy for the gain; keep only if a stabilized rerun clears `-0.0010`. |
| `film_wgan` | `20260414_233331` | `svi-excel / lp` | `128/128` channels, `4/2` FiLM blocks, batch `256`, same loss regime as the CNN reference | `47.83M / 19.25M` | `12` | `-0.001347` | `0.024508` | `0.560` | `+0.002520` | Best new architecture, but only with aggressive early checkpointing. |
| `stylemod_wgan` | `20260414_233543` | `svi-excel / lp` | tuned stabilization run: `112/112` channels, `2/2` blocks, batch `128`, `critic_iter=3`, `lambda_recon=0.5`, `lambda_smooth=0.02`, `style_noise=0.05` | `23.37M / 13.66M` | `40` | `+0.000851` | `0.026706` | `0.475` | `+0.004233` | Stable after tuning, but still non-competitive and above the current-surface baseline. |

- Fairness note: `film_wgan` and `crossattn_wgan` are close to an apples-to-apples comparison. They share the same dataset, embedding mode, batch size `256`, `220` epochs, generator/discriminator learning rates `2e-5 / 1e-5`, `critic_iter=5`, `lambda_recon=2.0`, and `lambda_smooth=0.1`.
- Fairness note: `stylemod_wgan` `20260414_233543` is a tuned stabilization run, not a strict one-to-one comparison. It changes width, depth, batch size, `critic_iter`, loss weights, style noise, and evaluation settings.
- `film_wgan` is the only new architecture that clearly beats the held-out `current_surface` baseline, but it is also the one with the sharpest late drift. From its best checkpoint to its final epoch, the gap worsens by `+0.003867`.
- `crossattn_wgan` degrades less (`+0.001797` from best to final gap), but the improvement ceiling is still too low relative to its parameter count.
- `stylemod_wgan` no longer collapses numerically, which is a useful engineering result, but the tuned run still never reaches a negative gap.
- None of the new architectures closes the gap to `cnn_wgan` `20260414_123032`, so the next round should optimize selectively rather than launch a broad architecture sweep.

## Next Optimization Plan

| Queue | Experiment | Target Failure Mode | Exact Change to Test | Success Threshold |
| --- | --- | --- | --- | --- |
| `1` | `film_wgan` short-horizon rerun | Best checkpoint arrives at epoch `12`, but the run keeps training until epoch `220` and drifts badly afterward. | Train for `60-80` epochs, reduce `checkpoint_warmup_epochs` to `5`, and treat epochs `8-20` as the primary selection window. | Match or beat `-0.001347` and keep the final-epoch gap below `+0.0010`. |
| `2` | `film_wgan` lighter-capacity rerun | The current `128/128` FiLM model is `47.8M / 19.3M` params for roughly `2968` train samples, which is likely overparameterized. | Test `112/112` or `96/96` channels with `batch_size=128` or `192`, while keeping the current learning-rate and constraint regime for the first pass. | Match or beat `-0.001347` with a smaller model and without faster late drift. |
| `3` | `crossattn_wgan` stabilization rerun | The model is reasonably stable, but its gain is too small and it lacks the newer stabilization stack. | Port cosine LR decay, GP warmup, and grad clipping to `crossattn_wgan`, then rerun with `attn_dim=128`, `num_attn_heads=4`, `num_text_tokens=4`, and `fusion_hidden_dim=1024`. | Improve beyond the current best `-0.000700` and reach a negative gap before epoch `80`. |
| `4` | `crossattn_wgan` keep-or-drop rule | Added complexity is not justified unless the architecture materially closes the gap to the CNN reference. | After one stabilized rerun, keep `crossattn_wgan` in the main candidate set only if it improves past `-0.0010`. | If best gap stays above `-0.0010`, demote it to a secondary architecture and stop broad tuning. |
| `5` | `stylemod_wgan` deprioritization | Style conditioning can still be bypassed by the final `condition_vector + global_style` fusion path, and the tuned run remains above baseline. | Run at most one optional diagnostic ablation that weakens or removes the final bypass path. Do not schedule a broader sweep before that test. | Only revisit StyleMod as a main candidate if the ablation produces a negative gap and materially improves on `+0.000851`; otherwise keep it deprioritized. |

This queue is intentionally narrow. The next goal is not to maximize architecture diversity. The next goal is to determine whether FiLM can beat the CNN baseline under disciplined checkpointing, and whether cross-attention can justify its extra complexity once it receives the same stabilization tools.
