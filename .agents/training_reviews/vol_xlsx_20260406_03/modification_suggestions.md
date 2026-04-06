# Modification Suggestions

## 1. Keep using the new best-checkpoint workflow

This comparison supports keeping the new infrastructure changes.

Reason:

- the modeling result is broadly comparable to the pre-change run
- the workflow now captures the actual best epoch automatically
- this is a real improvement in reproducibility and downstream evaluation quality

Operational recommendation:

- use `generator_best.pt` for evaluation by default
- stop using the final epoch checkpoint as the default reporting artifact

## 2. Fix artifact path portability in `best_checkpoint.json`

The current metadata stores artifact paths that reflect the original run location, not the archived timestamped folder.

Recommended change:

- store checkpoint filenames relative to `models_path`, not absolute or repo-root-style paths
- or store both:
  - `artifact_filenames`
  - `models_path_at_run_time`

Why this matters:

- archived experiment folders should remain self-describing after being renamed or copied
- this is especially important for thesis reproducibility and later audit work

## 3. Make the best checkpoint easier to interpret at a glance

The current `best_checkpoint.json` is already useful, but it would be more review-friendly if it also recorded:

- `final_epoch`
- `final_metric`
- `best_vs_final_delta`
- whether best epoch is also the final epoch

This would make comparison and reporting faster without needing to reopen the metrics history.

## 4. Add an option to save the best epoch’s full metric row

Recommended change:

- include the full epoch metric dictionary for the best epoch inside `best_checkpoint.json`

Why:

- it would preserve context such as `gp`, `g_calendar`, and `g_butterfly`
- it would make archived run summaries more complete

## 5. Do not over-interpret the post-change run as a model improvement

From this pair of runs, the post-change code should not be described as having clearly improved forecasting accuracy.

A better statement is:

- model quality stayed in the same general range
- best-point and late-stage average validation performance were slightly better in `03`
- final-epoch performance was slightly worse in `03`
- the strong gain is in experiment management, not clearly in predictive power

## 6. Next experimental priority: compare best-to-best, not final-to-final

The new infrastructure makes a better evaluation protocol possible.

Recommended policy for future comparisons:

- compare `best_checkpoint.json` metrics and `generator_best.pt`
- report periodic checkpoint best only as a secondary robustness check
- treat final epoch results as diagnostic, not as the main selection rule

## 7. Next modeling priority: reduce late-stage adversarial oscillation

The post-change run shows:

- slightly better late-stage average `val_recon`
- larger late-stage `g_total` oscillation

That suggests the next modeling experiments should focus on reducing adversarial instability without sacrificing reconstruction quality.

Recommended next trials:

- `lambda_recon = 15`
- `lambda_recon = 20`
- `discriminator_iter = 3`
- optionally add a validation-driven learning-rate scheduler

### 7.1 Dynamic learning rate is a reasonable next step, but the scheduler should be validation-driven

Dynamic learning rate is a sensible next direction for the `vol_xlsx` WGAN path.

The comparison results support that view because:

- late-stage validation is already in a plateau region
- late-stage `val_recon` still has room for controlled refinement
- adversarial dynamics remain non-smooth, so scheduler decisions should follow validation behavior rather than epoch count alone

For this reason, the preferred strategy is not to present the next change as “copying an LLM learning-rate schedule” directly.

Instead, the better default recommendation is:

- first choice: `ReduceLROnPlateau`
- monitor metric: `val_recon`
- suggested factor: `0.5`
- suggested patience: `5` to `8`
- suggested `min_lr`: `1e-5`
- apply the same scheduler policy to both generator and discriminator at first
- keep warmup optional and short, not the main recommendation

LLM-style schedules are still useful as inspiration for the general idea of dynamic decay.

However, this project’s WGAN setup is better served by validation-driven decay than by a fixed `warmup + cosine decay` schedule as the default first move.

## 8. Recommended thesis-facing interpretation

For thesis writing or chapter notes, a careful interpretation would be:

- the infrastructure revision improved reproducibility and model-selection quality
- it did not materially degrade training behavior
- numeric forecasting quality remained broadly comparable
- the new workflow is therefore a justified improvement even without a large performance jump
