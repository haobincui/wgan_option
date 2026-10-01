# Chapter 3: four-fold reporting reconciliation

## Reporting decision

The primary RQ1–RQ3 evaluation is the complete four-fold experiment, not Q4 alone. There are 110, 112, 135 and 143 test pairs (500 total), and 34, 36, 33 and 45 CME sessions (148 total). Each primary condition has ten seeds in every fold. No training, target construction, checkpoint selection or prediction was changed for this revision.

The reporting contract is:

1. Score each pair using its masked surface MAE, after averaging the 64 Monte Carlo forecasts within that trained seed.
2. Average pair losses within each seed–fold cell.
3. Average seeds equally within the fold and the four folds equally overall.
4. Calculate descriptive improvements and normalized MAEs from unrounded observed means.
5. Calculate paired proportional effects as the mean of original cell-level log-MAE ratios. Bootstrap draws supply uncertainty, not replacement point estimates.

The arithmetic improvement `100 * (1 - mean_MAE_A / mean_MAE_B)` need not equal the geometric gain `100 * (1 - exp(mean_cell_log_ratio))`. Both are observed-data estimators with explicitly defined weights.

## Reconciled results

RQ3 matched LP has observed four-fold MAE **0.0020257410644962314**, displayed as **0.0020257411** in both the branch table and the same-checkpoint intervention discussion. The standard and unchanged intervention panels contain identical 5,000 seed–pair predictions, including checkpoint and prediction hashes. The former display `0.002040977`, labelled a bootstrap mean, is no longer used as a second observed MAE; it is not necessary to recover that historical draw average to verify the original predictions.

The observed arithmetic reduction relative to the separately trained zero-text branch is about 0.0013%; the mean-cell-log-ratio geometric gain is about 0.0072%. The zero-input intervention is a different comparison and must not be confused with the zero-text training branch.

The shared-market bootstrap keeps the same sampled fold-occurrence/session schedule across seeds and paired conditions. With 10,000 draws, the main updated inference is:

| Comparison | Geometric gain (%) | Reported sign-tail probability |
|---|---:|---:|
| RQ1 LP / Pure-CNN, four-fold | 0.0188 | 0.3200, unadjusted |
| RQ2 LP / BoW | 0.1159 | 0.3234, Holm-2 |
| RQ2 LP / sentiment | 0.1792 | 0.0612, Holm-2 |
| RQ3 matched / shuffled branch | 0.1245 | 0.3174, Holm-2 |
| RQ3 matched / zero-text branch | 0.0072 | 0.4768, Holm-2 |

The chapter's tables, significance stars, interpretation paragraphs and conclusion use these updated uncertainty results. In particular, the old significant matched-versus-shuffled result is withdrawn. The probability summaries remain approximate bootstrap sign-tail diagnostics, not exact tests. Four inspected quarters do not provide independent confirmation in new market regimes.

## Scope of supplementary evidence

The architecture and alignment-window experiments also cover all four folds, with three seeds each. Their observed MAEs and paired log contrasts have been explicitly reweighted to the same equal-cell contract, and their uncertainty has been recomputed for that estimand. Their original comparison identities and Holm families are retained. This is a reporting harmonization, not a claim that the revised weighting was prospectively preregistered.

Historical pooled-pair robustness results remain unchanged in their original output directories. They are not the source of the revised chapter tables. The earlier primary-only audit in `chapter3_evaluation_estimand_audit.md` records a narrower stage of the work; its statement that robustness retained pooled weights is superseded by this explicit recomputation.

The archived capacity diagnostics cover Q3 and Q4 separately. They are labelled historical development diagnostics, with same-panel persistence references. Cross-experiment comparisons against a four-fold Pure-CNN MAE and the resulting 9.7%/16.0% apparent gains were removed. No missing capacity folds were invented. RQ4 results were not completed or re-estimated in this revision.

This paragraph records the state of that earlier reporting revision only. The active chapter now supersedes its historical capacity table with the separately trained F4 (2023Q4) FiLM-CNN/Pure-CNN capacity robustness test labelled `tab:ch3:f4_film_pure_capacity_robustness`. The old label `tab:ch3:legacy_full_wgan_capacity_vs_fixed_pure_cnn` and the old Q3/Q4 table remain provenance, not current Chapter 3 evidence.

## Reproducible artifacts

Paths are relative to the repository root:

| Purpose | Analysis script | Verified output root |
|---|---|---|
| Original MAE and matched-prediction identity | `scripts/rq123/chapter3_point_estimate_audit.py` | `outputs/analysis/chapter3_observed_point_estimates_v1` |
| Primary shared-market uncertainty | `scripts/rq123/chapter3_main_inference_audit.py` | `outputs/analysis/chapter3_main_inference_verified_v1` |
| Equal-cell robustness, coverage and historical capacity | `scripts/rq123/chapter3_robustness_point_audit.py` | `outputs/analysis/chapter3_robustness_equal_cell_v1` |

The capacity output in the last row belongs only to the superseded historical audit. The current F4 table is sourced from `outputs/experiments/rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1/analysis/` and is verified through its own terminal QA and the Chapter 3 external-table binding.

Each archive records source hashes and unrounded values. Bootstrap archives preserve panels, schedules and draws for replay. The main inference audit also records the upstream implementation drift observed while a separate task was updating the shared-bootstrap driver/adapter. It verifies the frozen inputs and replays the retained panels and schedules with the stable numerical core; it does not misrepresent the earlier driver's failed live-code hash check as a pass. The intermediate `chapter3_shared_market_panel_bootstrap_10000_point_audit_v1`/`v3` directories are upstream provenance, not the final standalone verification entrypoint.

The final primary audit independently passes for nine jobs, 52 contrasts, 59 arm summaries and 1,254 observed cell/fold/overall checks. Its standalone replay also passes. Only these main-analysis jobs are taken from the upstream archive: that archive's pooled robustness jobs are not the chapter's current robustness results. The latter come exclusively from `chapter3_robustness_equal_cell_v1`, whose two shared-schedule jobs also pass standalone replay.

Replay the retained inference archives with:

```bash
python -m scripts.rq123.chapter3_main_inference_audit --verify-only
python -m scripts.rq123.chapter3_robustness_point_audit --verify
python -m unittest discover -s tests/test_scripts -p 'test_chapter3*py'
```

For a fresh original-point audit, use `python -m scripts.rq123.chapter3_point_estimate_audit --output <new-directory>`. Generation commands refuse to overwrite existing archives.

Tests cover equal weighting with unequal fold sizes, the arithmetic/geometric distinction, duplicate and missing-cell rejection, original point estimates rather than draw means, source integrity, matched-prediction identity, preserved Holm families and bootstrap replay. The chapter's numerical transcription and LaTeX environment/label structure are checked separately. A full PDF build has not been performed because this workspace has no `pdflatex`, `latexmk` or `tectonic` executable.
