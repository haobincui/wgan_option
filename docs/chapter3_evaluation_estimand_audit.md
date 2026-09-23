# Chapter 3 evaluation point-estimate audit

Scope note: this records the initial primary-point audit. The subsequent
[four-fold reporting reconciliation](chapter3_four_fold_reporting_revision.md)
explicitly recomputes architecture/alignment robustness under equal-cell
weights and independently verifies shared-market uncertainty. Its final
archive mapping supersedes the narrower robustness and uncertainty scope
described below; the original-prediction checks here remain applicable.

## Assessment

Observed point estimates are verified and suitable for reporting with the limitations below. This audit, performed on 2026-09-04, changes reporting definitions, not training data, checkpoints, predictions, or frozen experiment outputs. The primary RQ1–RQ3 evaluation comprises all four 2023 rolling test quarters; Q4 is not a privileged holdout. The same market panel is evaluated under ten seeds.

The reported MAE is the original-sample arithmetic mean. Bootstrap means must not replace it. RQ3 matched LP has observed MAE **0.0020257410644962314**, identically in the full-branch, primary-branch, and same-checkpoint intervention panels. The previously displayed **0.002040977** was labelled a bootstrap mean, not the original-sample MAE; its exact archived numerical provenance is not established by this audit.

## Reproduction and source lineage

```bash
python -m scripts.rq123.audit_evaluation_estimands
python -m unittest discover -s tests/test_scripts -p 'test_audit_evaluation_estimands.py'
```

The CLI prints JSON to stdout and performs no writes. It verifies the following frozen checksums before calculation and checks again afterward. Paths below are relative to `outputs/experiments/`.

| Source | Path | Rows | SHA-256 |
|---|---|---:|---|
| Direct RQ1/RQ2 | `rq12_news_first_vol_text_10seed_composite_lr2p5e5_descriptive_no_bootstrap_exact_ttm_rolling_v1/analysis/text_10seed_composite_pair_metrics.csv` | 25,000 | `d832b3b057ea440845bdd97312744bf5f75e706c80c2057a44d8f88e64f38754` |
| RQ3 full branches | `rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1/evaluation/standard_pair_metrics.csv.gz` | 35,000 | `87e70dc0842c9fa99dcf98c40c9ec48c23ce1841a6e38f08559d5ea8e2db90e5` |
| RQ3 primary branches | `rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1/analysis/standard_primary_panel.csv.gz` | 15,000 | `705783913e5252d8754cf9820ba9ca3c6a361dcc4fe6defec0fada40817cf041` |
| RQ3 interventions | `rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1/analysis/intervention_primary_panel.csv.gz` | 15,000 | `4898c79b51db2d7aacee8c749e14b2edf18e8920ce50c9c04d59375b4d77009b` |

All four sources cover 500 distinct pairs and 148 CME sessions. Quarter-specific pair/session counts are 110/34, 112/36, 135/33, and 143/45. Duplicate records, incomplete seed/condition market panels, missing identifiers, non-finite losses, mismatched persistence values, or changed hashes fail validation. RQ1 and RQ2 reuse the same matched-LP rows and persistence inputs. All three RQ3 matched panels agree record by record in `(seed, fold, pair_id, session_id, target_mae, persistence_mae)`.

If the corrected bootstrap archive's `analysis/all_arm_summary.csv` exists, the CLI additionally checks its primary `observed_mean_mae` entries; bootstrap means and uncertainty columns are not validated by that comparison. The default saved summary was absent at this audit's execution, so the recorded status was `not_available_not_verified`, not a pass. The raw point audit is independent of that archive.

## Weighting contract

Let `e[a,s,f,p]` be the frozen pair-level masked MAE. Define

```text
m[a,s,f] = sum_p e[a,s,f,p] / N[f]
M[a,f]   = sum_s m[a,s,f] / 10
M[a]     = sum_f M[a,f] / 4
theta[a,r] = sum_s,f log(m[a,s,f] / m[r,s,f]) / 40
```

Within a quarter, each pair has equal weight after its own mask-normalized loss has been calculated. Each seed and, in the four-fold aggregate, each quarter has equal weight. Thus each seed–fold cell has weight 1/40 and each of its pair losses has weight `1/(40 N[f])`. Sessions are dependence blocks for resampling; they are **not** equally weighted observations in the point estimator. Nor does the primary aggregate pool all 500 pairs with a common weight.

Descriptive improvements use `100 * (1 - M[a]/M[persistence])`. Normalized MAEs divide the corresponding arithmetic means. Paired inference retains the original observed `theta`, and geometric gain is `100 * (1 - exp(theta))`. The geometric gain need not equal the arithmetic improvement because averaging logarithms of cell ratios is not taking the ratio of arithmetic averages. None of these formulas averages ten predicted surfaces before scoring; the ten separately evaluated model seeds are averaged at the loss level.

The chapter's explicitly labelled architecture and alignment-window robustness analyses retain their historically frozen **pooled-pair** estimand. Source `scripts/rq123/chapter3_bootstrap_sources.py` assigns `pooled_pair` to those jobs. They should not be relabelled equal-cell or silently reweighted. The primary RQ1–RQ3 point-estimate correction does not recalculate those robustness results. RQ4 also retains its separately specified conditional estimands.

## Verified original-sample values

The following quarterly MAEs are shared by the relevant RQ1 and RQ2 tables.

| Condition | 2023Q1 | 2023Q2 | 2023Q3 | 2023Q4 |
|---|---:|---:|---:|---:|
| Matched LP | 0.0015831176 | 0.0030479438 | 0.0018029870 | 0.0016699527 |
| Pure-CNN | 0.0015834535 | 0.0030503670 | 0.0018022226 | 0.0016702341 |
| BoW | 0.0015830288 | 0.0030462458 | 0.0018130471 | 0.0016694485 |
| Sentiment | 0.0015832816 | 0.0030496531 | 0.0018118503 | 0.0016726503 |
| Persistence | 0.0015842193 | 0.0030558655 | 0.0018011261 | 0.0016759210 |

The CLI also prints every quarterly improvement, normalization ratio, shuffled-LP diagnostic, and four-fold direct-training mean at full floating-point precision.

| RQ3 condition | Observed four-fold MAE | Improvement vs persistence (%) | MAE / Pure-CNN continuation |
|---|---:|---:|---:|
| Matched LP | 0.0020257411 | 0.1745 | 0.999895 |
| Zero text | 0.0020257684 | 0.1732 | 0.999909 |
| Shuffled LP | 0.0020281161 | 0.0575 | 1.001067 |
| BoW | 0.0020266582 | 0.1293 | 1.000348 |
| Sentiment | 0.0020276669 | 0.0796 | 1.000846 |
| Pure-CNN continuation | 0.0020259535 | 0.1641 | 1.000000 |
| Pure-CNN parent | 0.0020265517 | 0.1346 | 1.000295 |
| Persistence | 0.0020292830 | 0.0000 | 1.001643 |

The matched checkpoint evaluated with zero input has observed MAE 0.0020248708880486444; with wrong input it has 0.0020249322603761856. These are inference-time interventions, not the separately trained zero-text and shuffled-text branches.

Matched LP versus zero text illustrates the distinction between arithmetic and geometric reporting: its ratio of arithmetic MAEs is 0.9999865295674331, whereas its mean cell log ratio is −0.00007160920122681071. The difference is a defined aggregation distinction, not conflicting predictions.

## Bootstrap-mean reconciliation

The old rounded display, 0.002040977, exceeds the verified observed matched-LP mean by 0.000015235935503768363. This cannot be explained as display rounding. It must not be retained as a second observed point estimate.

Cluster resampling creates variable pair counts because sessions contain unequal numbers of pairs. A draw computes a ratio of a resampled loss sum to its resampled pair count, and the mean of these random-denominator ratios need not equal the original ratio. Finite Monte Carlo error is another difference. Resampling seeds and folds does not justify substituting the resulting draw mean for the observed point estimate.

As an independent diagnostic, the current `shared_panel_bootstrap_core` was run **in memory**, without saving schedules or outputs, on all seven RQ3 branches plus persistence with `estimand="equal_cell"`, 10,000 iterations and RNG seed 20260904. It returned:

| Matched-LP quantity | Value |
|---|---:|
| Observed equal-cell MAE | 0.0020257410644962314 |
| Current shared-panel draw mean | 0.0020447280155658615 |
| Previously displayed TeX draw mean | 0.002040977 |

Immediately after the replay, the SHA-256 was `1d65554defcde02e0c5e9db841678238e8433bad67bf5ff902869a9da14dba4d` for `scripts/rq123/shared_panel_bootstrap_core.py` and `010ff0afd4b856fe3983a54a2cb67ac1d56d20e7da8997faa5e235275b4b0cc8` for `configs/rq123/chapter3_shared_panel_bootstrap_v2.yaml`. These post-execution hashes identify the inspected implementation/configuration state; they are not an immutable execution manifest. The replay explicitly passed the three arguments above rather than running the full configuration-driven analysis. Its frozen RQ3 input hash is recorded in the source table. The exact read-only reconstruction is:

```bash
python - <<'PY'
import pandas as pd
from scripts.rq123.audit_evaluation_estimands import REPO_ROOT, SOURCES
from scripts.rq123.shared_panel_bootstrap_core import prepare_panel, make_schedule, run_bootstrap
data = pd.read_csv(REPO_ROOT / SOURCES['rq3_full'][0]).rename(
    columns={'arm': 'condition', 'target_mae': 'value'})
persistence = data[data.condition.eq('film_lp_matched')].copy()
persistence['condition'] = 'persistence'
persistence['value'] = persistence['persistence_mae']
panel = prepare_panel(pd.concat([data, persistence], ignore_index=True))
schedule = make_schedule(panel, iterations=10000, rng_seed=20260904)
result = run_bootstrap(panel, schedule, estimand='equal_cell')
index = result.conditions.index('film_lp_matched')
print(result.observed_means[index], result.mean_draws[:, index].mean())
PY
```

Therefore the old displayed number is **not** numerically reproduced by this current shared-panel replay. It may not be attributed to the current corrected archive without further provenance evidence. This diagnostic does not validate or update any chapter standard errors, intervals, or p-values. The main audit CLI deliberately performs no bootstrap and marks the historical display as not reproduced; the replay above was a separate bounded check.

## Verification and limitations

Nine focused regression tests pass, covering unequal fold sizes, pair versus session weights, duplicate and missing rows, invalid losses, persistence drift, exact matched-input reconciliation, arithmetic/geometric distinctions, and checking observed rather than bootstrap means. A separate chapter transcription check matched 96 displayed numeric entries to the frozen source calculations at the chapter's stated precision (RQ1/RQ2 MAEs, improvements and ratios; RQ3 MAEs, improvements and ratios).

The confidence assessment applies to these reporting point estimates. Frozen pair-level metrics were audited, not regenerated from prediction tensors. No model was retrained. Current raw-target, market-clock, test-panel reuse, and dependence limitations remain. The audit does not certify the uncertainty estimates or the missing-at-execution corrected archive, and it does not complete RQ4 results.
