# Rolling-Origin Evaluation for Short-Horizon Time-Series Forecasting

## Purpose and adopted decision

This note records the literature basis and executable data-splitting protocol
used to evaluate the five-minute volatility-surface forecasts in this thesis.
The primary experiments retain an **expanding-window rolling-origin** design.
For every fold, the model is fitted only with observations that would have been
available before that fold's validation and test periods. A training set formed
from the full sample after subtracting only the validation and test blocks is
not used for the primary results, because it would allow early folds to learn
from labelled observations occurring after their own forecast dates.

Here, *temporally causal* refers only to information availability: a forecast
at time \(t\) is constructed without observations from after \(t\). It does not
mean that the experiment identifies a causal economic effect of news text.

## 1. Literature basis

### 1.1 Forecasting evaluates the future conditional on the past

The relevant estimand is forecast performance when future observations are not
available at model-estimation time. Tashman (2000) recommends out-of-sample
evaluation with rolling forecast origins and multiple test periods, rather than
relying on a single split whose result may be specific to one historical
episode. Bürkner, Gabry, and Vehtari (2020) formalise the same information set
through leave-future-out evaluation: prediction targets lie after the data used
to fit the model. Hewamalage, Ackermann, and Bergmeir (2023) describe rolling
origin, time-series cross-validation, and prequential evaluation as closely
related procedures that preserve temporal order.

An expanding window incorporates every observation available before the next
forecast origin. A sliding window instead discards the oldest observations and
is useful when old regimes have become unrepresentative. The expanding design
is adopted here because the fold-level sample is limited and the primary aim is
to use all historically available information without introducing look-ahead.
A sliding-window analysis may be reported separately as a regime-sensitivity
check, but it answers a different question.

### 1.2 Why blocked complement cross-validation is not the primary design

Blocked cross-validation keeps each held-out block contiguous but generally
trains on all remaining blocks, including blocks occurring after the held-out
period. The `hv`-block variant additionally removes observations adjacent to
the held-out block to weaken short-range dependence (Racine, 2000). These
methods can be useful for model selection or average-risk estimation under
stationarity, but the gap does not convert a forecast of the past using future
training observations into a deployable real-time forecast.

Bergmeir and Benítez (2012) find blocked cross-validation useful in stationary
settings. Bergmeir, Hyndman, and Koo (2018) provide a narrower theoretical case
in which ordinary K-fold cross-validation can be valid for purely
autoregressive models with uncorrelated errors. Those conditions are not
assumed in this thesis: the forecasting system includes external news
representations, current volatility surfaces, a nonlinear FiLM U-Net, and an
adversarial training objective. In a broad empirical comparison, Cerqueira,
Torgo, and Mozetič (2020) find that blocked cross-validation is competitive for
stationary series, whereas temporally ordered out-of-sample procedures applied
over multiple testing periods are more accurate estimators for real-world
series with non-stationary variation.

### 1.3 Examples from modern forecasting research

The same temporal ordering appears in prominent forecasting benchmarks. The
Temporal Fusion Transformer study partitions each series into training,
validation, and later hold-out test periods, using validation for
hyperparameter selection and the subsequent period for out-of-sample
evaluation (Lim et al., 2021). The M4 competition supplied participants with
the historical training portion while retaining the final forecast horizon as
a hidden test set (Makridakis, Spiliotis, and Assimakopoulos, 2020). These
examples differ from this thesis in model and data domain, but they support the
same evaluation principle: future test outcomes are not inputs to model
fitting or selection.

## 2. Adopted four-fold protocol

Let \(t_i\) denote the effective forecast-origin timestamp of pair \(i\). For
fold \(k\), let \(a_k\) be the validation start, \(b_k\) the test start, and
\(c_k\) the test end. The partitions are

\[
\begin{aligned}
\mathcal{D}^{(k)}_{\mathrm{train}} &= \{i: t_i < a_k\}, \\
\mathcal{D}^{(k)}_{\mathrm{validation}} &= \{i: a_k \leq t_i < b_k\}, \\
\mathcal{D}^{(k)}_{\mathrm{test}} &= \{i: b_k \leq t_i < c_k\}.
\end{aligned}
\]

Thus, the training window expands as the forecast origin advances. An outer
test quarter from an earlier fold may enter the training set of a later fold,
but only after that quarter would have become historically observable. This is
the normal prequential update rule and does not leak that quarter into its own
earlier forecast.

The current exact-TTM five-minute pair universe produces the following frozen
partitions. Counts are reported as `pairs / CME sessions`.

| Fold | Training period | Validation period | Test period | Train | Validation | Test |
| --- | --- | --- | --- | ---: | ---: | ---: |
| F1 | \(t <\) 2022-10-01 | 2022Q4 | 2023Q1 | 382 / 93 | 144 / 40 | 110 / 34 |
| F2 | \(t <\) 2023-01-01 | 2023Q1 | 2023Q2 | 526 / 133 | 110 / 34 | 112 / 36 |
| F3 | \(t <\) 2023-04-01 | 2023Q2 | 2023Q3 | 636 / 167 | 112 / 36 | 135 / 33 |
| F4 | \(t <\) 2023-07-01 | 2023Q3 | 2023Q4 | 748 / 203 | 135 / 33 | 143 / 45 |

The executable configuration uses `effective_origin_utc` as its split key. A
pair-lineage audit found no pair or CME-session overlap across the three
partitions of any fold and no training pair whose five-minute target endpoint
crosses into the adjacent validation interval. Consequently, the current
origin-based split is also label-window safe for this frozen dataset. Future
rebuilds must re-run this audit and should enforce
`target_snapshot_time_utc < validation_start` directly, or purge boundary
observations spanning at least the forecast horizon.

## 3. Fold-local fitting and test isolation

The time boundaries alone are not sufficient to prevent leakage. The following
rules apply independently within every fold:

1. Model parameters are learned only from the training partition.
2. Early stopping, epoch selection, and learning-rate decisions use only the
   validation partition.
3. BoW vocabulary construction, dimensionality reduction, scaling, and any
   other learned preprocessing are fitted from the fold's training data only.
4. Validation and test observations are transformed using the already-frozen
   training statistics; they never update those statistics.
5. Test loaders and test predictions are created only after the eligible
   checkpoint set has been frozen.
6. Competing text arms within a fold use the same test pairs and Monte Carlo
   noise bank so that forecast errors can be compared pairwise.

These rules distinguish model selection from final error estimation. Validation
performance may choose a checkpoint, whereas test performance is used only to
estimate the error of that frozen choice.

## 4. Rejected primary alternative and allowed secondary use

If the complete pre-2024 universe were used and only each fold's validation and
test blocks were removed, the resulting training sets would be larger but
would contain the following future-labelled observations:

| Fold | Expanding-window train pairs | Full-complement train pairs | Complement pairs after that fold's test |
| --- | ---: | ---: | ---: |
| F1 | 382 | 772 | 390 |
| F2 | 526 | 804 | 278 |
| F3 | 636 | 779 | 143 |
| F4 | 748 | 748 | 0 |

There is no direct pair or session duplication in these complement sets. The
problem is temporal direction: F1--F3 would use labelled market and text data
from quarters that had not yet occurred at their own forecast origins. Such an
experiment may be run as a separately named
`retrospective_blocked_complement_cv` robustness analysis to study
representation stability over a fixed historical dataset. It must not replace
the primary rolling-origin estimate or be described as out-of-time forecasting.

## 5. Interpretation and claim boundary

The rolling protocol estimates how the frozen modelling procedure performs
across successive historical forecast origins while respecting the information
available at each origin. It does not by itself establish stationarity, causal
effects, or robustness to random initialisation. Multi-seed experiments are
needed for the latter. Moreover, because parts of the 2023 evaluation period
have been examined during model development, results using these folds are
labelled `retrospective_rolling_development`, rather than confirmatory evidence
from a previously untouched hold-out period.

After a model family and training rule have been selected, a deployment model
may be refitted on all observations available before a genuinely later forecast
origin. This refit increases the usable training sample without contaminating
the earlier rolling evaluation. Its performance still requires outcomes from a
subsequent period that were not used for architecture, hyperparameter, or epoch
selection.

## 6. Thesis-ready methodology paragraph

> Forecast performance is evaluated using four expanding-window rolling-origin
> splits rather than random or full-complement cross-validation. At each
> forecast origin, the model is estimated using all observations available
> before the validation quarter, the immediately following quarter is used for
> checkpoint and epoch selection, and the subsequent quarter is reserved for
> out-of-sample evaluation. The training window therefore grows through time,
> while no fold uses observations dated after its own forecast period. This
> design follows rolling-origin and leave-future-out principles, which align
> model assessment with the operational task of forecasting future outcomes
> conditional on past information \citep{tashman2000outofsample,
> burkner2020leavefutureout,hewamalage2023forecast}. Although blocked
> cross-validation can make more efficient use of data under stationarity, its
> training folds may contain observations occurring after the held-out period;
> moreover, the validity of ordinary K-fold evaluation for time series relies
> on conditions such as a purely autoregressive specification and uncorrelated
> errors that are not imposed here \citep{bergmeir2018validity,
> cerqueira2020evaluating}. All learned text transformations and scaling
> parameters are fitted within each training fold, validation data are used
> only for model selection, and test predictions are generated only after the
> eligible checkpoint set has been frozen.

## 7. Repository links

- Current direct five-arm experiment contract:
  [`docs/news_first_vol_film_unet_direct_5arm_seed42_execplan.md`](../news_first_vol_film_unet_direct_5arm_seed42_execplan.md)
- Ten-seed rolling experiment contract:
  [`docs/news_first_vol_film_unet_nolp_10seed_execplan.md`](../news_first_vol_film_unet_nolp_10seed_execplan.md)
- Thesis experiment and test-isolation principles:
  [`docs/thesis_experiment_plan_v2_20260611.md`](../thesis_experiment_plan_v2_20260611.md)
- RQ1/RQ2 methodology and evidence boundary:
  [`docs/summary/20260725-104431/rq1_rq2_literature_implementation_io_guide.md`](../summary/20260725-104431/rq1_rq2_literature_implementation_io_guide.md)

## 8. Copy-ready BibTeX

```bibtex
@article{tashman2000outofsample,
  author  = {Tashman, Leonard J.},
  title   = {Out-of-sample tests of forecasting accuracy: An analysis and review},
  journal = {International Journal of Forecasting},
  year    = {2000},
  volume  = {16},
  number  = {4},
  pages   = {437--450},
  doi     = {10.1016/S0169-2070(00)00065-0}
}

@article{burkner2020leavefutureout,
  author  = {B{\"u}rkner, Paul-Christian and Gabry, Jonah and Vehtari, Aki},
  title   = {Approximate leave-future-out cross-validation for {Bayesian} time series models},
  journal = {Journal of Statistical Computation and Simulation},
  year    = {2020},
  volume  = {90},
  number  = {14},
  pages   = {2499--2523},
  doi     = {10.1080/00949655.2020.1783262}
}

@article{bergmeir2012use,
  author  = {Bergmeir, Christoph and Ben{\'i}tez, Jos{\'e} M.},
  title   = {On the use of cross-validation for time series predictor evaluation},
  journal = {Information Sciences},
  year    = {2012},
  volume  = {191},
  pages   = {192--213},
  doi     = {10.1016/j.ins.2011.12.028}
}

@article{bergmeir2018validity,
  author  = {Bergmeir, Christoph and Hyndman, Rob J. and Koo, Bonsoo},
  title   = {A note on the validity of cross-validation for evaluating autoregressive time series prediction},
  journal = {Computational Statistics \& Data Analysis},
  year    = {2018},
  volume  = {120},
  pages   = {70--83},
  doi     = {10.1016/j.csda.2017.11.003}
}

@article{cerqueira2020evaluating,
  author  = {Cerqueira, Vitor and Torgo, Luis and Mozeti{\v{c}}, Igor},
  title   = {Evaluating time series forecasting models: An empirical study on performance estimation methods},
  journal = {Machine Learning},
  year    = {2020},
  volume  = {109},
  number  = {11},
  pages   = {1997--2028},
  doi     = {10.1007/s10994-020-05910-7}
}

@article{hewamalage2023forecast,
  author  = {Hewamalage, Hansika and Ackermann, Klaus and Bergmeir, Christoph},
  title   = {Forecast evaluation for data scientists: Common pitfalls and best practices},
  journal = {Data Mining and Knowledge Discovery},
  year    = {2023},
  volume  = {37},
  number  = {2},
  pages   = {788--832},
  doi     = {10.1007/s10618-022-00894-5}
}

@article{racine2000hvblock,
  author  = {Racine, Jeff},
  title   = {Consistent cross-validatory model-selection for dependent data: {$hv$}-block cross-validation},
  journal = {Journal of Econometrics},
  year    = {2000},
  volume  = {99},
  number  = {1},
  pages   = {39--61},
  doi     = {10.1016/S0304-4076(00)00030-0}
}

@article{lim2021temporal,
  author  = {Lim, Bryan and Ar{\i}k, Sercan {\"O}. and Loeff, Nicolas and Pfister, Tomas},
  title   = {{Temporal Fusion Transformers} for interpretable multi-horizon time series forecasting},
  journal = {International Journal of Forecasting},
  year    = {2021},
  volume  = {37},
  number  = {4},
  pages   = {1748--1764},
  doi     = {10.1016/j.ijforecast.2021.03.012}
}

@article{makridakis2020m4,
  author  = {Makridakis, Spyros and Spiliotis, Evangelos and Assimakopoulos, Vassilios},
  title   = {The {M4} Competition: 100,000 time series and 61 forecasting methods},
  journal = {International Journal of Forecasting},
  year    = {2020},
  volume  = {36},
  number  = {1},
  pages   = {54--74},
  doi     = {10.1016/j.ijforecast.2019.04.014}
}
```

