# RQ1 Intraday 60-Minute Lookback Plan

> Created: 2026-08-03 12:00:56 UTC  
> Scope: literature-supported intraday lookback design for raw-vol surface and news forecasting

## 1. Objective

The proposed experiment replaces the single-state input

```text
current surface + contemporaneous news -> next 5-minute surface
```

with a causal rolling-history input:

```text
surfaces and available news over (t - n, t] -> future surface(s)
```

Because the study is intraday, a daily `T-20` lookback is not appropriate. The
paper-facing primary specification should use the previous 60 minutes, equivalent
to 12 five-minute intervals.

```text
primary lookback_minutes = 60
primary lookback_steps   = 12
primary forecast_horizon = 5 minutes
```

If each time index represents a five-minute interval, the model input is therefore
`T-11, ..., T`, commonly described as a 12-step rolling history.

## 2. Literature Basis

### 2.1 Immediate and persistent announcement effects

[Ederington and Lee (1993)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04750.x)
study scheduled macroeconomic announcements in interest-rate and foreign-exchange
futures. Most price adjustment occurs within the first minute, while volatility
remains substantially above normal for approximately 15 minutes and can remain
slightly elevated for several hours.

[Balduzzi, Elton, and Green (2001)](https://ideas.repec.org/a/cup/jfinqa/v36y2001i04p523-543_00.html)
find similarly rapid price adjustment in the U.S. Treasury market, together with
persistent increases in volatility and trading volume. Bid-ask spreads generally
return toward normal after approximately 5 to 15 minutes.

These findings imply that a 15-minute window captures the strongest immediate news
reaction, but three five-minute observations are a short sequence for estimating
surface dynamics.

### 2.2 Five-minute Treasury dynamics

[Andersen, Bollerslev, Diebold, and Vega (2007)](https://www.federalreserve.gov/pubs/ifdp/2006/871/ifdp871.htm)
analyze global stock, bond, and foreign-exchange responses using five-minute
returns. Their volatility equation uses nine own-volatility lags, corresponding to
45 minutes, and announcement indicators up to 14 five-minute lags, corresponding
to 70 minutes. Their specification therefore places the relevant intraday
volatility and news-response range around 45 to 70 minutes.

[Herrmann, Teis, and Yu (2014)](https://www.eurex.com/ex-en/find/news-center/news/IWQW-Study-components-of-intraday-volatility--359538)
show for DAX and Bund futures that autoregressive behavior, intraday seasonality,
long-memory components, scheduled releases, and unscheduled news all contribute to
intraday volatility predictability.

### 2.3 Design inference

No cited paper establishes 60 minutes as a universally optimal window for Treasury
option implied-volatility surfaces. The 60-minute choice is a pre-registered design
inference from:

```text
strong immediate news response: approximately 0-15 minutes
volatility autoregressive range: approximately 45 minutes
extended announcement response: approximately 70 minutes
```

Sixty minutes lies inside this documented range, provides 12 sequential surfaces,
and remains short enough to avoid importing stale intraday news into every sample.

## 3. Exact Sample Definition

For forecast origin `t`, all input information must be observable by `t`.

```text
S_t     = raw-vol surface constructed from trades in [t-5min, t)
S_t-1   = raw-vol surface constructed from trades in [t-10min, t-5min)
...
S_t-11  = raw-vol surface constructed from trades in [t-60min, t-55min)

news input = articles with availability_time in (t-60min, t]
target     = raw-vol surface constructed from trades in [t, t+5min)
```

The primary supervised mapping is:

```text
X_t = {S_t-11, ..., S_t, news_(t-60min,t]}
Y_t = S_t+1
```

The model should predict a persistence-anchored residual:

```text
delta_surface_hat = model(surface_history, news_history)
surface_hat_t+1    = S_t + delta_surface_hat
```

This makes zero residual exactly equivalent to the persistence benchmark.

## 4. News Input

Only news that was operationally available by the forecast origin may enter the
input. The existing `Europe/London` Factiva timestamp conversion and any registered
publication-availability lag must be applied before selecting articles.

Within the 60-minute window, preserve article timing rather than taking an
unweighted mean over all text:

```text
0-15 minutes before t
15-30 minutes before t
30-60 minutes before t
```

At minimum, retain:

```text
pooled LP embedding per time bucket
news count
has_news indicator
minutes since latest news
```

A time-decay or time-aware attention mechanism may be used, but all parameters must
be fitted on the training fold only.

## 5. Forecast Horizons

The primary target remains the next five-minute surface:

```text
primary horizon = 5 minutes
```

Optional direct multi-horizon robustness targets are:

```text
15 minutes
30 minutes
```

Each horizon should have a direct output head. Recursive use of the five-minute
forecast as the next model input is not part of the primary specification because
it compounds forecast errors and changes the information set.

## 6. Lookback Robustness

The pre-registered primary lookback is 60 minutes. Sensitivity analysis may report:

```text
30 minutes = 6 five-minute surfaces
60 minutes = 12 five-minute surfaces, primary
90 minutes = 18 five-minute surfaces
```

The primary window must not be changed after inspecting test performance. Any
window tuning must use validation data only. Results across secondary windows and
horizons should be treated as a multiple-comparison family.

## 7. Leakage and Session Rules

- Historical surfaces must use backward-only trade windows.
- News published after `t` must never enter the input, including articles inside
  the target interval.
- The sequence must not be silently carried across a CME maintenance break.
- Missing surfaces should use an explicit support/missingness mask and elapsed-time
  feature rather than unrestricted forward filling.
- Surface normalization, text transformation, and any sequence encoder
  preprocessing must be fitted on the training fold only.
- Evaluation must use chronological rolling-origin folds.
- Forecast origins whose targets cross a split boundary must be purged.
- Since adjacent five-minute samples overlap heavily, inference must use
  trading-day clustered bootstrap or an appropriate HAC estimator.

## 8. Controlled Comparison

The main RQ1 comparison remains:

```text
surface-history no-text model
vs
identical surface-history + matched LP model
```

Required controls are:

```text
persistence
current-surface-only no-text
60-minute surface-history no-text
60-minute surface-history + news count
60-minute surface-history + matched LP
60-minute surface-history + shuffled LP
```

For metric `m`, define:

```text
text advantage = MAE(no_text, m) - MAE(text, m)
positive value => text has lower MAE
```

The news-count control separates semantic information from the effect of news
arrival itself. The shuffled-text control tests whether correct temporal and
semantic alignment matters.

## 9. Expected Contribution and Limitation

This design lets the quantitative branch learn recent IV-surface dynamics before
the LP representation conditions the forecast through FiLM. It is better aligned
with the paper's proposed `text embedding + FiLM-WGAN` contribution than a model
that observes only one current surface.

The rolling construction can create many training rows, but adjacent rows are not
independent. The effective evidence continues to depend on the number of trading
days, distinct news events, market regimes, and valid raw-vol observations. A
larger row count must therefore not be presented as an equivalent increase in
independent sample size.

## 10. Registered Recommendation

```text
surface source      = raw-vol interpolation
sampling interval   = 5 minutes
lookback             = 60 minutes / 12 steps
primary target       = next 5-minute surface
secondary targets   = 15 and 30 minutes
forecast form        = persistence-anchored residual
primary text input   = LP embeddings available within the previous 60 minutes
primary comparison  = matched LP vs structurally identical no-text
robustness lookbacks = 30 and 90 minutes
```
