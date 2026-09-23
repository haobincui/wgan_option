# Chapter 3 historical maturity-clock audit

The historical source IVs reproduce observed prices, and stored raw-IV grids
reproduce their frozen JSON parameters exactly. Their pricing clock and the
model's penalty clock nevertheless differ. A separate replay of CSV-to-JSON
aggregation also finds 200 differing node instances; the full raw-data lineage
is therefore not certified as reproduced. This audit preserves the existing
datasets, checkpoints, and reported results; it does not certify that the
penalties implement an actual-expiry no-arbitrage test.

## Scope and reproduction

Run from the repository root:

```bash
python scripts/rq123/audit_maturity_clock.py
PYTHONPATH=src python -m unittest discover -s tests/test_scripts -p 'test_audit_maturity_clock.py'
python -m unittest discover -s tests/test_vol -p 'test_raw_surface.py'
PYTHONPATH=src python -m unittest discover -s tests/test_market -p 'test_black76.py'
```

The CLI is read-only and emits JSON, including before/after SHA-256 verification
of every input. The audited dataset is
`data/processed/rq3/news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1`.
The full `tolerance_05m/merged_vol.xlsx` sheet `gan_input_ready` contains 1,610
news-to-pair rows mapping to 1,283 unique `pair_id` values. These counts are
before downstream support-mask/sample selection and are not the final
evaluation-panel sample size. The shared raw snapshot inputs
serve the 5/10/15/30-minute tolerance builds, so their observation counts are not
5-minute-specific trading counts. The audit does not extend its findings to
other dataset variants or regenerated workbooks.

| Input | SHA-256 |
| --- | --- |
| `shared_surface_inputs/surface-raw-excel-precalib-points.csv` | `50d922dd1feaa8f7fd3bce59d7204c4c4a350df2abad41263d52292400fd0cfd` |
| `shared_surface_inputs/surface-raw-excel.json` | `fbb1ce70c0bd12959cd8063b0d53e60a1826c1b9f7793a6e23aa66bbbfccfa9b` |
| `tolerance_05m/merged_vol.xlsx` | `661851ec64eefdef782df5a45feb09e22efe3bfe8781bcd90de2adaab9c4acd3` |

## Clocks and interpretation

Source Black-76 inversion uses elapsed time from each option trade timestamp
to its actual expiry timestamp: `tau_ACT365 = elapsed_seconds / (365 * 86400)`.
The audit reprices accepted source observations with the stored IV and verifies
their discount factor against `exp(-continuous_rate * tau_ACT365)`.

The raw surface instead uses business-day coordinates `q`: it interpolates
`sigma_ACT365**2 * q / 250` and divides by the query `q / 250` before taking
the square root. Replacing the constant 250 by 365 in both steps cancels, up
to floating-point rounding. This cancellation does not convert an ACT/365 IV
to a business-time volatility and does not establish equivalence to interpolation
on an actual-expiry-time grid.

The historical WGAN calendar and normalized-call butterfly penalties use
`q / 365` in `src/wgan_option/models/gan_model.py`. Here `q` still means
business days. This quantity is a business-day time proxy, not elapsed ACT/365
time. The relevant penalty magnitudes and violation diagnostics must therefore
be described as historical proxy-grid diagnostics. Simply changing 365 to 250
would alter the penalty objective without repairing the missing actual-expiry
mapping; preserving the historical results requires disclosing that limitation.

If changing a pricing variance clock while keeping the forward, strike, and
discount factor fixed, price preservation requires
`sigma_new = sigma_old * sqrt(tau_old / tau_new)`. Changing the discount factor
as well generally breaks this price-invariance identity. This follows from the
Black formula's separate standard-deviation and discount inputs, as documented
in [QuantLib's Black-formula interface](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/blackformula.hpp).
The stored surface
parameters contain only `business_days`, `implied_vols`, and `percent_strikes`;
they do not retain node-level actual expiry timestamps or pricing times.

## Observed audit results

| Check | Result |
| --- | --- |
| Raw observations / accepted by `passes_precalib_filter` | 79,060 / 65,108 |
| Accepted source window directions | 65,108 `backward` |
| Maximum absolute ACT/365 Black repricing error | 5.972111694e-12 price quote units |
| Maximum absolute discount-factor reproduction error | 1.776356839e-15 |
| News-to-pair workbook rows / current and target surface instances | 1,610 / 3,220 |
| Unique market pairs / unique endpoint reconstructions | 1,283 / 2,566 |
| Stored grid values checked | 824,320 |
| Maximum stored-grid versus BUS250 reconstruction error | 0 |
| Maximum BUS250 versus BUS365 reconstruction error | 1.110223025e-16 |
| Source JSON parameter matches / accepted raw snapshot links | 3,220 / 3,220 |
| Input files unchanged after audit | Yes |

Workbook current and target parameters match their source JSON entries exactly.
The corresponding raw-CSV key is the JSON direction's `snapshot_time_utc` plus
CSV direction `backward`, including for JSON `forward` entries. Looking up
the top-level JSON news timestamp plus direction `forward` would incorrectly
report missing raw data. Snapshot links verify accepted-source availability;
this check alone is not a replay of strike aggregation or quality-filter selection.

A separate replay groups accepted CSV observations by snapshot, window side,
business days, and strike, then applies volume-weighted median IV and
volume-weighted average percent strike. All 36,375 JSON node instances can be
compared, with no missing groups or point-count mismatches. At absolute
tolerance `1e-12`, 200 node instances differ in IV or percent strike. The maximum
IV difference is `0.00277151125466088`, and the maximum percent-strike difference
is `0.00030187415596161227`. Node instances can repeat when news rows share
snapshots. The cause of these differences has not been established; the audit
does not attribute them to the maturity clock. Frozen JSON parameters remain
the reference for reproducing historical training targets. The emitted audit
status is `verified_frozen_targets_with_clock_and_lineage_caveats`.

This limitation is consistent with the repository's existing frozen-parameter
policy in `scripts/rq3/market_jump_detection.py`: archived JSON parameters are
authoritative for historical surface reproduction, while the raw CSV provides
lineage checks. The broader market audit at
`outputs/rq3/atm_skew_jumps_20260810_final/validation_summary.json` also records
aggregation mismatches; its scope and counts differ from this chapter-target
audit. No CSV-based replacement labels are written here.

To quantify the clock difference, the audit also substitutes business-day
proxy times into Black-76 while holding each source IV and discount factor
fixed. These are hypothetical sensitivity calculations, not regenerated labels
or an estimate of historical option mispricing.

| Proxy variance clock | Mean proxy / actual time | Median ratio | Ratio range | Mean absolute price difference | Maximum absolute price difference |
| --- | --- | --- | --- | --- | --- |
| `q / 365` | 0.714341 | 0.703121 | 0.527140–1.013672 | 0.115311 | 0.419373 |
| `q / 250` | 1.042938 | 1.026556 | 0.769625–1.479960 | 0.017462 | 0.164608 |

Price differences are in the input option-price quote units, not dollars per
contract. The varying ratios demonstrate why a global denominator substitution
cannot recover actual elapsed pricing times. An economically unified-clock
experiment would require rebuilding the time mapping and grid from trade/expiry
lineage, rerunning affected training and diagnostics, and reporting it as a new
experiment. None of those changes is made here.

Regression tests cover a weekend clock mismatch, production-price
agreement for calls and puts, price-preserving IV reannualization, exclusion of
rejected source rows, non-finite/corrupted source values, forward snapshot
lineage, denominator cancellation, corrupted workbook targets/parameters,
source-row versus unique-pair counts, and explicit aggregation-lineage caveats.
They also check conflicting repeated workbook rows despite reconstruction
caching and unchanged input hashes. All twelve audit tests pass, together with
seven existing raw surface tests and three existing Black-76 tests (22 tests
in total). The full dataset audit also completes with unchanged input hashes.
LaTeX environment balance and the new equation labels were checked statically;
a full thesis PDF build was not run because no LaTeX compiler is available in
this environment.
