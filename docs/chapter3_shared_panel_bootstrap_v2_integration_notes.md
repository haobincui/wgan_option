# Chapter 3 shared-panel bootstrap v2 integration notes

Status: the approved v2 reporting contract has been merged into
`docs/chapter3.tex`; the final machine-readable bindings are complete and
verified in `docs/chapter3_bootstrap_bindings.json`.

This note is an integration and audit record, not a source of reported results.
The authoritative numerical archive is
`outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2`. RQ4 is
outside the v2 correction: its historically frozen results and resampling
method were not recomputed.

## Final reporting contract

The approved merge resolves the earlier concurrent-edit conflict as follows.

1. The main direct RQ1--RQ2 and common-parent RQ3 contrasts retain their
   original equal-seed, equal-fold (`equal_cell`) point estimands. Their three
   historical summary tables continue to display `bootstrap_mean_mae`, now
   refreshed from formal v2 and explicitly labelled as means across bootstrap
   draws. Arithmetic improvements and normalizations in those tables are
   computed from the corresponding unrounded arm draw means. These descriptive
   columns are not original-sample point estimates and are not used to
   reconstruct the paired log-ratio contrast.
2. Paired contrast points, geometric gains, bootstrap standard errors,
   percentile intervals, sign-tail probabilities, Holm adjustments, and stars
   come from the formal contrast records. In particular, contrast points and
   geometric gains remain original-sample quantities even when shown beside an
   arm-level bootstrap-mean column.
3. Generator-architecture and alignment-window robustness retain their
   historical `pooled_pair` point estimands. Their displayed MAEs are formal-v2
   `observed_mean_mae` values, with folds weighted by pair counts; their
   contrast point is the log ratio of the pooled original-sample means.
4. The RQ3 validation trajectory retains its original observed equal-cell
   definition. Its selected-checkpoint comparisons are explicitly
   post-selection optimization diagnostics, conditional on selection using the
   same validation data.
5. RQ4 keeps its frozen historical values and methodology and is explicitly
   excluded from the v2 correction.

This contract supersedes incompatible observed-only or robustness-equal-cell
reporting recommendations in the older
`docs/chapter3_four_fold_reporting_revision.md`. That file remains useful as a
historical audit record but is not the controlling Chapter 3 specification.

## Logic corrections retained in the merged chapter

- Non-significance is not described as equivalence, a zero effect, or proof of
  the reverse direction. This is especially important for positive contrasts
  with `p = 1` under the frozen focal-better alternative.
- The RQ3 epoch-0-to-30 tests compare changes within four continuation paths;
  they are not matched-minus-control difference-in-change tests. Their results
  therefore cannot establish a relative learning-speed effect. Comparisons
  involving a validation-selected checkpoint are conditional post-selection
  diagnostics.
- Near-equal aggregate MAE, including same-checkpoint replacement results, does
  not imply that pair-level predictions are identical or that text has zero
  information content.
- News timestamps proxy publication availability, not participant-level
  delivery latency; finer provider timestamp precision does not remove that
  limitation. Embeddings and LLM sentiment were constructed retrospectively,
  so chronological downstream splits alone do not establish an as-of-2023,
  hindsight-free encoder deployment.
- The dataset description now covers 2022--2023 raw inputs and all four 2023
  test quarters consistently. Contract terminology follows the CME 10-Year
  U.S. Treasury Note (`ZN`) specifications, distinguishes futures delivery from
  the maturity of deliverable bonds, and does not describe option expiry as a
  generic month-end rule. The older surface figures are identified as
  illustrations without unsupported event attribution. The official audit
  sources are the [Federal Reserve target-range history](https://www.federalreserve.gov/monetarypolicy/openmarket.htm),
  [June 2023 FOMC statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230614a.htm),
  [CME contract specifications](https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html),
  [CBOT Chapter 19](https://www.cmegroup.com/rulebook/CBOT/II/19.pdf), and
  [CBOT Chapter 19A](https://www.cmegroup.com/rulebook/CBOT/II/19A.pdf).
- Adversarial training is no longer presented as an architecture alternative to
  CNNs or LSTMs, because the Generator and Critic may themselves be
  convolutional. The original minimax GAN criterion is correctly associated
  with Jensen--Shannon divergence at the optimal discriminator, rather than a
  general KL-minimization claim; WGAN's optimization motivation is stated
  without a finite-model convergence or accuracy guarantee. Primary references
  are the [original GAN analysis](https://arxiv.org/abs/1406.2661) and the
  [Wasserstein GAN paper](https://arxiv.org/abs/1701.07875).

## Corrected resampling contract

For every formal-v2 draw, seed multiplicities are sampled once. Overall
analyses also sample four fold occurrences once; quarter-specific analyses hold
the named fold fixed. Each sampled fold occurrence receives its own resample of
whole CME-session clusters, retaining every eligible pair in a selected
session. Duplicate occurrences of the same fold receive independent session
resamples. The resulting fold-occurrence/session schedule is reused across
every sampled seed and every model arm or reference condition in the
comparison. This crosses algorithmic seed variation with one shared resample of
the economic market panel instead of independently resampling the market inside
each seed.

The chapter labels 95% intervals as unadjusted percentile intervals and the
reported directional tail areas as +1-corrected approximate bootstrap
sign-tail probabilities. The displayed
`theta / bootstrap standard error` quantity is a descriptive standard-error
ratio, not a studentized bootstrap statistic or Student-t test. The percentile
intervals and one-sided, sometimes Holm-adjusted probabilities are not exact
inferential duals. A probability of 1 means no support for the frozen
focal-better direction, not evidence of a zero effect or a test of the reverse
direction. Holm adjustment is local to the stated frozen families, not global
across the chapter.

The crossed-factor construction is motivated by Owen--Eckles-style dependence
concerns, but it supplies no exact finite-sample guarantee for nonlinear,
session-clustered MAE and log-ratio statistics. Sessions may remain dependent
through time; ten model seeds share the same economic observations; and the
four 2023 test quarters were inspected during development. The chapter
therefore treats the v2 uncertainty summaries as retrospective exploratory
evidence rather than confirmatory inference.

## Formal archive and replay

Independent `--verify-only` replay passed for 15 jobs, 93 contrasts, and 10,000
draws per comparison. `qa.json` records 111 arm summaries, preservation checks
for all 73 reproducible historical point estimates, unchanged input hashes, and
`rq4_recomputed = false`. The formal hash-registry digest is
`a8ad25739e50e3476ddf992669306a17d19778381f08c9f5fb780e46ef227415`.

Key artifact SHA-256 digests are:

- `chapter3_values.json`:
  `969ac9a3c4a426c490b797df6636a086b3a1a56c8df3b4220dab16e15246ff88`;
- `analysis/all_arm_summary.csv`:
  `89f0f669b2b72e247f8e5d44caac686d5e6bf101a229fc9e0ff5ee184a3c75e6`;
- `analysis/all_contrasts.csv`:
  `a1827043152ad78ce3d23c85be7db6220c77d57eec61eacc4bc703af6ae346b2`;
- `analysis/new_old_comparison.csv`:
  `368c29e8d2c66acfc41467c127150ecef63f54ac2746e9621418c98d15eab7ad`.

The final binding check passes for 103 binding groups and 539 numerical/display
atoms across 11 tables, all 15 formal jobs, and the principal repeated prose
and conclusion values. The six frozen training-diagnostic/RQ4 table hashes
remain unchanged. The two externally revised descriptive-table records retain
their prior hashes and explicitly record the authorized, source-verified
updates; they are not falsely certified as unchanged.

The binding generator's `--check` also passes, independently reconstructing the
explicit source mappings. Final SHA-256 digests are:

- `docs/chapter3.tex`:
  `c503ab0ba2a1817fb22e29aab78b8b719c7f036e5f99518c805a63c550af3f65`;
- `docs/chapter3_bootstrap_bindings.json`:
  `7eeefb0e9ff6b9d90f06b53cc218ed558ca0b2eb4fc31e88222ba7ddc6c82512`.

## High-impact integrated values

The main paired-inference values retained in the chapter include: RQ1 overall
`theta = -1.882e-4`, descriptive standard-error ratio `-0.42`, and one-sided
`p = 0.3200`; RQ2 LP-versus-BoW `p_Holm = 0.3234` and
LP-versus-sentiment `p_Holm = 0.0612`; RQ3 matched-versus-shuffled
`p_Holm = 0.3174`, matched-versus-zero `p_Holm = 0.4768`, and both
same-checkpoint intervention probabilities equal to `1.0000`. All four RQ3
validation epoch-30-to-selected rows have `p_Holm = 0.3612` and remain
conditional post-selection diagnostics.

The merged architecture table uses pooled observed MAEs `0.0019944614` for
FiLM-CNN, `0.0019962782` for cross-attention, `0.0019984845` for the
Transformer-token model, and `0.0019963409` for StyleMod. The merged alignment
common-panel table likewise uses the formal pooled points, beginning with
FiLM/Pure `0.0019944614 / 0.0019951546` at five minutes. The five pooled
FiLM-versus-Pure geometric gains are `0.0347%`, `0.0073%`, `0.0003%`,
`-0.0269%`, and `-0.0100%`; none of the frozen one-sided Holm-5 comparisons
supports a FiLM advantage.

## Descriptive-table source audit

These checks reconcile displayed descriptive statistics to archived source
rows; they do not imply retraining or re-estimation of RQ4.

- The baseline and RQ2 training tables reconcile to the 200-row file
  `outputs/experiments/rq12_news_first_vol_text_10seed_composite_lr2p5e5_descriptive_no_bootstrap_exact_ttm_rolling_v1/analysis/text_10seed_composite_training_summary.csv`
  (SHA-256
  `17f15da3ad24f9c221f6297748a1800f7428858bef64b992315b5c742f64977b`).
- The RQ3 training table reconciles to the 280-row file
  `outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1/analysis/training_summary.csv`
  (SHA-256
  `af62a197d4a761f08810bd715a3085aac787d781a51b43198e02130a9e55c627`).
  All 280 referenced metrics files were also checked for existence, content
  hash, maximum epoch, and selected validation-best epoch.
- The historical capacity table reconciles to the Q3 and Q4 pair-metric files
  under
  `outputs/experiments/rq3_news_first_vol_film_nolp_capacity_seed_exact_ttm_v1/analysis/`
  with respective SHA-256 digests
  `393b9ba1095cc5899bdbe8504deddb5884c2f3d982f87f1b0223b228800d9222`
  and
  `6975426abd6178f64a25c6585272bf7cc3745e46bbcf54a6bbb3121a096370c6`.
  Reproduction requires the selector `panel=core`,
  `stratum_type=overall`, `stratum_value=all`, and
  `tolerance_minutes=5`. The source also contains 30-minute records, so this
  selector must not be omitted when reproducing the table.
- Alignment coverage counts `500 / 588 / 634 / 673 / 721` reconcile to
  `inputs/pair_universe_summary.csv` (SHA-256
  `4619626efd7df93744577aa12824acd6a2060dccb90286e2afdf8e89a9dea72a`).
  The raw pair universe has SHA-256
  `98393abd4bf56875c9c9e8c6ef854f86b3ce5fd8fc480087a0651f40ba518570`,
  and the 46,044-row pair-metric source has SHA-256
  `c75fb74aa1504d43f6f8cd0926fc31559c3e6f9bf15d56ec0f40bf4a4980cb4a`.

## Static and rendered-document QA boundary

Static source QA reports 129 unique labels, balanced LaTeX environments, no
undefined internal references, resolved new bibliography keys, and no duplicate
bibliography keys. All 47 targeted tests, scoped Ruff checks, whitespace checks,
and the formal replay pass. The final verification commands are:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests/test_scripts -p 'test_*chapter3*bootstrap*.py' -v
python -m scripts.rq123.build_chapter3_bootstrap_bindings --check
python -m scripts.rq123.verify_chapter3_bootstrap_bindings
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.rq123.chapter3_shared_panel_bootstrap_v2 --verify-only
```

No TeX engine (`pdflatex`, `latexmk`, `tectonic`, or `lualatex`) is installed in
the current environment, so a final PDF build and visual inspection could not
be performed here. Three older `vol_example` figure files are unavailable at
their declared thesis paths. The ATM-jump PNG exists under `docs/figures`, but
the chapter source refers to the master-thesis `Chapter3/Chapter3Figs` layout.
These asset/path limitations affect rendered-document QA, not the formal-v2
numerical reconciliation.
