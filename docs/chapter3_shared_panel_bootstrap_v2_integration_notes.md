# Chapter 3 shared-panel bootstrap v2 integration notes

Status: the approved v2 reporting contract was merged into
`docs/chapter3.tex`. The machine-readable bindings record that earlier
verified revision; the current chapter has a later Conclusion revision whose
binding status is described at the end of this note.

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

The final binding check covers 103 binding groups and 539 numerical/display
atoms across 11 v2-bound tables, all 15 formal jobs, and the principal repeated
prose and conclusion values. The six frozen training-diagnostic/RQ4 table
hashes remain unchanged. Two active descriptive-table records are verified
outside the v2 inference archive: alignment-window coverage and the F4
model-capacity robustness table. The F4 record explicitly records that it
supersedes the former capacity-table label
`tab:ch3:legacy_full_wgan_capacity_vs_fixed_pure_cnn`; the verifier
requires that old label to be absent from the current chapter rather than
falsely certifying the replaced table as unchanged.

The binding generator's `--check` independently reconstructs the explicit
source mappings. After the F4 capacity subsection was rewritten and the
frozen table retained verbatim, the Chapter 3 source digest was:

- `docs/chapter3.tex`:
  `f758005d823e54a366dc08919801907a5fad3f1d61997910d9e7a45fa55b6aa1`;
- `docs/chapter3_bootstrap_bindings.json`:
  `1bd1fe0e6482a8e47842571029fbc05d9906144ee43954a57d012e7139a641e6`.

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
- The active F4 model-capacity table is reproduced from the isolated experiment
  root
  `outputs/experiments/rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1/`.
  Its frozen 5,148-row analysis panel is
  `analysis/f4_pair_metrics.csv.gz` (SHA-256
  `7eb7b391209a742ce6478867d09e2d2390d65c2f2f6e5a97493ebc2e6fcfd5a2`),
  derived from the frozen evaluation evidence with SHA-256
  `8c30ab2a74c6ecb0a5bd40d1aec113afc95b04eda3670cfc4fc5353482621760`.
  The summary CSV, summary JSON, and complete LaTeX table have respective
  SHA-256 digests
  `7aeaa83c8da6714c14fb92e23853fcfe887114d17407afa5433abc15d040ea3e`,
  `460c9161ad7eaf855af3ed440676f8c17467baaea94466817da2323714498555`,
  and
  `79ac7c4fb8e57990adc96e2aac37757f4ad14554331cacc49c01a54afdc581d4`.
  Terminal QA is frozen at `analysis/f4_capacity_qa.json` (SHA-256
  `da6c827be95f16ab9012b079466d53f2dd503069b64e1dafe4ce6d6491cafbd7`)
  and records `status=passed`, 36 training jobs, 72 checkpoint rows, 36
  prediction cells, and the same 5,148 pair-metric rows.
  The c32 regression-control record is
  `analysis/f4_c32_regression_control.json` (SHA-256
  `c200e68d3f5f625365192cb054655e488fff717e671ed75081435658c18b6589`).
  Both architecture-level c32 MAEs pass the frozen relative tolerance of
  `5e-5`, and their 143-pair / 45-session panels match the immutable anchors.
  The 12 old/new Generator and Critic checkpoint hashes are not bitwise
  identical, so this control explicitly makes no exact-determinism claim.
  The matrix is six capacities by two architectures by three seeds by 143 F4
  test pairs. MAE is averaged within seed and then equally across seeds;
  improvement is calculated from unrounded means as
  `100*(1-mae_capacity/mae_c32)` within each architecture. Q3 supplies only
  validation/checkpoint selection and Q4 is the held-out test panel, not
  training data. The earlier Q3/Q4 capacity experiment remains read-only
  provenance, but its
  `tab:ch3:legacy_full_wgan_capacity_vs_fixed_pure_cnn` table is superseded and
  is no longer a source for the active chapter.
- Alignment coverage counts `500 / 588 / 634 / 673 / 721` reconcile to
  `inputs/pair_universe_summary.csv` (SHA-256
  `4619626efd7df93744577aa12824acd6a2060dccb90286e2afdf8e89a9dea72a`).
  The raw pair universe has SHA-256
  `98393abd4bf56875c9c9e8c6ef854f86b3ce5fd8fc480087a0651f40ba518570`,
  and the 46,044-row pair-metric source has SHA-256
  `c75fb74aa1504d43f6f8cd0926fc31559c3e6f9bf15d56ec0f40bf4a4980cb4a`.

## Static and rendered-document QA boundary

Static source QA after the F4 table integration reports 129 unique labels,
balanced LaTeX table and tabular environments, no undefined internal
references, resolved bibliography keys, and no duplicate bibliography keys.
The exact generated capacity-table block matches the block in
`docs/chapter3.tex`. The final verification commands are:

```bash
python scripts/rq3/main.py train-news-first-vol-f4-film-pure-capacity-3seed qa --resume
python -m unittest tests.test_scripts.test_rq3_news_first_vol_f4_film_pure_capacity_3seed -v
python -m unittest tests.test_scripts.test_rq3_news_first_vol_f4_film_pure_capacity_3seed_analysis -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests/test_scripts -p 'test_*chapter3*bootstrap*.py' -v
python -m scripts.rq123.build_chapter3_bootstrap_bindings --write
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

## Subsequent Conclusion revision

The Conclusion was rewritten to reflect the F4-only capacity experiment and
the current alignment-window reporting. The current `docs/chapter3.tex` SHA-256
is `5f59aa80713bf4a135dd367a88adfb21ce09d77b3a8ae3218fd5d6a15dda86a4`.
The chapter's alignment-coverage table label was also corrected to match its
existing reference; no numerical table cell was changed as part of this
revision. The earlier chapter digest and binding totals above describe the
previous verified revision, not the current source file.

Current binding regeneration is pending reconciliation of pre-existing RQ1
table drift. The table's four Panel C standard-error ratios are
`-0.89 / -1.92 / 0.74 / -0.38`, whereas the frozen v2 contrast records render
`-0.39 / -1.36 / 0.41 / -0.29`. The current binding builder also still
expects the former RQ1 panel layout. These differences were not introduced
by the Conclusion rewrite, and neither the table values nor frozen numerical
archive were changed to force a passing check.
