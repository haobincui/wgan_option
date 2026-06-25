# RQ2 Text Feature Literature Notes

This note summarizes the literature basis for the RQ2 text-feature comparison:

```text
no-text vs n-gram BoW frequency vs Sun-style LLaMA sentiment vs LLM embedding
```

The purpose is practical thesis support. These papers are used mainly to justify
feature families and comparison design. They are not treated as directly
comparable empirical results, because the asset class, target variable, horizon,
data source, and downstream model differ from this thesis.

## 1. How This Maps to the Thesis

The thesis question is:

```text
Can news text embeddings help forecast short-horizon changes in bond-option
volatility structures?
```

For RQ2, the comparison should isolate the text representation while keeping the
downstream forecasting setup fixed. In this repository, the intended comparison is:

| Thesis baseline | Literature role | Repository implementation |
| --- | --- | --- |
| no-text | Control with same downstream FiLM WGAN architecture | `text_embedding_mode=none` |
| n-gram BoW frequency | Traditional sparse text / news-frequency representation | `text_embedding_mode=bow` |
| Sun-style LLaMA sentiment | Multi-dimensional LLM sentiment decomposition for volatility | `text_embedding_mode=llm_sentiment` reads `sentiment_embedding` |
| LLM embedding | Main proposed representation | `text_embedding_mode=lp`, `hd`, or `concat` depending on experiment |

Implementation note: `llm_sentiment` now follows the Sun (2026)-style
zero-shot LLaMA 3 decomposition. It scores each article on
`macroeconomic_uncertainty`, `institutional_action`, and
`risk_off_intensity`, then pads those three scores into the existing
`sentiment_embedding` interface for FiLM WGAN.

## 2. BoW / N-Gram Frequency Literature

### Manela and Moreira (2017)

**Citation**

Manela, A., & Moreira, A. (2017). *News implied volatility and disaster
concerns*. Journal of Financial Economics, 123(1), 137-162.
DOI: https://doi.org/10.1016/j.jfineco.2016.01.032

Useful source pages:

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2382197
- RePEc: https://ideas.repec.org/a/eee/jfinec/v123y2017i1p137-162.html
- Published PDF mirror: https://amoreira2.github.io/alan-moreira.github.io/NVIX_published.pdf

**What they do**

Manela and Moreira construct a text-based uncertainty / volatility measure,
`NVIX`, from Wall Street Journal front-page articles. Their text representation
uses high-dimensional n-gram frequency features. These n-gram frequencies are
used in a supervised model to fit option-implied volatility and extend a
news-implied volatility series back to 1890.

The relevant methodological idea for this thesis is not their full `NVIX`
construction. The relevant idea is:

```text
financial news -> n-gram frequency representation -> volatility-related target
```

**Key empirical message**

Their `NVIX` peaks during market crashes, wars, financial crises, and periods of
policy uncertainty. In post-war U.S. data, high `NVIX` is followed by above
average stock returns, even after controlling for contemporaneous and
forward-looking stock-market volatility measures.

**How to use it in this thesis**

Use this paper as the main justification for a traditional
`n-gram / bag-of-words frequency` volatility baseline. In this repo, the baseline
is intentionally adapted, not replicated:

```text
Dow Jones news LP text -> unigram/bigram vocabulary -> log1p count vector
```

The downstream model remains the same FiLM WGAN used by the LLM embedding
experiments. This gives a clean method-level comparison on the thesis dataset.

**Do not claim**

Do not claim that the thesis reproduces Manela and Moreira's `NVIX`, their
training sample, their asset universe, or their return-predictability results.
The thesis borrows the text representation logic and applies it to bond-option
volatility-surface forecasting.

### Kogan et al. (2009)

**Citation**

Kogan, S., Levin, D., Routledge, B. R., Sagi, J. S., & Smith, N. A. (2009).
*Predicting Risk from Financial Reports with Regression*. Proceedings of NAACL,
272-280.

Source:

- ACL Anthology: https://aclanthology.org/N09-1031/

**What they do**

Kogan et al. predict future stock-return volatility from SEC-mandated financial
reports. The paper is useful because it frames volatility prediction as a text
regression problem and shows that textual features can rival strong volatility
baselines such as past volatility.

**How to use it in this thesis**

Use this as a broader precedent for:

```text
financial text -> numerical text features -> future volatility / risk
```

It is less directly aligned than Manela and Moreira because the text source is
10-K filings, not news, and the target is stock-return volatility, not option
surface movement.

### Forecasting Option Returns with News

**Citation / source**

The available working-paper PDFs found in this environment discuss using news
features, including n-grams and tf-idf style adjustments, for option-return
prediction.

Example source:

- SSRN PDF mirror: https://papers.ssrn.com/sol3/Delivery.cfm/4964058.pdf?abstractid=4964058&mirid=1

**How to use it in this thesis**

This is useful as a related option-market text paper, but it should be treated
carefully. Option returns are not the same target as implied-volatility surface
forecasting. If cited, it should support the general point that news text has
been used in option-market prediction, not the exact design of the thesis model.

## 3. Dictionary Sentiment Literature

### Loughran and McDonald (2011)

**Citation**

Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks*. Journal of Finance, 66(1), 35-65.

Useful source pages:

- Master Dictionary page: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1331573

**What they do**

Loughran and McDonald show that general-purpose negative word lists misclassify
many financial-domain words. They develop finance-specific sentiment and
complexity word lists, including categories such as negative, positive,
uncertainty, litigious, constraining, and superfluous.

**How to use it in this thesis**

Use this as the main citation if a dictionary sentiment baseline is restored or
reported separately. The local reference file is:

```text
data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv
```

This baseline should be described as a finance-domain dictionary sentiment
representation, not as the current `llm_sentiment` pipeline.

**Do not claim**

Do not claim this baseline captures contextual sentiment like GPT/LLaMA/FinBERT.
It is a lexicon/count-based benchmark.

## 4. LLM / Modern Language-Model Sentiment and Volatility Literature

### Liu and Shi (2025)

**Citation**

Liu, T., & Shi, Y. (2025). *News sentiment and investment risk management:
Innovative evidence from the large language models*. Economics Letters, 247,
112124.
DOI: https://doi.org/10.1016/j.econlet.2024.112124

Useful source pages:

- RePEc: https://ideas.repec.org/a/eee/ecolet/v247y2025ics0165176524006086.html
- Macquarie record: https://researchers.mq.edu.au/en/publications/news-sentiment-and-investment-risk-management-innovative-evidence/

**What they do**

This paper studies whether news sentiment measured by GPT-4 helps explain
intraday stock-return volatility and volatility states for firms in the Dow
Jones Composite Average. It compares GPT-4-classified news sentiment with
RavenPack sentiment.

**Key empirical message**

Both negative and positive firm-specific and macroeconomic news significantly
affect intraday stock-return volatility. The paper argues that GPT-4 can provide
more accurate news sentiment classification than RavenPack in this setting.

**How to use it in this thesis**

This is one of the closest references for:

```text
LLM-derived financial news sentiment -> volatility dynamics
```

It supports the motivation for testing richer language-model-derived news
signals against more traditional baselines. It does not directly address
bond-option implied-volatility surfaces.

### Hashami and Maldonado (2025)

**Citation**

Hashami, R., & Maldonado, F. (2025). *Can News Predict the Direction of Oil
Price Volatility? A Language Model Approach with SHAP Explanations*. arXiv:
2508.20707.

Sources:

- arXiv abstract: https://arxiv.org/abs/2508.20707
- arXiv HTML: https://arxiv.org/html/2508.20707

**What they do**

This paper predicts the direction of Brent crude oil realized volatility using
news-derived features. It compares sentiment methods and embedding methods,
including FastText, FinBERT, Gemini, and LLaMA, against a HAR benchmark.

**Key empirical message**

The strongest result is not that the largest LLM always wins. Their reported
results show:

- raw news count is a strong sentiment-side predictor
- FastText is the strongest embedding method
- many sentiment scores do not consistently beat HAR
- SHAP can identify changing keyword drivers across market regimes

**How to use it in this thesis**

This paper is useful for justifying the RQ2 comparison design:

```text
simple text counts / BoW vs sentiment vs language-model embeddings
```

It also supports the thesis caveat that complex language models must be tested
empirically rather than assumed to dominate simpler text features.

### Sun (2026)

**Citation**

Sun, Y. (2026). *Zero-Shot Meets ZeroHedge: Multi-Dimensional LLM Sentiment
Decomposition of Contrarian Media for VIX Prediction*. SSRN working paper.
DOI: https://doi.org/10.2139/ssrn.6736563

Source:

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6736563

**What they do**

This working paper proposes a structured zero-shot LLM sentiment framework for
VIX forecasting from alternative financial media. The important design point is
multi-dimensional sentiment decomposition rather than a single positive/negative
score.

**How to use it in this thesis**

This is useful as a direct VIX / volatility reference for true LLM sentiment.
Because it is a recent working paper, cite it as supporting evidence rather than
as a settled benchmark.

If a true LLM sentiment pipeline is added later, this paper supports using a
multi-dimensional vector such as:

```text
polarity, uncertainty, risk-off intensity, macro relevance, policy relevance
```

### Parvini and Assa (2025)

**Citation**

Parvini, N., & Assa, H. (2025). *Textual Regression for Realized Volatility: A
Model for Long-Term Forecasting*. SSRN working paper.
DOI: https://doi.org/10.2139/ssrn.5136391

Source:

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5136391

**What they do**

This paper uses LLM-based textual regression models, including LLaMA, OPT, BERT,
and RoBERTa, to forecast realized volatility in agricultural commodity markets.
It compares text models with HAR, sentiment-based approaches, and news-count
benchmarks.

**How to use it in this thesis**

This is not exactly LLM sentiment; it is closer to LLM text-to-volatility
forecasting. It is useful for arguing that language models can be evaluated
against traditional sentiment and count benchmarks in volatility forecasting.

### Kong et al. (2025)

**Citation**

Kong, Y., Hwang, Y., Kaiser, M., Vryonides, C., Oomen, R., & Zohren, S. (2025).
*Fusing Narrative Semantics for Financial Volatility Forecasting*. ICAIF 2025 /
arXiv:2510.20699.

Sources:

- ACM: https://dl.acm.org/doi/10.1145/3768292.3771256
- arXiv: https://arxiv.org/abs/2510.20699

**What they do**

This paper proposes a multimodal volatility forecasting framework that combines
market features with news embeddings generated by a point-in-time LLM. The
point-in-time design is important because it directly addresses look-ahead bias.

**How to use it in this thesis**

This is more relevant to LLM embeddings than LLM sentiment. It supports the
general thesis direction of combining market states with unstructured news
representations for volatility forecasting, while preserving temporal integrity.

## 5. Recommended Thesis Positioning

A defensible RQ2 paragraph can use the literature as follows:

```text
Traditional text-volatility work has used sparse n-gram or bag-of-words
representations to extract volatility-related information from news, most
notably Manela and Moreira's news-implied volatility framework. Finance-domain
dictionary approaches such as Loughran and McDonald provide a separate
sentiment-based benchmark. More recent work uses large language models to
extract sentiment or semantic representations for volatility prediction, but the
evidence does not imply that larger or more complex language models always
dominate simpler representations. Therefore, this chapter compares no-text,
n-gram frequency BoW, dictionary sentiment, and LLM embeddings under the same
bond-option volatility-surface forecasting pipeline.
```

## 6. Implementation Notes for This Repository

Current intended artifacts:

```text
data/processed/text_features/rq2/<timestamp>/bow_features.xlsx
data/processed/text_features/rq2/<timestamp>/llm_sentiment_features.xlsx
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

Current intended training scripts:

```bash
./run_film_wgan_bow.sh
./run_film_wgan_llm_sentiment.sh
```

Current output roots:

```text
outputs/training/film_wgan/bow
outputs/training/film_wgan/llm_sentiment
```

For thesis wording, use:

- `n-gram frequency BoW` for the BoW baseline
- `Sun-style LLaMA sentiment` for the current `llm_sentiment` baseline
- `LLM embedding` for the OpenAI embedding baseline
- `Loughran-McDonald sentiment` only if a separate dictionary baseline is
  restored or reported from older artifacts

## 7. Caveats

- None of the above papers directly forecasts this thesis target:
  short-horizon changes in bond-option volatility surfaces.
- Results from oil realized volatility, equity return volatility, VIX, and
  option returns should not be ranked against this thesis's surface metrics.
- The literature supports the experimental design and baseline selection; the
  thesis's empirical claims must come from the repository's own experiments.
- Recent 2025-2026 LLM papers may be working papers or preprints; mark them as
  such in the bibliography if they are not peer reviewed.

最后修改日期：2026-06-25
