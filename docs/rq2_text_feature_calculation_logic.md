# RQ2 Text Feature Calculation Logic

本文档说明 RQ2 中两个传统或中间层 text baseline 的计算逻辑：

- `bow`: n-gram / bag-of-words style frequency features
- `llm_sentiment`: OpenAI ChatGPT Sun-style multi-dimensional sentiment features

这两个 baseline 的目标不是单独训练文本模型，而是把同一批 news text 转换成 FiLM WGAN 可以读取的固定宽度向量。后续比较固定在同一数据、同一 split、同一 volatility surface 下游模型中进行：

```text
no-text vs n-gram frequency BoW vs OpenAI ChatGPT sentiment vs LLM embedding
```

当前实现中，feature generation 只读取 `data/raw/text_embedding/news_with_openai_embeddings_large.xlsx`，不会修改 `data/raw/`。

## 1. 总体位置 / Overall Role

RQ2 的实验问题是比较不同层级的 news text representation 在同一 FiLM WGAN 下游中的预测贡献。当前 pipeline 是：

```text
raw news workbook
  -> BoW / llm_sentiment feature generation
  -> feature workbooks under data/processed/text_features/rq2/<timestamp>/
  -> enrich merged_vol.xlsx
  -> merged_vol_rq2_text.xlsx
  -> FiLM WGAN training with text_embedding_mode=bow or text_embedding_mode=llm_sentiment
```

这里的核心约束是：不同 text representation 只改变输入文本向量，不改变 volatility data、train/validation/test split 或 FiLM WGAN 下游训练逻辑。这样 RQ2 的结果可以解释为 text representation 的差异，而不是模型架构或数据切分差异。

## 2. BoW: n-gram frequency features

`bow` 是传统文本表示 baseline。当前实现位于 `src/bow/features.py`，命令入口是 `scripts/bow/main.py` 或综合入口 `scripts/generate_rq2_text_features.py`。

### 输入

默认输入是：

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

默认文本列是：

```text
LP
```

如果原始 workbook 中存在这些列，输出会保留用于追踪：

```text
news_row_id
ArticleID / article_id
SourceFile / source_file
```

### 计算逻辑

当前 BoW 不使用 TF-IDF，也不使用 SVD。它是 Manela and Moreira-style 的 n-gram / bag-of-words frequency representation，只复用其传统文本表示思想，不复现他们论文中的 NVIX/SVR 下游模型。

具体步骤：

1. 对每条新闻的 `LP` 文本做简单英文 tokenization。
2. token 转为 lower-case。
3. 默认生成 unigram 和 bigram，即 `ngram_range=(1, 2)`。
4. 在整个 news corpus 中统计所有 n-gram 的出现频率。
5. 选择 corpus 中频率最高的 `target_dim=1024` 个 n-grams 作为固定 vocabulary。
6. 对每条新闻生成长度为 `1024` 的向量。
7. 第 `j` 个向量元素对应 vocabulary 中第 `j` 个 n-gram，在该新闻中出现次数为 `ngram_count` 时，使用 `log(1 + ngram_count)` 作为该维度的值：

```text
bow_feature_j = log(1 + ngram_count)
```

如果某个 n-gram 不在该新闻中出现，对应位置为 `0.0`。如果新闻文本为空，则整条 `bow_embedding` 是全零向量。

### 输出

BoW feature workbook 写入：

```text
data/processed/text_features/rq2/<timestamp>/bow_features.xlsx
```

主要列包括：

```text
news_row_id
article_id
source_file
bow_embedding
bow_dim
```

其中：

- `bow_embedding` 是 JSON serialized list，长度为 `bow_dim`。
- `bow_dim` 默认是 `1024`。

同时会输出：

```text
bow_manifest.json
bow_vocabulary.json
```

`bow_manifest.json` 中关键字段包括：

```text
representation = ngram_frequency
weighting = log1p_count
reference_method = Manela_Moreira_2017_style_ngram_frequency
```

`bow_vocabulary.json` 是最终使用的 n-gram vocabulary，决定了 `bow_embedding` 每个维度的含义。

## 3. llm_sentiment: OpenAI ChatGPT Sun-style sentiment decomposition

`llm_sentiment` 是 LLM-style sentiment baseline。当前实现位于 `src/llm_sentiment/features.py`，命令入口是 `scripts/llm_sentiment/main.py` 或综合入口 `scripts/generate_rq2_text_features.py`。

当前实现使用 OpenAI ChatGPT API，不使用 HuggingFace LLaMA，也不是 Loughran-McDonald dictionary。API key 通过环境变量传入：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

默认模型来自代码中的 `DEFAULT_MODEL_ID`，也可以通过 `--model` 或环境变量 `OPENAI_MODEL` 覆盖。

### 输入

默认输入仍然是：

```text
data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
```

默认文本列仍然是：

```text
LP
```

每条新闻最多发送 `--max-input-chars` 个字符给模型，默认是 `6000`。

### Prompt 逻辑

Prompt 要求 ChatGPT 按 Sun-style zero-shot financial news sentiment decomposition 给新闻打分。三个维度固定为：

```text
macroeconomic_uncertainty
institutional_action
risk_off_intensity
```

语义解释：

- `macroeconomic_uncertainty`: 新闻中是否包含宏观不确定性、经济压力、政策不确定性、增长或通胀风险等信息。
- `institutional_action`: 新闻中是否包含央行、财政部门、监管机构或大型机构行动相关信息。
- `risk_off_intensity`: 新闻中是否反映市场避险、风险资产抛售、流动性压力或波动风险上升。

Prompt 明确要求模型只根据新闻中的事实金融内容评分，不机械继承作者的情绪化措辞、标题党风格或一般 bearish tone。模型被要求只返回 JSON：

```json
{
  "macroeconomic_uncertainty": 0.0,
  "institutional_action": 0.0,
  "risk_off_intensity": 0.0
}
```

每个分数都应该在 `[0, 1]` 之间。

### Response parsing

解析逻辑分两层：

1. 优先解析 JSON object 或 fenced JSON block。
2. 如果不是严格 JSON，则用 regex fallback 从文本中提取三个 key 对应的数值。

解析出的分数会被 clamp 到 `[0.0, 1.0]`。如果 response 无法解析，或者文本为空，则使用全零 fallback vector，并在 `sentiment_parse_status` 中记录状态，例如：

```text
json
regex
parse_failed
empty_text
cache_json
api_error
```

### 向量化逻辑

`llm_sentiment` 最终也输出固定宽度向量，默认 `target_dim=1024`。前三维固定为：

```text
0: macroeconomic_uncertainty
1: institutional_action
2: risk_off_intensity
```

其余维度全部补零：

```text
sentiment_embedding = [
  macroeconomic_uncertainty,
  institutional_action,
  risk_off_intensity,
  0.0,
  0.0,
  ...
]
```

这样做的目的是保持 FiLM WGAN 的 text input interface 和其他 text representation 一致，同时让 sentiment baseline 的有效信息只来自三个可解释维度。

### Cache / resume

OpenAI sentiment generation 是长任务，当前实现支持中断后继续。默认 cache 文件是：

```text
data/processed/text_features/rq2/<timestamp>/openai_sentiment_cache.jsonl
```

每条成功 API response 会立即写入 JSONL cache。若请求 timeout 或任务中断，重新运行同一个命令并使用同一个 `--output-dir` 或同一个 `--cache-path`，已完成的新闻会从 cache 命中，不会重复调用 API。

默认行为是：请求失败并超过 retry 次数后停止任务，但已完成行已经保存在 cache 中。也可以使用 `--continue-on-error` 让失败行写入全零 `api_error` 向量后继续。

### 输出

LLM sentiment feature workbook 写入：

```text
data/processed/text_features/rq2/<timestamp>/llm_sentiment_features.xlsx
```

主要列包括：

```text
news_row_id
article_id
source_file
sentiment_embedding
sentiment_dim
sentiment_dictionary_source
sentiment_model_id
sentiment_prompt_version
sentiment_parse_status
sentiment_raw_response
```

其中：

- `sentiment_embedding` 是 JSON serialized list，默认长度为 `1024`。
- `sentiment_dim` 默认是 `1024`。
- `sentiment_dictionary_source` 是兼容旧字段名，当前值类似 `openai_chatgpt_sun2026_style:<model_id>`，不是 dictionary source。
- `sentiment_raw_response` 保存原始模型回复，便于审计和排查解析问题。

同时会输出：

```text
llm_sentiment_manifest.json
openai_sentiment_cache.jsonl
```

manifest 中关键字段包括：

```text
representation = sun2026_style_openai_chatgpt_multidimensional_sentiment
prompt_version = sun2026_zero_shot_chatgpt_v1
base_feature_dim = 3
sentiment_dimensions = [
  macroeconomic_uncertainty,
  institutional_action,
  risk_off_intensity
]
```

## 4. 输出 artifacts 与 FiLM WGAN 对接

BoW 和 `llm_sentiment` 都先生成 article-level feature workbooks，然后通过 enrich step 合并进 paired volatility workbook。

原始 paired vol workbook 是：

```text
data/processed/svi-excel/20260410-174929/merged_vol.xlsx
```

RQ2 enriched workbook 是：

```text
data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

`scripts/rq2/enrich_merged_vol.py` 会保留原 workbook sheets，并把 feature columns 合并到相关 sheet。对 `gan_input_ready` sheet，必须能按 `news_row_id` 匹配到对应 feature，否则会报错。

合并后，FiLM WGAN 通过以下模式读取不同文本向量：

```text
text_embedding_mode=bow
text_embedding_mode=llm_sentiment
```

对应读取：

```text
bow_embedding
sentiment_embedding
```

训练输出目录固定为：

```text
outputs/training/film_wgan/bow
outputs/training/film_wgan/llm_sentiment
```

## 5. 复现实验命令

### 单独生成 BoW features

```bash
RQ2_TS=$(date -u +%Y%m%d-%H%M%S)
FEATURE_DIR=data/processed/text_features/rq2/${RQ2_TS}

python scripts/bow/main.py \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-dir "${FEATURE_DIR}" \
  --text-column LP \
  --target-dim 1024 \
  --ngram-min 1 \
  --ngram-max 2
```

### 单独生成 llm_sentiment features

```bash
export OPENAI_API_KEY="your_api_key_here"

RQ2_TS=$(date -u +%Y%m%d-%H%M%S)
FEATURE_DIR=data/processed/text_features/rq2/${RQ2_TS}

python scripts/llm_sentiment/main.py \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-dir "${FEATURE_DIR}" \
  --text-column LP \
  --target-dim 1024 \
  --model gpt-5.5 \
  --max-input-chars 6000 \
  --max-output-tokens 256 \
  --reasoning-effort low \
  --max-retries 5 \
  --retry-backoff-seconds 5 \
  --progress-every 100
```

如果上面的命令因 timeout 中断，使用同一个 `FEATURE_DIR` 重新运行即可从 `openai_sentiment_cache.jsonl` 继续。

### 同时生成 BoW 和 llm_sentiment features

```bash
export OPENAI_API_KEY="your_api_key_here"

RQ2_TS=$(date -u +%Y%m%d-%H%M%S)
FEATURE_DIR=data/processed/text_features/rq2/${RQ2_TS}

python scripts/generate_rq2_text_features.py \
  --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
  --output-dir "${FEATURE_DIR}" \
  --text-column LP \
  --target-dim 1024 \
  --ngram-min 1 \
  --ngram-max 2 \
  --model gpt-5.5 \
  --max-input-chars 6000 \
  --max-output-tokens 256 \
  --reasoning-effort low \
  --max-retries 5 \
  --retry-backoff-seconds 5 \
  --progress-every 100
```

### Enrich merged_vol workbook

```bash
python scripts/rq2/enrich_merged_vol.py \
  --merged-vol data/processed/svi-excel/20260410-174929/merged_vol.xlsx \
  --bow-features "${FEATURE_DIR}/bow_features.xlsx" \
  --sentiment-features "${FEATURE_DIR}/llm_sentiment_features.xlsx" \
  --output data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx
```

### Train FiLM WGAN with BoW

```bash
bash scripts/rq2/run_film_wgan_bow.sh
```

可追加透传参数，例如：

```bash
bash scripts/rq2/run_film_wgan_bow.sh --set seed=43
```

### Train FiLM WGAN with llm_sentiment

```bash
bash scripts/rq2/run_film_wgan_llm_sentiment.sh
```

可追加透传参数，例如：

```bash
bash scripts/rq2/run_film_wgan_llm_sentiment.sh --set batch_size=64
```

## 6. Caveats / 注意事项

- `bow` 当前是 n-gram frequency baseline，不是 TF-IDF，也不是 TF-IDF + SVD。
- `llm_sentiment` 当前是 OpenAI ChatGPT baseline，不是 HuggingFace LLaMA，也不是 Loughran-McDonald dictionary。
- `sentiment_dictionary_source` 是为了兼容 workbook schema 保留的列名；当前它记录的是 OpenAI ChatGPT sentiment source。
- BoW 和 `llm_sentiment` 都是 article-level text representations，最终通过 `news_row_id` 合并进 `merged_vol_rq2_text.xlsx`。
- `merged_vol_rq2_text.xlsx` 是训练输入 artifact；不要把 feature generation 产物放进 `outputs/training/`。
- `data/raw/` 是只读输入区。生成 BoW、`llm_sentiment`、enriched workbook 都应写入 `data/processed/`。
- RQ2 的比较必须在相同 volatility workbook、相同 split、相同 FiLM WGAN 下游和相同 metrics 下解释。不要把这些结果和旧 `val_recon`、short-ATM 或其他不可比 metric 直接混排。
