"""Build Sun-style LLaMA sentiment features for RQ2 text baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT_VERSION = "sun2026_zero_shot_v1"
SENTIMENT_DIMENSIONS = (
    "macroeconomic_uncertainty",
    "institutional_action",
    "risk_off_intensity",
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})")


class SentimentBackend(Protocol):
    """Minimal generation interface used by the feature builder and tests."""

    def generate(self, prompt: str) -> str:
        """Generate a raw model response for one prompt."""


@dataclass(frozen=True)
class ParsedSentiment:
    scores: dict[str, float]
    parse_status: str


@dataclass(frozen=True)
class SentimentFeatureResult:
    """Feature frame plus metadata describing the LLaMA sentiment source."""

    frame: pd.DataFrame
    manifest: dict[str, Any]


class HuggingFaceLlamaSentimentBackend:
    """Lazy HuggingFace text-generation backend for LLaMA-style instruct models."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.torch_dtype = str(torch_dtype)
        self.max_new_tokens = int(max_new_tokens)

        try:
            import torch
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps at runtime.
            raise ImportError(
                "LLaMA sentiment generation requires transformers, accelerate, "
                "huggingface_hub, safetensors, sentencepiece, and protobuf. "
                "Install project requirements before running the real HuggingFace backend."
            ) from exc

        token = os.environ.get("HF_TOKEN") or None
        dtype = None
        if self.torch_dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if self.torch_dtype not in dtype_map:
                raise ValueError(
                    "torch_dtype must be one of ['auto', 'float16', 'bfloat16', 'float32'], "
                    f"got: {self.torch_dtype}"
                )
            dtype = dtype_map[self.torch_dtype]

        tokenizer_kwargs: dict[str, Any] = {}
        pipeline_kwargs: dict[str, Any] = {"model": self.model_id}
        if token is not None:
            tokenizer_kwargs["token"] = token
            pipeline_kwargs["token"] = token
        if dtype is not None:
            pipeline_kwargs["torch_dtype"] = dtype

        if self.device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        elif self.device == "cpu":
            pipeline_kwargs["device"] = -1
        elif self.device == "cuda":
            pipeline_kwargs["device"] = 0
        else:
            raise ValueError("device must be one of ['auto', 'cpu', 'cuda'], got: " + self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        self.generator = pipeline("text-generation", tokenizer=self.tokenizer, **pipeline_kwargs)

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        formatted = self._format_prompt(prompt)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        outputs = self.generator(
            formatted,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
            pad_token_id=eos_token_id,
        )
        if not outputs:
            return ""
        first = outputs[0]
        if isinstance(first, dict):
            return str(first.get("generated_text", ""))
        return str(first)


def _serialize_vector(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values.tolist()], ensure_ascii=True)


def _ensure_news_row_id(frame: pd.DataFrame) -> pd.Series:
    if "news_row_id" in frame.columns:
        return pd.to_numeric(frame["news_row_id"], errors="coerce").fillna(0).astype(int)
    return pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="int64")


def _metadata_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _text_series(frame: pd.DataFrame, text_column: str) -> pd.Series:
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column '{text_column}'. Available columns: {list(frame.columns)}")
    return frame[text_column].fillna("").astype(str)


def _empty_scores() -> dict[str, float]:
    return {dimension: 0.0 for dimension in SENTIMENT_DIMENSIONS}


def _clamp_score(value: Any) -> float:
    score = float(value)
    return float(min(1.0, max(0.0, score)))


def _normalize_score_object(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    scores: dict[str, float] = {}
    for dimension in SENTIMENT_DIMENSIONS:
        if dimension not in value:
            return None
        try:
            scores[dimension] = _clamp_score(value[dimension])
        except (TypeError, ValueError):
            return None
    return scores


def _json_candidates(raw_response: str) -> list[str]:
    text = str(raw_response).strip()
    candidates = [text]
    for match in JSON_BLOCK_RE.finditer(text):
        candidate = match.group(1) or match.group(2)
        if candidate:
            candidates.append(candidate.strip())
    return candidates


def _regex_score(raw_response: str, key: str) -> float | None:
    pattern = re.compile(rf'"?{re.escape(key)}"?\s*[:=]\s*(-?\d+(?:\.\d+)?)(?:\s*(%))?', re.I)
    match = pattern.search(raw_response)
    if not match:
        return None
    try:
        value = float(match.group(1))
        if match.group(2) == "%":
            value = value / 100.0
        return _clamp_score(value)
    except ValueError:
        return None


def parse_llama_sentiment_response(raw_response: str) -> ParsedSentiment:
    """Parse Sun-style three-dimensional scores from a LLaMA response."""

    text = str(raw_response or "").strip()
    if not text:
        return ParsedSentiment(scores=_empty_scores(), parse_status="parse_failed")

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        scores = _normalize_score_object(parsed)
        if scores is not None:
            return ParsedSentiment(scores=scores, parse_status="json")

    regex_scores: dict[str, float] = {}
    for dimension in SENTIMENT_DIMENSIONS:
        score = _regex_score(text, dimension)
        if score is None:
            return ParsedSentiment(scores=_empty_scores(), parse_status="parse_failed")
        regex_scores[dimension] = score
    return ParsedSentiment(scores=regex_scores, parse_status="regex")


def build_sun_prompt(text: str, *, max_input_chars: int = 6000) -> str:
    clipped = str(text or "")[: int(max_input_chars)]
    return (
        "You are a financial-news sentiment scorer for volatility forecasting.\n"
        "Follow the method of decomposing narratives into three theory-driven\n"
        "dimensions: macroeconomic uncertainty, institutional action, and risk-off\n"
        "intensity.\n\n"
        "Score only the factual financial content of the article. Do not mechanically\n"
        "inherit the author's editorial tone, sensational wording, or general bearish\n"
        "style unless it is supported by concrete facts in the text.\n\n"
        "Return only JSON:\n"
        "{\n"
        '  "macroeconomic_uncertainty": number from 0 to 1,\n'
        '  "institutional_action": number from 0 to 1,\n'
        '  "risk_off_intensity": number from 0 to 1\n'
        "}\n\n"
        "News text:\n"
        f"{clipped}"
    )


def _align_vector(scores: dict[str, float], *, target_dim: int) -> np.ndarray:
    if int(target_dim) < len(SENTIMENT_DIMENSIONS):
        raise ValueError(
            f"target_dim must be at least {len(SENTIMENT_DIMENSIONS)} for Sun-style sentiment, got {target_dim}"
        )
    aligned = np.zeros(int(target_dim), dtype=np.float32)
    for idx, dimension in enumerate(SENTIMENT_DIMENSIONS):
        aligned[idx] = float(scores.get(dimension, 0.0))
    return aligned


def _cache_key(
    *,
    model_id: str,
    prompt_version: str,
    text: str,
    max_input_chars: int,
) -> str:
    payload = {
        "model_id": model_id,
        "prompt_version": prompt_version,
        "dimensions": list(SENTIMENT_DIMENSIONS),
        "text": str(text or "")[: int(max_input_chars)],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(cache_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not str(cache_path).strip():
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(record.get("cache_key", ""))
        if key:
            cache[key] = record
    return cache


def _append_cache_record(cache_path: str | Path | None, record: dict[str, Any]) -> None:
    if cache_path is None or not str(cache_path).strip():
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _default_backend(
    *,
    model_id: str,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
) -> SentimentBackend:
    return HuggingFaceLlamaSentimentBackend(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )


def fit_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    torch_dtype: str = "auto",
    max_new_tokens: int = 256,
    max_input_chars: int = 6000,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    backend: SentimentBackend | None = None,
) -> SentimentFeatureResult:
    """Build fixed-width Sun-style LLaMA sentiment vectors from news text."""

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if limit is not None and int(limit) >= 0:
        news_df = news_df.head(int(limit)).copy()

    texts = _text_series(news_df, text_column)
    backend = backend or _default_backend(
        model_id=str(model_id),
        device=str(device),
        torch_dtype=str(torch_dtype),
        max_new_tokens=int(max_new_tokens),
    )
    cache = _load_cache(cache_path)
    source = f"hf_llama3_sun2026_style:{model_id}"

    vectors: list[np.ndarray] = []
    raw_responses: list[str] = []
    parse_statuses: list[str] = []

    for text in texts.tolist():
        clean_text = str(text or "").strip()
        if not clean_text:
            parsed = ParsedSentiment(scores=_empty_scores(), parse_status="empty_text")
            raw_response = ""
        else:
            key = _cache_key(
                model_id=str(model_id),
                prompt_version=PROMPT_VERSION,
                text=clean_text,
                max_input_chars=int(max_input_chars),
            )
            cached = cache.get(key)
            if cached is not None:
                raw_response = str(cached.get("raw_response", ""))
                parsed = parse_llama_sentiment_response(raw_response)
                parsed = ParsedSentiment(scores=parsed.scores, parse_status=f"cache_{parsed.parse_status}")
            else:
                prompt = build_sun_prompt(clean_text, max_input_chars=int(max_input_chars))
                raw_response = backend.generate(prompt)
                parsed = parse_llama_sentiment_response(raw_response)
                record = {
                    "cache_key": key,
                    "model_id": str(model_id),
                    "prompt_version": PROMPT_VERSION,
                    "raw_response": raw_response,
                    "parse_status": parsed.parse_status,
                }
                cache[key] = record
                _append_cache_record(cache_path, record)
                if sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))
        vectors.append(_align_vector(parsed.scores, target_dim=target_dim))
        raw_responses.append(raw_response)
        parse_statuses.append(parsed.parse_status)

    frame = pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "sentiment_embedding": [_serialize_vector(vector) for vector in vectors],
            "sentiment_dim": [target_dim] * len(news_df),
            "sentiment_dictionary_source": [source] * len(news_df),
            "sentiment_model_id": [str(model_id)] * len(news_df),
            "sentiment_prompt_version": [PROMPT_VERSION] * len(news_df),
            "sentiment_parse_status": parse_statuses,
            "sentiment_raw_response": raw_responses,
        }
    )
    return SentimentFeatureResult(
        frame=frame,
        manifest={
            "text_column": text_column,
            "target_dim": target_dim,
            "row_count": int(len(news_df)),
            "model_id": str(model_id),
            "prompt_version": PROMPT_VERSION,
            "sentiment_dimensions": list(SENTIMENT_DIMENSIONS),
            "base_feature_dim": int(len(SENTIMENT_DIMENSIONS)),
            "representation": "sun2026_style_llama3_multidimensional_sentiment",
            "sentiment_source": source,
            "max_new_tokens": int(max_new_tokens),
            "max_input_chars": int(max_input_chars),
            "device": str(device),
            "torch_dtype": str(torch_dtype),
            "cache_path": str(cache_path) if cache_path is not None else "",
        },
    )


def build_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    model_id: str = DEFAULT_MODEL_ID,
    backend: SentimentBackend | None = None,
) -> pd.DataFrame:
    """Return only the sentiment feature frame for simple callers."""

    return fit_sentiment_features(
        news_df,
        text_column=text_column,
        target_dim=target_dim,
        model_id=model_id,
        backend=backend,
    ).frame
