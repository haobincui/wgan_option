"""Build Sun-style ChatGPT sentiment features for RQ2 text baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


DEFAULT_MODEL_ID = "gpt-5.5"
PROMPT_VERSION = "sun2026_zero_shot_chatgpt_v1"
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
    """Feature frame plus metadata describing the ChatGPT sentiment source."""

    frame: pd.DataFrame
    manifest: dict[str, Any]


def _sentiment_schema() -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            dimension: {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "A score from 0 to 1.",
            }
            for dimension in SENTIMENT_DIMENSIONS
        },
        "required": list(SENTIMENT_DIMENSIONS),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "sun2026_sentiment_scores",
        "strict": True,
        "schema": schema,
    }


def _chat_completion_response_format() -> dict[str, Any]:
    response_schema = _sentiment_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_schema["name"],
            "schema": response_schema["schema"],
            "strict": response_schema["strict"],
        },
    }


class OpenAIChatGPTSentimentBackend:
    """Lazy OpenAI Responses API backend for ChatGPT-style sentiment scoring."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_output_tokens: int = 256,
        reasoning_effort: str = "low",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model_id = str(model_id)
        self.max_output_tokens = int(max_output_tokens)
        self.reasoning_effort = str(reasoning_effort)
        self.api_key_env = str(api_key_env)

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"OpenAI API key is not configured. Set {self.api_key_env}, for example: "
                f"export {self.api_key_env}='your_api_key_here'"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep at runtime.
            raise ImportError("OpenAI sentiment generation requires the openai Python package.") from exc

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        if hasattr(self.client, "responses"):
            return self._generate_with_responses(prompt)
        return self._generate_with_chat_completions(prompt)

    def _generate_with_responses(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_id,
            reasoning={"effort": self.reasoning_effort},
            input=[
                {
                    "role": "developer",
                    "content": (
                        "You are a financial-news sentiment scorer for volatility forecasting. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text={"format": _sentiment_schema()},
            max_output_tokens=self.max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return _extract_response_text(response)

    def _generate_with_chat_completions(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial-news sentiment scorer for volatility forecasting. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=self.reasoning_effort,
            response_format=_chat_completion_response_format(),
            max_completion_tokens=self.max_output_tokens,
        )
        choices = getattr(response, "choices", None)
        if not choices and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return str(content or "")


def _extract_response_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return ""
    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


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


def parse_chatgpt_sentiment_response(raw_response: str) -> ParsedSentiment:
    """Parse Sun-style three-dimensional scores from a ChatGPT response."""

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
        "Score the following financial news article for a volatility-forecasting experiment.\n"
        "Follow the Sun (2026)-style decomposition of narratives into three theory-driven\n"
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
    max_output_tokens: int,
    reasoning_effort: str,
    api_key_env: str,
) -> SentimentBackend:
    return OpenAIChatGPTSentimentBackend(
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        api_key_env=api_key_env,
    )


def _generate_with_retries(
    backend: SentimentBackend,
    prompt: str,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    attempts = max(1, int(max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return backend.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - keep long API jobs resumable across OpenAI timeout classes.
            last_exc = exc
            if attempt >= attempts:
                break
            wait_seconds = max(0.0, float(retry_backoff_seconds)) * float(attempt)
            print(
                f"OpenAI sentiment request failed on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait_seconds:.1f}s.",
                file=sys.stderr,
                flush=True,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


def fit_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    model_id: str | None = None,
    max_output_tokens: int = 256,
    max_input_chars: int = 6000,
    reasoning_effort: str = "low",
    api_key_env: str = "OPENAI_API_KEY",
    cache_path: str | Path | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    max_retries: int = 5,
    retry_backoff_seconds: float = 5.0,
    progress_every: int = 100,
    continue_on_error: bool = False,
    backend: SentimentBackend | None = None,
) -> SentimentFeatureResult:
    """Build fixed-width Sun-style ChatGPT sentiment vectors from news text."""

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if limit is not None and int(limit) >= 0:
        news_df = news_df.head(int(limit)).copy()

    resolved_model_id = str(model_id or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL_ID)
    texts = _text_series(news_df, text_column)
    backend = backend or _default_backend(
        model_id=resolved_model_id,
        max_output_tokens=int(max_output_tokens),
        reasoning_effort=str(reasoning_effort),
        api_key_env=str(api_key_env),
    )
    cache = _load_cache(cache_path)
    source = f"openai_chatgpt_sun2026_style:{resolved_model_id}"

    vectors: list[np.ndarray] = []
    raw_responses: list[str] = []
    parse_statuses: list[str] = []
    total_rows = int(len(news_df))
    cache_hits = 0
    api_calls = 0
    api_errors = 0

    for row_idx, text in enumerate(texts.tolist(), start=1):
        clean_text = str(text or "").strip()
        if not clean_text:
            parsed = ParsedSentiment(scores=_empty_scores(), parse_status="empty_text")
            raw_response = ""
        else:
            key = _cache_key(
                model_id=resolved_model_id,
                prompt_version=PROMPT_VERSION,
                text=clean_text,
                max_input_chars=int(max_input_chars),
            )
            cached = cache.get(key)
            if cached is not None:
                cache_hits += 1
                raw_response = str(cached.get("raw_response", ""))
                parsed = parse_chatgpt_sentiment_response(raw_response)
                parsed = ParsedSentiment(scores=parsed.scores, parse_status=f"cache_{parsed.parse_status}")
            else:
                prompt = build_sun_prompt(clean_text, max_input_chars=int(max_input_chars))
                try:
                    raw_response = _generate_with_retries(
                        backend,
                        prompt,
                        max_retries=int(max_retries),
                        retry_backoff_seconds=float(retry_backoff_seconds),
                    )
                    api_calls += 1
                    parsed = parse_chatgpt_sentiment_response(raw_response)
                    record = {
                        "cache_key": key,
                        "model_id": resolved_model_id,
                        "prompt_version": PROMPT_VERSION,
                        "raw_response": raw_response,
                        "parse_status": parsed.parse_status,
                    }
                    cache[key] = record
                    _append_cache_record(cache_path, record)
                    if sleep_seconds > 0:
                        time.sleep(float(sleep_seconds))
                except Exception as exc:  # noqa: BLE001 - preserve checkpoint and optionally keep going.
                    api_errors += 1
                    if not continue_on_error:
                        raise RuntimeError(
                            "OpenAI sentiment request failed after retries. "
                            "Successful rows have already been cached; rerun the same command "
                            "with the same --output-dir/--cache-path and --model to resume."
                        ) from exc
                    raw_response = f"{type(exc).__name__}: {exc}"
                    parsed = ParsedSentiment(scores=_empty_scores(), parse_status="api_error")
        vectors.append(_align_vector(parsed.scores, target_dim=target_dim))
        raw_responses.append(raw_response)
        parse_statuses.append(parsed.parse_status)
        progress_interval = int(progress_every)
        if progress_interval > 0 and (row_idx % progress_interval == 0 or row_idx == total_rows):
            print(
                f"sentiment progress {row_idx}/{total_rows} "
                f"(cache_hits={cache_hits}, api_calls={api_calls}, api_errors={api_errors})",
                file=sys.stderr,
                flush=True,
            )

    frame = pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "sentiment_embedding": [_serialize_vector(vector) for vector in vectors],
            "sentiment_dim": [target_dim] * len(news_df),
            "sentiment_dictionary_source": [source] * len(news_df),
            "sentiment_model_id": [resolved_model_id] * len(news_df),
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
            "model_id": resolved_model_id,
            "prompt_version": PROMPT_VERSION,
            "sentiment_dimensions": list(SENTIMENT_DIMENSIONS),
            "base_feature_dim": int(len(SENTIMENT_DIMENSIONS)),
            "representation": "sun2026_style_openai_chatgpt_multidimensional_sentiment",
            "sentiment_source": source,
            "max_output_tokens": int(max_output_tokens),
            "max_input_chars": int(max_input_chars),
            "reasoning_effort": str(reasoning_effort),
            "api_key_env": str(api_key_env),
            "cache_path": str(cache_path) if cache_path is not None else "",
            "max_retries": int(max_retries),
            "retry_backoff_seconds": float(retry_backoff_seconds),
            "continue_on_error": bool(continue_on_error),
            "cache_hits": int(cache_hits),
            "api_calls": int(api_calls),
            "api_errors": int(api_errors),
        },
    )


def build_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    model_id: str | None = None,
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
