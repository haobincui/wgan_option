"""Shared-panel bootstrap for crossed model seeds and market observations.

The same frozen market panel is commonly evaluated by several model seeds and
conditions.  Those rows are not independent copies of the market.  This module
therefore draws one seed-weight vector and one nested fold/session market
schedule per bootstrap replicate.  The market schedule is shared by every
seed and condition.

Folds are represented as ordered bootstrap occurrences.  When a fold is drawn
twice, each occurrence receives an independent within-fold session draw.  The
result is a crossed ``seed x market-schedule`` bootstrap with sessions nested
inside fold occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


METHOD_VERSION = "crossed_seed_shared_fold_occurrence_session_v2"
DEFAULT_ITERATIONS = 10_000
DEFAULT_RNG_SEED = 20_260_904
Estimand = Literal["equal_cell", "pooled_pair"]

_REQUIRED_COLUMNS = (
    "condition",
    "seed",
    "fold",
    "pair_id",
    "session_id",
    "value",
)
_LINEAGE_COLUMNS = ("fold", "pair_id", "session_id")


class PanelValidationError(ValueError):
    """Raised when a condition/seed panel is incomplete or malformed."""


class ScheduleCompatibilityError(ValueError):
    """Raised when bootstrap weights do not belong to the supplied panel."""


class ScheduleArchiveError(ValueError):
    """Raised when an archived schedule is incomplete or internally invalid."""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_float(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _normalise_estimand(estimand: str) -> Estimand:
    value = str(estimand).strip()
    if value not in {"equal_cell", "pooled_pair"}:
        raise ValueError("estimand must be 'equal_cell' or 'pooled_pair'")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class Panel:
    """Validated canonical condition-by-seed panel.

    Public attributes are deliberately small and stable.  ``frame`` returns a
    defensive copy of the canonical rows.  The private arrays aggregate values
    to CME-session blocks and are used by :func:`run_bootstrap`.
    """

    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[str, ...]
    sessions_by_fold: tuple[tuple[str, ...], ...]
    market_fingerprint: str
    panel_fingerprint: str
    has_persistence: bool
    _frame: pd.DataFrame
    _pair_counts_by_fold: tuple[np.ndarray, ...]
    _value_sums_by_fold: tuple[np.ndarray, ...]

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy of the normalised canonical panel."""

        return self._frame.copy(deep=True)

    def observed_means(self, estimand: Estimand = "equal_cell") -> np.ndarray:
        """Return one observed arithmetic mean per condition.

        ``equal_cell`` gives every seed-fold cell equal weight after taking its
        pair-level mean.  ``pooled_pair`` pools all pair rows.  Contrast point
        estimates are log-ratio statistics and are computed by
        :meth:`Result.contrast`, not by taking a ratio of equal-cell arithmetic
        means.
        """

        mode = _normalise_estimand(estimand)
        condition_count = len(self.conditions)
        seed_count = len(self.seeds)
        if mode == "equal_cell":
            total = np.zeros(condition_count, dtype=np.float64)
            for sums, counts in zip(
                self._value_sums_by_fold,
                self._pair_counts_by_fold,
                strict=True,
            ):
                total += (sums.sum(axis=2) / float(counts.sum())).mean(axis=1)
            means = total / float(len(self.folds))
        else:
            numerator = np.zeros(condition_count, dtype=np.float64)
            pair_count = 0
            for sums, counts in zip(
                self._value_sums_by_fold,
                self._pair_counts_by_fold,
                strict=True,
            ):
                numerator += sums.sum(axis=(1, 2))
                pair_count += int(counts.sum())
            means = numerator / float(seed_count * pair_count)
        return _readonly_float(means)


def _canonical_fingerprints(frame: pd.DataFrame, *, has_persistence: bool) -> tuple[str, str]:
    first_condition = str(frame["condition"].iloc[0])
    first_seed = int(frame["seed"].iloc[0])
    market = frame[
        frame["condition"].eq(first_condition) & frame["seed"].eq(first_seed)
    ][list(_LINEAGE_COLUMNS)].sort_values(list(_LINEAGE_COLUMNS), kind="stable")
    market_payload = [
        [str(row.fold), str(row.pair_id), str(row.session_id)]
        for row in market.itertuples(index=False)
    ]

    panel_payload: list[list[Any]] = []
    columns = list(_REQUIRED_COLUMNS) + (["persistence_mae"] if has_persistence else [])
    for values in frame[columns].itertuples(index=False, name=None):
        row = [
            str(values[0]),
            int(values[1]),
            str(values[2]),
            str(values[3]),
            str(values[4]),
            float(values[5]).hex(),
        ]
        if has_persistence:
            row.append(float(values[6]).hex())
        panel_payload.append(row)
    return _json_sha256(market_payload), _json_sha256(panel_payload)


def prepare_panel(frame: pd.DataFrame) -> Panel:
    """Normalise and strictly validate a shared evaluation panel.

    Every condition-by-seed combination must contain exactly the same
    ``(fold, pair_id, session_id)`` lineage.  Values must be strictly positive
    because downstream contrasts use logarithms of means.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PanelValidationError(f"panel is missing canonical columns: {missing}")
    if frame.empty:
        raise PanelValidationError("panel must not be empty")

    has_persistence = "persistence_mae" in frame.columns
    columns = list(_REQUIRED_COLUMNS) + (["persistence_mae"] if has_persistence else [])
    normalised = frame.loc[:, columns].copy()

    string_columns = ("condition", "fold", "pair_id", "session_id")
    if normalised[list(string_columns)].isna().any().any():
        raise PanelValidationError("condition and market lineage must not be null")
    for column in string_columns:
        normalised[column] = normalised[column].astype(str).str.strip()
        if normalised[column].eq("").any():
            raise PanelValidationError(f"{column} must not contain empty values")

    numeric_seed = pd.to_numeric(normalised["seed"], errors="coerce")
    seed_values = numeric_seed.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(seed_values).all()
        or not np.equal(seed_values, np.floor(seed_values)).all()
        or np.any(seed_values < np.iinfo(np.int64).min)
        or np.any(seed_values > np.iinfo(np.int64).max)
    ):
        raise PanelValidationError("seed must contain finite int64-compatible integers")
    normalised["seed"] = numeric_seed.astype(np.int64)

    numeric_columns = ["value"] + (["persistence_mae"] if has_persistence else [])
    for column in numeric_columns:
        values = pd.to_numeric(normalised[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise PanelValidationError(f"{column} must be finite and strictly positive")
        normalised[column] = values

    unique_key = ["condition", "seed", "fold", "pair_id"]
    duplicated = normalised.duplicated(unique_key, keep=False)
    if duplicated.any():
        sample = normalised.loc[duplicated, unique_key].head(3).to_dict("records")
        raise PanelValidationError(f"duplicate condition/seed/fold/pair rows: {sample}")

    conditions = tuple(sorted(normalised["condition"].unique().tolist()))
    seeds = tuple(sorted(int(value) for value in normalised["seed"].unique()))
    folds = tuple(sorted(normalised["fold"].unique().tolist()))
    if not conditions or not seeds or not folds:
        raise PanelValidationError("condition, seed, and fold universes must be non-empty")

    normalised = normalised.sort_values(unique_key, kind="stable").reset_index(drop=True)
    reference = normalised[
        normalised["condition"].eq(conditions[0]) & normalised["seed"].eq(seeds[0])
    ][list(_LINEAGE_COLUMNS)].sort_values(list(_LINEAGE_COLUMNS), kind="stable")
    expected_lineage = tuple(reference.itertuples(index=False, name=None))
    if not expected_lineage:
        raise PanelValidationError("reference condition/seed panel is empty")

    grouped = normalised.groupby(["condition", "seed"], sort=False, observed=True)
    for condition in conditions:
        for seed in seeds:
            try:
                candidate = grouped.get_group((condition, seed))
            except KeyError as error:
                raise PanelValidationError(
                    f"missing condition/seed panel: condition={condition!r}, seed={seed}"
                ) from error
            lineage = tuple(
                candidate[list(_LINEAGE_COLUMNS)]
                .sort_values(list(_LINEAGE_COLUMNS), kind="stable")
                .itertuples(index=False, name=None)
            )
            if lineage != expected_lineage:
                expected_set = set(expected_lineage)
                observed_set = set(lineage)
                raise PanelValidationError(
                    "condition/seed market panel drift: "
                    f"condition={condition!r}, seed={seed}, "
                    f"missing={sorted(expected_set - observed_set)[:3]}, "
                    f"extra={sorted(observed_set - expected_set)[:3]}"
                )

    market = reference.reset_index(drop=True)
    if market["pair_id"].duplicated().any():
        raise PanelValidationError("pair_id must identify exactly one market row")
    session_fold_counts = market.groupby("session_id", sort=False)["fold"].nunique()
    if session_fold_counts.gt(1).any():
        raise PanelValidationError("session_id must be nested in exactly one fold")

    if has_persistence:
        persistence_counts = normalised.groupby(
            list(_LINEAGE_COLUMNS), sort=False, observed=True
        )["persistence_mae"].nunique(dropna=False)
        if persistence_counts.ne(1).any():
            raise PanelValidationError(
                "persistence_mae must agree across all conditions and seeds"
            )

    sessions_by_fold = tuple(
        tuple(sorted(market.loc[market["fold"].eq(fold), "session_id"].unique()))
        for fold in folds
    )
    if any(not sessions for sessions in sessions_by_fold):
        raise PanelValidationError("every fold must contain at least one session")

    condition_index = {value: index for index, value in enumerate(conditions)}
    seed_index = {value: index for index, value in enumerate(seeds)}
    aggregate = (
        normalised.groupby(
            ["condition", "seed", "fold", "session_id"],
            sort=False,
            observed=True,
        )["value"]
        .sum()
        .to_dict()
    )
    pair_counts_by_fold: list[np.ndarray] = []
    value_sums_by_fold: list[np.ndarray] = []
    for fold, sessions in zip(folds, sessions_by_fold, strict=True):
        fold_market = market[market["fold"].eq(fold)]
        counts = np.asarray(
            [int(fold_market["session_id"].eq(session).sum()) for session in sessions],
            dtype=np.int64,
        )
        sums = np.empty(
            (len(conditions), len(seeds), len(sessions)), dtype=np.float64
        )
        for condition in conditions:
            condition_position = condition_index[condition]
            for seed in seeds:
                seed_position = seed_index[seed]
                sums[condition_position, seed_position, :] = [
                    float(aggregate[(condition, seed, fold, session)])
                    for session in sessions
                ]
        counts.setflags(write=False)
        sums.setflags(write=False)
        pair_counts_by_fold.append(counts)
        value_sums_by_fold.append(sums)

    market_fingerprint, panel_fingerprint = _canonical_fingerprints(
        normalised, has_persistence=has_persistence
    )
    stored = normalised.copy(deep=True)
    return Panel(
        conditions=conditions,
        seeds=seeds,
        folds=folds,
        sessions_by_fold=sessions_by_fold,
        market_fingerprint=market_fingerprint,
        panel_fingerprint=panel_fingerprint,
        has_persistence=has_persistence,
        _frame=stored,
        _pair_counts_by_fold=tuple(pair_counts_by_fold),
        _value_sums_by_fold=tuple(value_sums_by_fold),
    )


@dataclass(frozen=True)
class Schedule:
    """Complete integer bootstrap weights plus replay metadata."""

    draw_id: np.ndarray
    seed_weights: np.ndarray
    fold_indices: np.ndarray
    session_weights: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("draw_id", "seed_weights", "fold_indices", "session_weights"):
            source = np.asarray(getattr(self, name))
            if not np.issubdtype(source.dtype, np.integer):
                raise ScheduleArchiveError(f"{name} must use an integer dtype")
            value = np.array(source, copy=True)
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)
        try:
            metadata = json.loads(_json_bytes(dict(self.metadata)).decode("utf-8"))
        except (TypeError, ValueError) as error:
            raise ScheduleArchiveError("schedule metadata must be JSON serialisable") from error
        object.__setattr__(self, "metadata", metadata)
        _validate_schedule(self, arrays=arrays)

    @property
    def iterations(self) -> int:
        return int(self.metadata["iterations"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["seeds"])

    @property
    def folds(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["folds"])

    @property
    def sessions_by_fold(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(str(session) for session in sessions)
            for sessions in self.metadata["sessions_by_fold"]
        )


def _validate_schedule(
    schedule: Schedule, *, arrays: Mapping[str, np.ndarray] | None = None
) -> None:
    metadata = schedule.metadata
    required_metadata = {
        "method_version",
        "rngseed",
        "market_fingerprint",
        "seeds",
        "folds",
        "sessions_by_fold",
        "iterations",
        "max_sessions",
        "weights_sha256",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ScheduleArchiveError(f"schedule metadata is missing keys: {missing}")
    if metadata["method_version"] != METHOD_VERSION:
        raise ScheduleArchiveError("unsupported bootstrap method_version")

    seeds = tuple(int(value) for value in metadata["seeds"])
    folds = tuple(str(value) for value in metadata["folds"])
    sessions_by_fold = tuple(
        tuple(str(session) for session in sessions)
        for sessions in metadata["sessions_by_fold"]
    )
    iterations = int(metadata["iterations"])
    max_sessions = int(metadata["max_sessions"])
    if iterations < 2 or not seeds or not folds:
        raise ScheduleArchiveError("schedule needs >=2 draws and non-empty seed/fold universes")
    if len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ScheduleArchiveError("schedule seed/fold universes must be unique")
    if len(sessions_by_fold) != len(folds) or any(
        not sessions for sessions in sessions_by_fold
    ):
        raise ScheduleArchiveError("sessions_by_fold does not match the fold universe")
    if any(
        len(set(sessions)) != len(sessions)
        or any(not session.strip() for session in sessions)
        for sessions in sessions_by_fold
    ):
        raise ScheduleArchiveError("archived session universes must be unique and non-empty")
    if max_sessions != max(map(len, sessions_by_fold)):
        raise ScheduleArchiveError("max_sessions metadata drift")

    draw_id = schedule.draw_id
    seed_weights = schedule.seed_weights
    fold_indices = schedule.fold_indices
    session_weights = schedule.session_weights
    expected_shapes = {
        "draw_id": (iterations,),
        "seed_weights": (iterations, len(seeds)),
        "fold_indices": (iterations, len(folds)),
        "session_weights": (iterations, len(folds), max_sessions),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(schedule, name).shape) != shape:
            raise ScheduleArchiveError(
                f"{name} shape drift: expected={shape}, observed={getattr(schedule, name).shape}"
            )
    if not np.array_equal(draw_id, np.arange(iterations, dtype=draw_id.dtype)):
        raise ScheduleArchiveError("draw_id must be the ordered range [0, iterations)")
    if np.any(seed_weights < 0) or not np.all(seed_weights.sum(axis=1) == len(seeds)):
        raise ScheduleArchiveError("each seed-weight row must be non-negative and sum to S")
    if np.any(fold_indices < 0) or np.any(fold_indices >= len(folds)):
        raise ScheduleArchiveError("fold_indices contains an out-of-universe fold")
    if np.any(session_weights < 0):
        raise ScheduleArchiveError("session weights must be non-negative")

    for occurrence in range(len(folds)):
        for fold_index, sessions in enumerate(sessions_by_fold):
            mask = fold_indices[:, occurrence] == fold_index
            if not np.any(mask):
                continue
            session_count = len(sessions)
            active = session_weights[mask, occurrence, :session_count]
            if not np.all(active.sum(axis=1) == session_count):
                raise ScheduleArchiveError(
                    "each selected fold occurrence must draw its session count"
                )
            if session_count < max_sessions and np.any(
                session_weights[mask, occurrence, session_count:] != 0
            ):
                raise ScheduleArchiveError("inactive session-weight padding must be zero")

    digest_arrays = (
        dict(arrays)
        if arrays is not None
        else {
            "draw_id": draw_id,
            "seed_weights": seed_weights,
            "fold_indices": fold_indices,
            "session_weights": session_weights,
        }
    )
    if str(metadata["weights_sha256"]) != _array_sha256(digest_arrays):
        raise ScheduleArchiveError("schedule weight fingerprint drift")


def make_schedule(
    panel: Panel,
    iterations: int = DEFAULT_ITERATIONS,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> Schedule:
    """Create one replayable crossed seed/market bootstrap schedule.

    Separate ``SeedSequence`` children isolate seed sampling from market
    sampling.  Consequently, changing only the number of seeds leaves
    ``fold_indices`` and ``session_weights`` byte-for-byte unchanged for a
    fixed market panel, iteration count, and root seed.
    """

    if not isinstance(panel, Panel):
        raise TypeError("panel must be prepared by prepare_panel")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or int(iterations) < 2
    ):
        raise ValueError("iterations must be an integer of at least two")
    if (
        isinstance(rng_seed, (bool, np.bool_))
        or not isinstance(rng_seed, (int, np.integer))
        or int(rng_seed) < 0
    ):
        raise ValueError("rng_seed must be a non-negative integer")
    iteration_count = int(iterations)
    root_seed = int(rng_seed)
    seed_count = len(panel.seeds)
    fold_count = len(panel.folds)
    session_counts = tuple(len(sessions) for sessions in panel.sessions_by_fold)
    max_sessions = max(session_counts)

    seed_sequence, market_sequence = np.random.SeedSequence(root_seed).spawn(2)
    seed_rng = np.random.Generator(np.random.PCG64(seed_sequence))
    market_rng = np.random.Generator(np.random.PCG64(market_sequence))
    seed_probabilities = np.full(seed_count, 1.0 / seed_count, dtype=np.float64)
    seed_weights = seed_rng.multinomial(
        seed_count, seed_probabilities, size=iteration_count
    ).astype(np.int32, copy=False)
    fold_indices = market_rng.integers(
        0,
        fold_count,
        size=(iteration_count, fold_count),
        dtype=np.int32,
    )
    session_weights = np.zeros(
        (iteration_count, fold_count, max_sessions), dtype=np.int32
    )
    for occurrence in range(fold_count):
        for fold_index, session_count in enumerate(session_counts):
            mask = fold_indices[:, occurrence] == fold_index
            selected_count = int(np.count_nonzero(mask))
            if selected_count == 0:
                continue
            probabilities = np.full(
                session_count, 1.0 / session_count, dtype=np.float64
            )
            session_weights[mask, occurrence, :session_count] = market_rng.multinomial(
                session_count, probabilities, size=selected_count
            ).astype(np.int32, copy=False)

    draw_id = np.arange(iteration_count, dtype=np.int64)
    arrays = {
        "draw_id": draw_id,
        "seed_weights": seed_weights,
        "fold_indices": fold_indices,
        "session_weights": session_weights,
    }
    metadata: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "rngseed": root_seed,
        "rng_algorithm": "numpy.random.PCG64",
        "rng_streams": {
            "seed": {"spawn_key": list(seed_sequence.spawn_key)},
            "market": {"spawn_key": list(market_sequence.spawn_key)},
        },
        "market_fingerprint": panel.market_fingerprint,
        "seeds": list(panel.seeds),
        "folds": list(panel.folds),
        "sessions_by_fold": [list(sessions) for sessions in panel.sessions_by_fold],
        "pair_counts_by_session": [
            counts.astype(int).tolist() for counts in panel._pair_counts_by_fold
        ],
        "iterations": iteration_count,
        "max_sessions": max_sessions,
        "weights_sha256": _array_sha256(arrays),
    }
    return Schedule(metadata=metadata, **arrays)


def save_schedule(schedule: Schedule, path: Path) -> None:
    """Save all integer weights and JSON metadata in one compressed NPZ."""

    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule")
    _validate_schedule(schedule)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = np.asarray(
        json.dumps(
            schedule.metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            draw_id=schedule.draw_id,
            seed_weights=schedule.seed_weights,
            fold_indices=schedule.fold_indices,
            session_weights=schedule.session_weights,
            metadata=metadata,
        )


def load_schedule(path: Path) -> Schedule:
    """Load and validate a schedule written by :func:`save_schedule`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "draw_id",
        "seed_weights",
        "fold_indices",
        "session_weights",
        "metadata",
    }
    try:
        with np.load(source, allow_pickle=False) as archive:
            observed = set(archive.files)
            if observed != required:
                raise ScheduleArchiveError(
                    f"schedule archive members drift: missing={sorted(required-observed)}, "
                    f"extra={sorted(observed-required)}"
                )
            metadata_raw = archive["metadata"]
            if metadata_raw.size != 1:
                raise ScheduleArchiveError("metadata must be one JSON scalar")
            metadata = json.loads(str(metadata_raw.reshape(()).item()))
            arrays = {
                "draw_id": np.array(archive["draw_id"], copy=True),
                "seed_weights": np.array(archive["seed_weights"], copy=True),
                "fold_indices": np.array(archive["fold_indices"], copy=True),
                "session_weights": np.array(archive["session_weights"], copy=True),
            }
    except ScheduleArchiveError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ScheduleArchiveError(f"could not load schedule archive: {source}") from error
    if not isinstance(metadata, dict):
        raise ScheduleArchiveError("schedule metadata JSON must contain an object")
    return Schedule(metadata=metadata, **arrays)


@dataclass(frozen=True)
class Result:
    """Bootstrap condition means and lazily addressable paired contrasts."""

    conditions: tuple[str, ...]
    observed_means: np.ndarray
    mean_draws: np.ndarray
    estimand: Estimand
    _observed_components: np.ndarray
    _draw_components: np.ndarray
    _seed_components: np.ndarray
    _fold_components: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "observed_means",
            "mean_draws",
            "_observed_components",
            "_draw_components",
            "_seed_components",
            "_fold_components",
        ):
            object.__setattr__(self, name, _readonly_float(getattr(self, name)))

    def contrast(self, focal: str, reference: str) -> dict[str, Any]:
        """Return paired log-ratio inference; negative values favour focal."""

        try:
            focal_index = self.conditions.index(str(focal))
            reference_index = self.conditions.index(str(reference))
        except ValueError as error:
            raise ValueError(
                f"unknown contrast condition: focal={focal!r}, reference={reference!r}"
            ) from error
        draws = np.asarray(
            self._draw_components[:, focal_index]
            - self._draw_components[:, reference_index],
            dtype=np.float64,
        )
        point = float(
            self._observed_components[focal_index]
            - self._observed_components[reference_index]
        )
        numerical_spread = float(np.ptp(draws))
        numerical_scale = max(1.0, abs(point), float(np.max(np.abs(draws))))
        if numerical_spread <= 64.0 * np.finfo(np.float64).eps * numerical_scale:
            bootstrap_se = 0.0
            lower = upper = point
        else:
            bootstrap_se = float(draws.std(ddof=1))
            lower, upper = np.quantile(draws, (0.025, 0.975))
        p_one = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (
            float(len(draws)) + 1.0
        )
        p_reverse = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (
            float(len(draws)) + 1.0
        )
        p_two = min(1.0, 2.0 * min(p_one, p_reverse))
        statistic = None if bootstrap_se == 0.0 else float(point / bootstrap_se)
        seed_ratios = (
            self._seed_components[focal_index]
            - self._seed_components[reference_index]
        )
        fold_ratios = (
            self._fold_components[focal_index]
            - self._fold_components[reference_index]
        )
        draws.setflags(write=False)
        return {
            "point": point,
            "bootstrap_se": bootstrap_se,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "p_one": float(p_one),
            "p_two": float(p_two),
            "statistic": statistic,
            "draws": draws,
            "seed_consistent": int(np.count_nonzero(seed_ratios <= 0.0)),
            "fold_consistent": int(np.count_nonzero(fold_ratios <= 0.0)),
        }


def _assert_schedule_compatible(panel: Panel, schedule: Schedule) -> None:
    _validate_schedule(schedule)
    metadata = schedule.metadata
    mismatches: list[str] = []
    if str(metadata["market_fingerprint"]) != panel.market_fingerprint:
        mismatches.append("market_fingerprint")
    if tuple(int(value) for value in metadata["seeds"]) != panel.seeds:
        mismatches.append("seeds")
    if tuple(str(value) for value in metadata["folds"]) != panel.folds:
        mismatches.append("folds")
    archived_sessions = tuple(
        tuple(str(session) for session in sessions)
        for sessions in metadata["sessions_by_fold"]
    )
    if archived_sessions != panel.sessions_by_fold:
        mismatches.append("sessions_by_fold")
    if mismatches:
        raise ScheduleCompatibilityError(
            "schedule is incompatible with panel: " + ", ".join(mismatches)
        )


def _observed_components(
    panel: Panel, estimand: Estimand
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition_count = len(panel.conditions)
    seed_count = len(panel.seeds)
    fold_count = len(panel.folds)
    if estimand == "equal_cell":
        log_cells = np.empty(
            (condition_count, seed_count, fold_count), dtype=np.float64
        )
        for fold_index, (sums, counts) in enumerate(
            zip(
                panel._value_sums_by_fold,
                panel._pair_counts_by_fold,
                strict=True,
            )
        ):
            log_cells[:, :, fold_index] = np.log(
                sums.sum(axis=2) / float(counts.sum())
            )
        return (
            log_cells.mean(axis=(1, 2)),
            log_cells.mean(axis=2),
            log_cells.mean(axis=1),
        )

    observed = panel.observed_means("pooled_pair")
    seed_numerator = np.zeros((condition_count, seed_count), dtype=np.float64)
    fold_components = np.empty((condition_count, fold_count), dtype=np.float64)
    total_pairs = 0
    for fold_index, (sums, counts) in enumerate(
        zip(panel._value_sums_by_fold, panel._pair_counts_by_fold, strict=True)
    ):
        pairs = int(counts.sum())
        total_pairs += pairs
        seed_numerator += sums.sum(axis=2)
        fold_components[:, fold_index] = np.log(
            sums.sum(axis=(1, 2)) / float(seed_count * pairs)
        )
    seed_components = np.log(seed_numerator / float(total_pairs))
    return np.log(observed), seed_components, fold_components


def run_bootstrap(
    panel: Panel,
    schedule: Schedule,
    estimand: Estimand = "equal_cell",
) -> Result:
    """Apply an archived shared schedule and return condition/contrast results."""

    if not isinstance(panel, Panel):
        raise TypeError("panel must be prepared by prepare_panel")
    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be produced by make_schedule/load_schedule")
    mode = _normalise_estimand(estimand)
    _assert_schedule_compatible(panel, schedule)

    iterations = schedule.iterations
    condition_count = len(panel.conditions)
    seed_count = len(panel.seeds)
    fold_count = len(panel.folds)
    mean_draws = np.zeros((iterations, condition_count), dtype=np.float64)
    if mode == "equal_cell":
        draw_components = np.zeros_like(mean_draws)
    else:
        pooled_numerator = np.zeros_like(mean_draws)
        pooled_denominator = np.zeros(iterations, dtype=np.float64)

    for occurrence in range(fold_count):
        for fold_index, (sums, counts) in enumerate(
            zip(
                panel._value_sums_by_fold,
                panel._pair_counts_by_fold,
                strict=True,
            )
        ):
            mask = schedule.fold_indices[:, occurrence] == fold_index
            if not np.any(mask):
                continue
            session_count = len(counts)
            weights = schedule.session_weights[mask, occurrence, :session_count].astype(
                np.float64, copy=False
            )
            numerator = np.einsum("bc,asc->bas", weights, sums, optimize=True)
            denominator = weights @ counts.astype(np.float64, copy=False)
            cell_means = numerator / denominator[:, None, None]
            seed_weights = schedule.seed_weights[mask].astype(np.float64, copy=False)
            if mode == "equal_cell":
                mean_draws[mask] += np.einsum(
                    "bs,bas->ba", seed_weights, cell_means, optimize=True
                ) / float(seed_count * fold_count)
                draw_components[mask] += np.einsum(
                    "bs,bas->ba", seed_weights, np.log(cell_means), optimize=True
                ) / float(seed_count * fold_count)
            else:
                pooled_numerator[mask] += np.einsum(
                    "bs,bas->ba", seed_weights, numerator, optimize=True
                )
                pooled_denominator[mask] += denominator * seed_weights.sum(axis=1)

    if mode == "pooled_pair":
        if np.any(pooled_denominator <= 0.0):
            raise RuntimeError("pooled bootstrap produced a non-positive denominator")
        mean_draws = pooled_numerator / pooled_denominator[:, None]
        draw_components = np.log(mean_draws)
    if not np.isfinite(mean_draws).all() or np.any(mean_draws <= 0.0):
        raise RuntimeError("bootstrap produced invalid condition means")
    if not np.isfinite(draw_components).all():
        raise RuntimeError("bootstrap produced invalid log components")

    observed_components, seed_components, fold_components = _observed_components(
        panel, mode
    )
    return Result(
        conditions=panel.conditions,
        observed_means=panel.observed_means(mode),
        mean_draws=mean_draws,
        estimand=mode,
        _observed_components=observed_components,
        _draw_components=draw_components,
        _seed_components=seed_components,
        _fold_components=fold_components,
    )


__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_RNG_SEED",
    "METHOD_VERSION",
    "Panel",
    "PanelValidationError",
    "Result",
    "Schedule",
    "ScheduleArchiveError",
    "ScheduleCompatibilityError",
    "load_schedule",
    "make_schedule",
    "prepare_panel",
    "run_bootstrap",
    "save_schedule",
]
