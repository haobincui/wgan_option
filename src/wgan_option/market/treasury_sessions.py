"""CME Globex trading sessions for 10-Year Treasury futures and options."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd


CHICAGO = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
REQUIRED_CLOSURE_COLUMNS = {
    "closed_start_utc",
    "closed_end_utc",
    "event_name",
    "source_url",
    "notes",
}


def _utc_minute(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Missing session timestamp: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("min")


@dataclass(frozen=True)
class TreasuryClosure:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    event_name: str
    source_url: str
    notes: str


@dataclass(frozen=True)
class TreasuryTradingSession:
    session_id: str
    open_utc: pd.Timestamp
    close_utc: pd.Timestamp


class TreasuryGlobexSessionCalendar:
    """Regular 23-hour Globex week with frozen historical closure overrides.

    Continuous matching normally runs Sunday-Friday from 17:00 to 16:00
    America/Chicago, with a daily 16:00-17:00 halt. Closure overrides split or
    suppress those regular intervals for historical holiday schedules.
    """

    def __init__(self, closures: list[TreasuryClosure] | None = None) -> None:
        self.closures = sorted(closures or [], key=lambda item: item.start_utc)
        self._validate_closures()
        self._closure_starts = [int(item.start_utc.value) for item in self.closures]

    @classmethod
    def from_csv(cls, path: Path) -> "TreasuryGlobexSessionCalendar":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CME session calendar does not exist: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = sorted(REQUIRED_CLOSURE_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"CME session calendar is missing columns {missing}: {path}"
            )
        closures = [
            TreasuryClosure(
                start_utc=_utc_minute(row.closed_start_utc),
                end_utc=_utc_minute(row.closed_end_utc),
                event_name=str(row.event_name).strip(),
                source_url=str(row.source_url).strip(),
                notes=str(row.notes).strip(),
            )
            for row in frame.itertuples(index=False)
        ]
        return cls(closures)

    def _validate_closures(self) -> None:
        previous_end: Optional[pd.Timestamp] = None
        for closure in self.closures:
            if closure.end_utc <= closure.start_utc:
                raise ValueError(
                    "CME closure end must be after start: "
                    f"{closure.start_utc} -> {closure.end_utc}"
                )
            if not closure.event_name:
                raise ValueError("CME closure event_name cannot be empty")
            if previous_end is not None and closure.start_utc < previous_end:
                raise ValueError("CME closure intervals must not overlap")
            previous_end = closure.end_utc

    @staticmethod
    def _is_regular_open(timestamp: pd.Timestamp) -> bool:
        local = timestamp.tz_convert(CHICAGO)
        weekday = local.weekday()
        minute = int(local.hour) * 60 + int(local.minute)
        if weekday == 5:
            return False
        if weekday == 6:
            return minute >= 17 * 60
        if weekday <= 3:
            return minute < 16 * 60 or minute >= 17 * 60
        return minute < 16 * 60

    @staticmethod
    def _chicago_timestamp(day: Any, hour: int) -> pd.Timestamp:
        local_datetime = datetime.combine(
            day,
            time(hour=int(hour)),
            tzinfo=CHICAGO,
        )
        return pd.Timestamp(local_datetime)

    @staticmethod
    def _regular_session(timestamp: pd.Timestamp) -> TreasuryTradingSession:
        local = timestamp.tz_convert(CHICAGO)
        local_date = local.date()
        weekday = local.weekday()
        minute = int(local.hour) * 60 + int(local.minute)
        if weekday == 6 and minute >= 17 * 60:
            open_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date, 17
            )
            close_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date + timedelta(days=1), 16
            )
        elif weekday <= 4 and minute < 16 * 60:
            open_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date - timedelta(days=1), 17
            )
            close_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date, 16
            )
        elif weekday <= 3 and minute >= 17 * 60:
            open_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date, 17
            )
            close_local = TreasuryGlobexSessionCalendar._chicago_timestamp(
                local_date + timedelta(days=1), 16
            )
        else:
            raise ValueError(f"Timestamp is outside regular Globex hours: {timestamp}")
        open_utc = open_local.tz_convert("UTC")
        close_utc = close_local.tz_convert("UTC")
        return TreasuryTradingSession(
            session_id=f"cme_ty_{open_utc.strftime('%Y%m%dT%H%MZ')}",
            open_utc=open_utc,
            close_utc=close_utc,
        )

    def closure_at(self, value: Any) -> Optional[TreasuryClosure]:
        timestamp = _utc_minute(value)
        position = bisect_right(self._closure_starts, int(timestamp.value)) - 1
        if position < 0:
            return None
        closure = self.closures[position]
        if closure.start_utc <= timestamp < closure.end_utc:
            return closure
        return None

    def closed_reason(self, value: Any) -> str:
        timestamp = _utc_minute(value)
        closure = self.closure_at(timestamp)
        if closure is not None:
            return f"holiday:{closure.event_name}"
        local = timestamp.tz_convert(CHICAGO)
        weekday = local.weekday()
        minute = int(local.hour) * 60 + int(local.minute)
        if weekday == 5 or (weekday == 6 and minute < 17 * 60):
            return "weekend"
        if weekday == 4 and minute >= 16 * 60:
            return "weekend"
        if 16 * 60 <= minute < 17 * 60:
            return "daily_halt"
        return "outside_regular_session"

    def is_open(self, value: Any) -> bool:
        timestamp = _utc_minute(value)
        return self._is_regular_open(timestamp) and self.closure_at(timestamp) is None

    def containing_session(self, value: Any) -> Optional[TreasuryTradingSession]:
        timestamp = _utc_minute(value)
        if not self.is_open(timestamp):
            return None
        regular = self._regular_session(timestamp)
        open_utc = regular.open_utc
        close_utc = regular.close_utc
        for closure in self.closures:
            if closure.end_utc <= regular.open_utc:
                continue
            if closure.start_utc >= regular.close_utc:
                break
            if closure.end_utc <= timestamp:
                open_utc = max(open_utc, closure.end_utc)
            elif closure.start_utc > timestamp:
                close_utc = min(close_utc, closure.start_utc)
                break
        return TreasuryTradingSession(
            session_id=f"cme_ty_{open_utc.strftime('%Y%m%dT%H%MZ')}",
            open_utc=open_utc,
            close_utc=close_utc,
        )

    def next_open(self, value: Any) -> pd.Timestamp:
        timestamp = _utc_minute(value)
        if self.is_open(timestamp):
            return timestamp
        candidate = timestamp
        for _ in range(8 * 24 * 60):
            closure = self.closure_at(candidate)
            if closure is not None:
                candidate = closure.end_utc
                continue
            if self._is_regular_open(candidate):
                return candidate
            local = candidate.tz_convert(CHICAGO)
            local_date = local.date()
            weekday = local.weekday()
            minute = int(local.hour) * 60 + int(local.minute)
            if weekday == 5:
                candidate = self._chicago_timestamp(
                    local_date + timedelta(days=1), 17
                ).tz_convert("UTC")
            elif weekday == 6 and minute < 17 * 60:
                candidate = self._chicago_timestamp(
                    local_date, 17
                ).tz_convert("UTC")
            elif weekday == 4 and minute >= 16 * 60:
                candidate = self._chicago_timestamp(
                    local_date + timedelta(days=2), 17
                ).tz_convert("UTC")
            elif 16 * 60 <= minute < 17 * 60:
                candidate = self._chicago_timestamp(
                    local_date, 17
                ).tz_convert("UTC")
            else:
                candidate += pd.Timedelta(minutes=1)
        raise ValueError(f"No CME Treasury session open found after {timestamp}")

    def session_for_interval(
        self,
        start: Any,
        end: Any,
    ) -> Optional[TreasuryTradingSession]:
        start_utc = _utc_minute(start)
        end_utc = _utc_minute(end)
        if end_utc <= start_utc:
            raise ValueError("Session interval end must be after start")
        session = self.containing_session(start_utc)
        if session is None:
            return None
        if start_utc < session.open_utc or end_utc > session.close_utc:
            return None
        return session
