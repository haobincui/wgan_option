"""Date/time conversion helpers shared by market-data and calendar modules."""

from datetime import date, datetime
import re
from typing import Union

_DATE_PATTERN = re.compile(r"^\s*(?P<Y>\d{4})[-/]?(?P<M>\d{2})[-/]?(?P<D>\d{2})")
_DATETIME_PATTERN = re.compile(
    r"^\s*"
    r"(?P<Y>\d{4})[-/]?(?P<M>\d{2})[-/]?(?P<D>\d{2})"
    r"[T\s]"
    r"(?P<h>\d{2})[:\-]?(?P<m>\d{2})[:\-]?(?P<s>\d{2})"
    r"(?:[.,:](?P<frac>\d+))?"
    r"(?:Z)?"
    r"\s*$"
)


def _fraction_to_microseconds(frac: str) -> int:
    """Convert arbitrary-length fractional-second digits to microseconds."""
    if not frac:
        return 0
    return int((frac + "000000")[:6])


class DatetimeConverter:
    """Convert between date/datetime objects and common text formats."""

    @staticmethod
    def to_datetime(input_datetime: Union[date, datetime]) -> datetime:
        """Normalize `date`/`datetime` into `datetime` at midnight for dates."""
        if isinstance(input_datetime, datetime):
            return input_datetime
        if isinstance(input_datetime, date):
            return datetime(input_datetime.year, input_datetime.month, input_datetime.day)
        raise TypeError(f"Unsupported input type: {type(input_datetime)}")

    @staticmethod
    def from_string_to_date(date_str: str) -> date:
        """Parse string prefix like `YYYYMMDD`/`YYYY-MM-DD`/`YYYY/MM/DD` into `date`."""
        if not isinstance(date_str, str):
            raise TypeError(f"date_str must be str, got {type(date_str)}")
        matched = _DATE_PATTERN.match(date_str)
        if not matched:
            raise ValueError(f"Unsupported date string: {date_str}")
        parts = {k: int(v) for k, v in matched.groupdict().items()}
        return date(parts["Y"], parts["M"], parts["D"])

    @staticmethod
    def from_string_to_datetime(datetime_str: str, ms: bool = True) -> datetime:
        """
        Parse date-time string into `datetime`.

        Supported examples:
        - `20200130 083633`
        - `2020-01-30 08:36:33`
        - `2020-01-30T08:36:33.449341031Z`
        - `2020/01/30T08:36:33:449341`
        """
        if not isinstance(datetime_str, str):
            raise TypeError(f"datetime_str must be str, got {type(datetime_str)}")

        matched = _DATETIME_PATTERN.match(datetime_str)
        if not matched:
            raise ValueError(f"Unsupported datetime string: {datetime_str}")

        groups = matched.groupdict()
        y = int(groups["Y"])
        m = int(groups["M"])
        d = int(groups["D"])
        h = int(groups["h"])
        minute = int(groups["m"])
        sec = int(groups["s"])
        micro = _fraction_to_microseconds(groups.get("frac") or "") if ms else 0
        return datetime(y, m, d, h, minute, sec, microsecond=micro)

    @staticmethod
    def from_datetime_to_searcher_input(d: Union[date, datetime]) -> str:
        """Format as UTC-like string expected by search APIs: `YYYY-MM-DDTHH:MM:SS.000Z`."""
        if isinstance(d, datetime):
            dt = d.replace(microsecond=0)
        elif isinstance(d, date):
            dt = datetime(d.year, d.month, d.day)
        else:
            raise TypeError(f"Unsupported input type: {type(d)}")
        return dt.isoformat() + ".000Z"
