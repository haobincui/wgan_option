from dataclasses import dataclass, field

from quantlib.calendar.schedule import TimeUnit, Period


class ScenarioDefinition:
    pass


@dataclass
class BlackScholesScenarioDefinitionSingleAsset(ScenarioDefinition):
    spot_bump: float = 0
    vol_bump: float = 0
    r_bump: float = 0
    q_bump: float = 0
    strike_bump: float = 0
    valuation_date_shift: Period = field(
        default_factory=lambda: Period(length=0, time_unit=TimeUnit.BUSINESS_DAY)
    )
