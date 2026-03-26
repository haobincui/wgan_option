from typing import Optional, List
from datetime import date
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass


class OptionType(Enum):
    CALL = 1
    PUT = 0


class BarrierDirection(Enum):
    UP = 1,
    DOWN = 2


class RebateType(Enum):
    PAY_WHEN_HIT = 1
    PAY_AT_EXPIRY = 2


class KiCouponType(Enum):
    CONTINUE = 1
    STOP = 2
    WRITE_OFF = 3


@dataclass
class Instrument:
    pass


@dataclass
class Security(Instrument):
    instrument_id: str


@dataclass
class BasketInstrument(Instrument):
    components: List[Security]
    weights: List[float]


@dataclass
class SpotContract(Security):
    instrument_type: str = ''


@dataclass
class FuturesContract(Security):
    instrument_type: str = ''


default_security = SpotContract(instrument_id='DEFAULT_SECURITY')


@dataclass
class SingleAssetOption(Instrument):
    underlying: Security


@dataclass
class BasketOption(Instrument):
    underlyer: BasketInstrument


@dataclass
class Position:
    """
    A instruments position consists of an position and its quantity.
    tags are optional and they are irrelevant to pricing but may be used by pricing service for aggregation etc.
    """
    quantity: float
    instrument: Instrument
    tags: Optional[dict] = None


class Decomposable(ABC):
    @abstractmethod
    def decompose(self, as_of: date) -> List[Position]:
        pass


"""
Most option payoffs are just functions. So it is tempting to design, for example, a barrier
option's final nko payoff as a function. This is not an issue for computation, but problematic
for data serialization because functions generally cannot be serialized as pure data. Yet it is
a hard requirement for a system used by business people. So here we adopt a solution that
puts payoff function in a dataclass with __call__ interface so that this payoff class can be used
as a function, yet still serializable. Essentially we are making a closure a dataclass. 
"""


@dataclass
class PayoffFunction:

    def __call__(self, x):
        pass


@dataclass
class ConstantPayoff(PayoffFunction):
    payment: float

    def __call__(self, x):
        return self.payment


@dataclass
class VanillaEuropeanPayoff(PayoffFunction):
    strike: float
    option_type: OptionType

    def __call__(self, x):
        return max(x - self.strike, 0) if self.option_type == OptionType.CALL else max(self.strike - x, 0)


@dataclass
class DigitalCashPayoff(PayoffFunction):
    strike: float
    option_type: OptionType
    payment: float

    def __call__(self, x):
        if self.option_type == OptionType.CALL and x >= self.strike:
            return self.payment
        if self.option_type == OptionType.PUT and x <= self.strike:
            return self.payment
        return 0


@dataclass
class VerticalSpreadPayoff(PayoffFunction):
    strike_low: float
    strike_high: float
    option_type: OptionType

    def __call__(self, x):
        if self.option_type == OptionType.CALL:
            if x < self.strike_low:
                return 0
            elif self.strike_low <= x <= self.strike_high:
                return x - self.strike_low
            else:
                return self.strike_high - self.strike_low
        else:
            if x < self.strike_low:
                return self.strike_high - self.strike_low
            elif self.strike_low <= x <= self.strike_high:
                return self.strike_high - x
            else:
                return 0


@dataclass
class SnowballNonePayoff(PayoffFunction):
    strike: float
    participation: float = 1

    def __call__(self, x):
        return (x - self.strike) * self.participation if self.strike > x else 0


@dataclass
class SnowballPartialPayoff(PayoffFunction):
    strike: float
    min_return: float
    participation: float = 1

    def __call__(self, x):
        y = (x - self.strike) * self.participation if self.strike > x else 0
        return self.min_return if self.min_return > y else y


@dataclass
class SnowballDoubleDigitalPayoff(PayoffFunction):
    payoff: float
    participation: float = 1

    def __call__(self, x):
        return self.payoff * self.participation


@dataclass
class SnowballForwardPayoff(PayoffFunction):
    strike: float
    participation: float = 1

    def __call__(self, x):
        return (x - self.strike) * self.participation


@dataclass()
class EnhancedPayoff:
    participation: float = 1

    def __call__(self, x, y):
        return (x - y) * self.participation


@dataclass
class SnowballForwardPartialPayoff(PayoffFunction):
    strike: float
    min_return: float
    participation: float = 1

    def __call__(self, x):
        y = self.participation * (x - self.strike)
        return self.min_return if self.min_return > y else y


@dataclass
class AsianEnhancementPrice(PayoffFunction):
    strike: float
    option_type: OptionType

    def __call__(self, x):
        return max(x, self.strike) if self.option_type == OptionType.CALL else min(self.strike, x)


@dataclass
class SnowballBoosterPayoff(PayoffFunction):
    strike: float
    participation1: float
    participation2: float
    max_payment: float
    min_payment: float

    def __call__(self, x):
        y = x - self.strike
        return min(y * self.participation1, self.max_payment) if y > 0 else max(y * self.participation2,
                                                                                self.min_payment)


@dataclass
class StranglePayoff(PayoffFunction):
    strike_low: float
    strike_high: float
    participation_low: float
    participation_high: float

    def __call__(self, x):
        return max(x - self.strike_high, 0) * self.participation_high + max(self.strike_low - x,
                                                                            0) * self.participation_low
