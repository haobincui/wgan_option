from dataclasses import dataclass

from quantlib.calculation.analytics.position.instruments.interest_rate.interest_rate_instrument import IRInstrument


@dataclass()
class IRSwaption(IRInstrument):
    pass
