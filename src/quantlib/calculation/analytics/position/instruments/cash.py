from dataclasses import dataclass
from datetime import date

from quantlib.calculation.analytics.currency_pair import Currency, CNY
from quantlib.calculation.analytics.position.instruments.features import Instrument


@dataclass
class CashPayment(Instrument):
    amount: float
    payment_date: date
    currency: Currency = CNY


