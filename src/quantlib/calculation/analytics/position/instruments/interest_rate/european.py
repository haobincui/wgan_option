from dataclasses import dataclass
from datetime import date

from quantlib.calculation.analytics.position.instruments.features import SingleAssetOption, OptionType


@dataclass
class Forward(SingleAssetOption):
    strike: float
    expiration_date: date
    delivery_date: date


@dataclass
class VanillaEuropean(SingleAssetOption):
    strike: float
    expiration_date: date
    option_type: OptionType
    delivery_date: date


@dataclass
class DigitalCash(SingleAssetOption):
    strike: float
    expiration_date: date
    option_type: OptionType
    payout: float
    delivery_date: date


@dataclass()
class VanillaImpliedVolatility(SingleAssetOption):
    price: float
    strike: float
    expiration_date: date
    option_type: OptionType
    delivery_date: date


