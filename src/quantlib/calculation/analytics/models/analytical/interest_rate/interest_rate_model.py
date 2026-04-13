from enum import Enum


class InterestRateModelType(Enum):
    ShortRateModel = 0
    MarketModel = 1
    Other = 9

class ShortRateModelType(Enum):
    VasicekModel = 0
    CirModel = 1
    BlackKarasinskiModel = 2
    HullWhiteExtendedVasicekModel = 3
    HullWhiteOneFactorModel = 4
    CirppModel = 5
    HullWhiteTwoFactorModel = 5
    Cir2ppModel = 6
    Other = 9





