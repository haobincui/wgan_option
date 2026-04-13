from enum import Enum


class RiskMeasure(Enum):
    PV = 0
    DELTA = 1 # d pv / d s = PV(S_t + h) - PV(S_t - h) / 2h => model-free,
    GAMMA = 2
    VEGA = 3
    THETA = 4
    RHO = 5
    RHO_Q = 6
    CEGA = 7
    IMMEDIATE_EXERCISE = 8  # sort of settle/intrinsic value. not the same as vol = 0 which still needs discounting.
    FX_DELTA = 8  # d pv / d fx
    FX_GAMMA = 9  # d pv^2 / d^2 fx
    FX_VEGA = 10  # d pv / d fx_vol
    DENSITY = 11 # d pv^2 / d^2 k


    # PnL = -10000
    # reward = PL_t - (.) PL_t^2, PL_{t-1} = 0
    # variance, (PL_{t-1} - PL_(t))^2 ?????????????
    # wilmott delta


