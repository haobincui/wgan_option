import logging
from enum import Enum
from typing import List

import numpy as np

from quantlib.calculation.analytics.models.math_tools.optimize_tool import optimize_tool


class MarketModelVolStructure(Enum):
    PiecewiseConstant = 0
    HumpedFunction = 1


class MarketModelCalibrationer:
    """
    calibration for market model
    """

    @staticmethod
    def libor_market_model(caplet_vols: List[float],
                           taus: List[float],
                           reset_tau: float,
                           vol_structure: MarketModelVolStructure = MarketModelVolStructure.PiecewiseConstant):
        """return vols"""

        if vol_structure == MarketModelVolStructure.PiecewiseConstant:
            return MarketModelCalibrationer._piecewise_constant_calibration(caplet_vols, taus, reset_tau)
        elif vol_structure == MarketModelVolStructure.HumpedFunction:
            return MarketModelCalibrationer._function_calibration(caplet_vols, taus, reset_tau)
        else:
            raise ValueError(f'Unsupported Vol Structure [{vol_structure.name}]')

    @staticmethod
    def _piecewise_constant_calibration(caplet_vols: List[float],
                                        taus: List[float],
                                        reset_tau: float):
        # calibrated_vars = []
        calibrated_vols = []
        caplet_vars = np.power(caplet_vols, 2) * np.array(taus)
        temp_taus = [0, *taus]
        diff_taus = np.diff(temp_taus)

        def _target_func(params):
            # params: list of vols
            diff = caplet_vars[len(params) - 1] - np.sum(np.power(params, 2) * diff_taus[idx])
            return diff ** 2
        logging.debug('Start Piecewise-Constant Vol Calibration')
        for idx, var in enumerate(caplet_vars):
            initial_guess = np.array(np.sqrt(var / taus[:idx + 1]))
            bound = (0.001, 10)
            bounds = np.array([bound for _ in range(idx + 1)])
            res = optimize_tool(_target_func, initial_guess, bounds)
            # calibrated_vars.append(res.x)
            calibrated_vols.append(res.x)

        return calibrated_vols

    @staticmethod
    def _function_calibration(caplet_vols: List[float],
                              taus: List[float],
                              reset_tau: float, ):
        pass
