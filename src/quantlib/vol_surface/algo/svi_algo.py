from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Union, Tuple
import logging
import warnings

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeWarning
from scipy.stats import norm

from quantlib.calendar.daycount import DayCount
from .svi_surface import TermVolSurfaceByDays, SviVolSurface

logger = logging.getLogger(__name__)

@dataclass
class SviCalibration(ABC):
    valuation_date: date

    @abstractmethod
    def get_calibrated_vol_surface(self) -> SviVolSurface:
        pass

    @abstractmethod
    def get_calibrated_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        pass


def _black_scholes_vega(strike: float, forward: np.array, vars: np.array, tau: np.array, r: float) -> np.array:
    dfr = np.exp(np.multiply(-r, tau))
    var_sqrt = np.sqrt(vars)
    d1 = np.log(forward / strike) / var_sqrt + 0.5 * var_sqrt
    return forward * dfr * norm.pdf(d1) * np.sqrt(tau)


def _calibration_function(svi_params: List[float], log_moneyness: np.array, implied_vars: np.array, t: float) -> float:
    # a: float, b: float, rho: float, m: float, sigma: float)
    # add vega weights
    forward = 1 / np.exp(log_moneyness)
    vegas = _black_scholes_vega(strike=1, forward=forward, vars=implied_vars, tau=t, r=0.0175)
    weights = np.divide(vegas, vegas.sum())
    a = svi_params[0]
    b = svi_params[1]
    rho = svi_params[2]
    m = svi_params[3]
    sigma = svi_params[4]
    vars = _svi_function(a, b, rho, m, sigma, np.exp(log_moneyness))
    # change weights here
    sum_error = np.multiply(np.power(np.subtract(implied_vars, vars), 2), weights).sum()
    # sum_sqrt_error = np.sqrt(sum_error)
    return sum_error


def _svi_function(a: float, b: float, rho: float, m: float, sigma: float, percent_strike: List[float]) -> np.array:
    log_moneyness = np.log(np.array(percent_strike))
    help1 = np.multiply(rho, np.subtract(log_moneyness, m))
    help2 = np.sqrt(np.power(np.subtract(log_moneyness, m), 2) + sigma * sigma)
    vars = a + np.multiply(b, (help1 + help2))
    return vars


@dataclass
class SviCalibrationGatheral(SviCalibration):
    """
    Gatheral' Svi mode calibration (Gatheral, 2004)
    """
    # valuation_date: date
    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    vol_daycount: DayCount
    days_in_year: int = field(init=False)
    log_moneyness: List[List[float]] = field(init=False)
    vars: List[List[float]] = field(init=False)
    params: Dict[str, List[float]] = field(init=False)

    def __post_init__(self):
        try:
            self.days_in_year = self.vol_daycount.days_in_year
        except:
            self.days_in_year = 365

        def _get_svi_params() -> Dict[str, List[Union[float, int]]]:
            a = []
            b = []
            rho = []
            m = []
            sigma = []
            ttm = []
            for idx, t in enumerate(self.business_days):
                # initial guess

                initial_params = np.array([np.min(self.vars),
                                           0,
                                           0.5,
                                           0,
                                           0.0001])
                i = 0  # maximum calibration time  = 100
                calibrated_result_within_boundary = False
                while not calibrated_result_within_boundary:
                    # boundary for a, b, rho, m, sigma
                    bounds = np.array([(0, np.max(self.vars[idx])),
                                       (0, 4 / (t / self.days_in_year * 2)),
                                       (-1, 1),
                                       (np.log(1 / 10), np.log(10)),
                                       (0.0001, 10)])

                    res = optimize.minimize(_calibration_function, initial_params,
                                            (np.array(self.log_moneyness[idx]), np.array(self.vars[idx]), t),
                                            bounds=bounds, method='Nelder-Mead')
                    res_a = res.x[0]
                    res_b = res.x[1]
                    res_rho = res.x[2]
                    res_m = res.x[3]
                    res_sigma = res.x[4]

                    # check for boundary conditions
                    max_a = np.max(self.vars[idx])
                    max_b = 4 / (t / self.days_in_year * (1 + np.abs(res_rho)))
                    max_m = np.max(self.log_moneyness[idx])
                    min_m = np.min(self.log_moneyness[idx])

                    if 0 <= res_a <= max_a and \
                            0 < res_b <= max_b and \
                            -1 <= res_rho <= 1 and \
                            min_m <= res_m <= max_m and \
                            0.0001 <= res_sigma <= 10:
                        a.append(res_a)
                        b.append(res_b)
                        rho.append(res_rho)
                        m.append(res_m)
                        sigma.append(res_sigma)
                        ttm.append(t)
                        calibrated_result_within_boundary = True
                    else:
                        i = i + 1
                        initial_params = np.multiply(initial_params, 1.05)
                    if i > 100:
                        logger.debug(f'该时间 {t} 未能找到对应SVI Params; 跳过该时间')
                        break
            # special handle if there is no param founded
            if len(a) == 0 or len(b) == 0 or len(rho) == 0 or len(sigma) == 0 or len(m) == 0:
                logger.warning(f'所有时间都未能找到SVI Params; 使用最短TTM对应的平均variance')
                a = [np.mean(self.vars[0])]
                b = [0]
                rho = [0]
                m = [0]
                sigma = [0]
                ttm = [self.business_days[0]]

            params = {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma, 'business_days': ttm}
            return params

        self.log_moneyness = []
        self.vars = []
        for idx, t in enumerate(self.business_days):
            vol_array = np.array(self.vols[idx])
            moneyness_array = np.array(self.percent_strikes[idx])
            self.log_moneyness.append(np.log(moneyness_array).tolist())
            var = np.multiply(np.power(vol_array, 2), t / self.days_in_year)
            self.vars.append(var.tolist())
        self.params = _get_svi_params()

    def get_calibrated_vol_surface(self) -> SviVolSurface:
        return SviVolSurface(valuation_date=self.valuation_date,
                             svi_params=self.params,
                             vol_daycount=self.vol_daycount)

    def get_calibrated_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        # var = a + b * (rho * (k - m) + sqrt((k - m) ^ 2 + sigma ^ 2))
        log_moneyness = np.log(strike / forward)
        svi_params = self.params
        a = np.array(svi_params['a'])
        b = np.array(svi_params['b'])
        rho = np.array(svi_params['rho'])
        m = np.array(svi_params['m'])
        sigma = np.array(svi_params['sigma'])
        ttm = np.array(svi_params['business_days'])
        help1 = np.multiply(rho, -np.subtract(m, -log_moneyness))
        help2 = np.sqrt(np.power(-np.subtract(m, -log_moneyness), 2) + np.power(sigma, 2))
        vars = a + np.multiply(b, (help1 + help2))
        vols = np.sqrt(vars * self.days_in_year / ttm).tolist()

        vol_surface = TermVolSurfaceByDays(valuation_date=self.valuation_date,
                                           vols=vols,
                                           business_days=self.business_days,
                                           vol_daycount=self.vol_daycount)
        return vol_surface.implied_vol(expiration_date)


def _quasi_explicit_function(y: np.array, implied_vars: np.array, sigma: float) -> Tuple[
    float, float, float]:
    """
    quasi_explicit function: var = a + d * y + c * z
    var = a + \sqrt(2) / 2 * (d - c) * (y + z) + \sqrt(2) / 2 * (d + c) * (- y + z)
    a_star = a; d_star = \sqrt(2) / 2 * (d - c); c_star = \sqrt(2) / 2 * (d + c)
    boundary condition:
            1) 0 <= a <= max(vars)
            2) abs(d) - c <= 0
            3) abs(d) + c <= 4 * sigma
    """
    z = np.sqrt(np.square(y) + 1)
    A = np.column_stack([np.ones(len(implied_vars)), np.sqrt(2) / 2 * (y + z), np.sqrt(2) / 2 * (-y + z)])
    s = max(sigma, 1e-4)
    bnd = ((0, 0, 0), (max(implied_vars.max(), 1e-4), 2 * np.sqrt(2) * s, 2 * np.sqrt(2) * s))

    res = optimize.lsq_linear(A=A, b=implied_vars, bounds=bnd, tol=1e-12, verbose=False)
    a_star, d_star, c_star = res.x
    a = a_star
    d = np.sqrt(2) / 2 * (d_star - c_star)
    c = np.sqrt(2) / 2 * (d_star + c_star)
    return a, d, c


def svi_quasi_variances(y: np.array, a: np.array, d: np.array, c: np.array) -> np.array:
    return a + d * y + c * np.sqrt(np.square(y) + 1)


def svi_quasi_sqrt_error(implied_variances: np.array, y: np.array, a: np.array, d: np.array, c: np.array,
                         weights: np.array) -> float:
    return np.sqrt(np.sum(np.square(svi_quasi_variances(y, a, d, c) - implied_variances) * weights))


def _calibration_function_qls(log_moneyness: np.array, implied_vars: np.array, t: float, maxiter: int = 20,
                              eps: float = 1e-12, use_weights: bool = True) -> List[float]:
    #  m: float, sigma: float) -> List:
    if use_weights:
        forward = 1 / np.exp(log_moneyness)
        vegas = _black_scholes_vega(strike=1, forward=forward, vars=implied_vars, tau=t, r=0.0175)
        weights = np.divide(vegas, vegas.sum())
    else:
        weights = np.ones_like(implied_vars)

    def target_func(params: List[float]):
        m = params[0]
        sigma = params[1]
        y = np.divide((log_moneyness - m), sigma)
        a, d, c = _quasi_explicit_function(y, implied_vars, sigma)
        vars = svi_quasi_variances(y, a, d, c)
        return np.sqrt(np.multiply(np.power(np.subtract(implied_vars, vars), 2), weights).sum())

    # m, sigma
    initial_guess = np.array([0, 0.01])
    bounds = np.array([(np.min(log_moneyness), np.max(log_moneyness)), (0.0001, 10)])
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    # try different initial guess of m and sigma
    for i in range(0, maxiter + 1):
        initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            res = optimize.minimize(target_func, initial_guess, bounds=bounds, method='Nelder-Mead', tol=eps)
        m, sigma = res.x
        cost = res.fun
        y = np.divide((log_moneyness - m), sigma)
        a, d, c = _quasi_explicit_function(y, implied_vars, sigma)
        if abs(cost) < eps:
            logger.debug(f'minimum sqrt error [{cost}] founded with iterated [{i + 1}] times')
            break
        # new initial guess
        initial_guess = np.array([m + np.random.normal() * 0.001,
                                  sigma + np.random.normal() * 0.001])

    logger.debug(f'minimum sqrt error [{cost}] founded with iterated [{i + 1}] times')
    b = c / sigma
    rho = d / c

    return [a, b, rho, m, sigma]


@dataclass
class SviCalibrationQuasiExplicit(SviCalibration):
    """
    svi params calibration using quasi explicit parameters method (C. Martini and S. De Marco, 2012)
    """
    # valuation_date: date
    vols: List[List[float]]
    percent_strikes: List[List[float]]
    business_days: List[int]
    vol_daycount: DayCount
    days_in_year: int = field(init=False)
    log_moneyness: List[List[float]] = field(init=False)
    vars: List[List[float]] = field(init=False)
    params: Dict[str, List[float]] = field(init=False)

    def __post_init__(self):
        try:
            self.days_in_year = self.vol_daycount.days_in_year
        except:
            self.days_in_year = 365

        def _get_svi_params() -> Dict[str, List[float]]:
            a = []
            b = []
            rho = []
            m = []
            sigma = []
            ttm = []
            for idx, t in enumerate(self.business_days):
                res = _calibration_function_qls(np.array(self.log_moneyness[idx]), np.array(self.vars[idx]), t,
                                                use_weights=True)
                res_a = res[0]
                res_b = res[1]
                res_rho = res[2]
                res_m = res[3]
                res_sigma = res[4]

                # check for boundary conditions
                max_a = np.max(self.vars[idx])
                max_b = 4 / (t / self.days_in_year * (1 + np.abs(res_rho)))
                max_m = np.max(self.log_moneyness[idx])
                min_m = np.min(self.log_moneyness[idx])

                if 0 <= res_a <= max_a and \
                        0 < res_b <= max_b and \
                        -1 <= res_rho <= 1 and \
                        min_m <= res_m <= max_m and \
                        0.0001 <= res_sigma <= 10:
                    a.append(res_a)
                    b.append(res_b)
                    rho.append(res_rho)
                    m.append(res_m)
                    sigma.append(res_sigma)
                    ttm.append(t)
                else:
                    logger.debug(f'该时间 [{t}] 未能找到对应SVI Params; 跳过该时间')
                    break
            # special handle if there is no param founded
            if len(a) == 0 or len(b) == 0 or len(rho) == 0 or len(sigma) == 0 or len(m) == 0:
                logger.warning(f'所有时间都未能找到SVI Params; 使用最短TTM对应的平均variance')
                a = [np.mean(self.vars[0])]
                b = [0]
                rho = [0]
                m = [0]
                sigma = [0]
                ttm = [self.business_days[0]]

            params = {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma, 'business_days': ttm}
            return params

        self.log_moneyness = []
        self.vars = []
        for idx, t in enumerate(self.business_days):
            vol_array = np.array(self.vols[idx])
            moneyness_array = np.array(self.percent_strikes[idx])
            self.log_moneyness.append(np.log(moneyness_array).tolist())
            var = np.multiply(np.power(vol_array, 2), t / self.days_in_year)
            self.vars.append(var.tolist())
        self.params = _get_svi_params()

    def get_calibrated_vol_surface(self) -> SviVolSurface:
        return SviVolSurface(valuation_date=self.valuation_date,
                             svi_params=self.params,
                             vol_daycount=self.vol_daycount)

    def get_calibrated_vol(self, forward: float, strike: float, expiration_date: date) -> float:
        # var = a + b * (rho * (k - m) + sqrt((k - m) ^ 2 + sigma ^ 2))
        log_moneyness = np.log(strike / forward)
        svi_params = self.params
        a = np.array(svi_params['a'])
        b = np.array(svi_params['b'])
        rho = np.array(svi_params['rho'])
        m = np.array(svi_params['m'])
        sigma = np.array(svi_params['sigma'])
        ttm = np.array(svi_params['business_days'])
        help1 = np.multiply(rho, -np.subtract(m, -log_moneyness))
        help2 = np.sqrt(np.power(-np.subtract(m, -log_moneyness), 2) + np.power(sigma, 2))
        vars = a + np.multiply(b, (help1 + help2))
        vols = np.sqrt(vars * self.days_in_year / ttm).tolist()

        vol_surface = TermVolSurfaceByDays(valuation_date=self.valuation_date,
                                           vols=vols,
                                           business_days=self.business_days,
                                           vol_daycount=self.vol_daycount)

        return vol_surface.implied_vol(expiration_date)
