from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional, Union, Tuple
import logging
import warnings

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeWarning
from scipy.stats import norm

from quantlib.calendar.daycount import DayCount
from .svi_surface import SviVolSurface

logger = logging.getLogger(__name__)

HARD_SIGMA_MIN = 1e-4
HARD_SIGMA_MAX = 10.0
SOFT_SIGMA_MIN = 5e-4
SOFT_SIGMA_MAX = 5.0
SOFT_RHO_LIMIT = 0.999
MIN_A_VALUE = 1e-12
MIN_B_VALUE = 1e-12
FALLBACK_MAX_ERROR_RATIO = 1.05
FALLBACK_MAX_ABSOLUTE_ERROR_DELTA = 5e-4
QLS_POWELL_MAXITER = 400
DIRECT_FALLBACK_MAXITER = 400


@dataclass(frozen=True)
class _CalibrationStageResult:
    params: List[float]
    objective: float
    vol_objective: float
    hard_valid: bool
    boundary_hit: bool
    method: str

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


def _calibration_weights(log_moneyness: np.array, implied_vars: np.array, t: float,
                         use_weights: bool = True) -> np.array:
    if not use_weights:
        return np.ones_like(implied_vars, dtype=np.float64)

    forward = 1 / np.exp(log_moneyness)
    vegas = _black_scholes_vega(strike=1, forward=forward, vars=implied_vars, tau=t, r=0.0175)
    vegas_sum = float(np.sum(vegas))
    if not np.isfinite(vegas_sum) or vegas_sum <= 0:
        return np.ones_like(implied_vars, dtype=np.float64)
    return np.divide(vegas, vegas_sum)


def _weighted_svi_sqrt_error(implied_vars: np.array, model_vars: np.array, weights: np.array) -> float:
    diff = np.subtract(implied_vars, model_vars)
    weighted_error = np.multiply(np.square(diff), weights).sum()
    if not np.isfinite(weighted_error):
        return float("inf")
    return float(np.sqrt(weighted_error))


def _vars_to_vols(total_vars: np.array, t: float, days_in_year: int) -> np.array:
    tau = max(float(t) / float(days_in_year), 1e-12)
    return np.sqrt(np.maximum(total_vars, 0.0) / tau)


def _weighted_vol_sqrt_error(implied_vars: np.array, model_vars: np.array, weights: np.array, t: float,
                             days_in_year: int) -> float:
    implied_vols = _vars_to_vols(implied_vars, t=t, days_in_year=days_in_year)
    model_vols = _vars_to_vols(model_vars, t=t, days_in_year=days_in_year)
    diff = np.subtract(implied_vols, model_vols)
    weighted_error = np.multiply(np.square(diff), weights).sum()
    if not np.isfinite(weighted_error):
        return float("inf")
    return float(np.sqrt(weighted_error))


def _max_b_for_slice(t: float, days_in_year: int, rho: float) -> float:
    if t <= 0:
        return float("inf")
    return float(4 / (t / days_in_year * (1 + np.abs(rho))))


def _make_invalid_stage_result(method: str, objective: float = float("inf")) -> _CalibrationStageResult:
    return _CalibrationStageResult(
        params=[0.0, 0.0, 0.0, 0.0, HARD_SIGMA_MIN],
        objective=float(objective),
        vol_objective=float("inf"),
        hard_valid=False,
        boundary_hit=True,
        method=method,
    )


def _evaluate_stage_result(params: List[float], objective: float, log_moneyness: np.array, implied_vars: np.array,
                           t: float, days_in_year: int, method: str,
                           vol_objective: float) -> _CalibrationStageResult:
    a, b, rho, m, sigma = [float(x) for x in params]
    max_a = float(np.max(implied_vars))
    min_m = float(np.min(log_moneyness))
    max_m = float(np.max(log_moneyness))
    max_b = _max_b_for_slice(t=t, days_in_year=days_in_year, rho=rho)

    hard_valid = (
        np.isfinite(objective)
        and np.isfinite(a)
        and np.isfinite(b)
        and np.isfinite(rho)
        and np.isfinite(m)
        and np.isfinite(sigma)
        and 0 <= a <= max_a
        and 0 < b <= max_b
        and -1 <= rho <= 1
        and min_m <= m <= max_m
        and HARD_SIGMA_MIN <= sigma <= HARD_SIGMA_MAX
    )
    boundary_hit = (
        (not hard_valid)
        or abs(rho) >= SOFT_RHO_LIMIT
        or sigma <= SOFT_SIGMA_MIN
        or sigma >= SOFT_SIGMA_MAX
        or a <= MIN_A_VALUE
        or b <= MIN_B_VALUE
    )
    return _CalibrationStageResult(
        params=[a, b, rho, m, sigma],
        objective=float(objective),
        vol_objective=float(vol_objective),
        hard_valid=bool(hard_valid),
        boundary_hit=bool(boundary_hit),
        method=method,
    )


def _quasi_explicit_stage_result(m: float, sigma: float, log_moneyness: np.array, implied_vars: np.array, t: float,
                                 days_in_year: int, weights: np.array, method: str) -> _CalibrationStageResult:
    sigma = float(sigma)
    if not np.isfinite(m) or not np.isfinite(sigma) or sigma <= 0:
        return _make_invalid_stage_result(method=method)

    y = np.divide((log_moneyness - m), sigma)
    if not np.all(np.isfinite(y)):
        return _make_invalid_stage_result(method=method)

    a, d, c = _quasi_explicit_function(y, implied_vars, sigma)
    model_vars = svi_quasi_variances(y, a, d, c)
    objective = _weighted_svi_sqrt_error(implied_vars=implied_vars, model_vars=model_vars, weights=weights)
    vol_objective = _weighted_vol_sqrt_error(
        implied_vars=implied_vars,
        model_vars=model_vars,
        weights=weights,
        t=t,
        days_in_year=days_in_year,
    )

    if not np.isfinite(c) or c <= 0:
        b = 0.0
        rho = float("inf")
    else:
        b = float(c / sigma)
        rho = float(d / c)

    return _evaluate_stage_result(
        params=[a, b, rho, m, sigma],
        objective=objective,
        log_moneyness=log_moneyness,
        implied_vars=implied_vars,
        t=t,
        days_in_year=days_in_year,
        method=method,
        vol_objective=vol_objective,
    )


def _unique_preserve_order(values: List[float]) -> List[float]:
    deduped: List[float] = []
    for value in values:
        numeric = float(value)
        if any(np.isclose(numeric, existing, atol=1e-12, rtol=0.0) for existing in deduped):
            continue
        deduped.append(numeric)
    return deduped


def _build_qls_seed_pairs(log_moneyness: np.array) -> List[Tuple[float, float]]:
    min_m = float(np.min(log_moneyness))
    max_m = float(np.max(log_moneyness))
    mid_m = float((min_m + max_m) / 2.0)
    strike_span = max(float(max_m - min_m), 1e-3)

    m_seeds = _unique_preserve_order([min_m, mid_m, max_m])
    sigma_seeds = _unique_preserve_order(
        [
            HARD_SIGMA_MIN,
            SOFT_SIGMA_MIN,
            0.01,
            0.05,
            0.15,
            strike_span / 4.0,
            strike_span / 2.0,
            strike_span,
        ]
    )
    sigma_seeds = [float(np.clip(seed, HARD_SIGMA_MIN, HARD_SIGMA_MAX)) for seed in sigma_seeds]

    return [(m_seed, sigma_seed) for m_seed in m_seeds for sigma_seed in sigma_seeds]


def _safe_sigmoid(x: np.array) -> np.array:
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softplus(x: np.array) -> np.array:
    clipped = np.clip(x, -50.0, 50.0)
    return np.log1p(np.exp(clipped))


def _inverse_softplus(value: float) -> float:
    clipped = max(float(value), 1e-12)
    if clipped > 50:
        return clipped
    return float(np.log(np.expm1(clipped)))


def _inverse_unit_interval(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1 - 1e-6))
    return float(np.log(clipped / (1 - clipped)))


def _inverse_tanh(value: float) -> float:
    clipped = float(np.clip(value, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def _increment_stat(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + 1


def _calibration_function(svi_params: List[float], log_moneyness: np.array, implied_vars: np.array, t: float) -> float:
    # a: float, b: float, rho: float, m: float, sigma: float)
    # add vega weights
    weights = _calibration_weights(log_moneyness=log_moneyness, implied_vars=implied_vars, t=t, use_weights=True)
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
        return self.get_calibrated_vol_surface().implied_vol(
            forward=forward,
            strike=strike,
            expiration_date=expiration_date,
        )


def _quasi_explicit_function(y: np.array, implied_vars: np.array, sigma: float) -> Tuple[
    float, float, float]:
    r"""
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


def _calibration_function_qls(log_moneyness: np.array, implied_vars: np.array, t: float, days_in_year: int,
                              maxiter: int = 20, eps: float = 1e-12,
                              use_weights: bool = True) -> _CalibrationStageResult:
    weights = _calibration_weights(log_moneyness=log_moneyness, implied_vars=implied_vars, t=t,
                                   use_weights=use_weights)
    min_m = float(np.min(log_moneyness))
    max_m = float(np.max(log_moneyness))
    best_result: Optional[_CalibrationStageResult] = None
    seed_pairs = _build_qls_seed_pairs(log_moneyness)

    def target_func(params: np.array) -> float:
        m = float(params[0])
        sigma = float(params[1])
        return _quasi_explicit_stage_result(
            m=m,
            sigma=sigma,
            log_moneyness=log_moneyness,
            implied_vars=implied_vars,
            t=t,
            days_in_year=days_in_year,
            weights=weights,
            method="qls-stage1",
        ).objective

    for seed_m, seed_sigma in seed_pairs:
        if np.isclose(min_m, max_m):
            candidate = _quasi_explicit_stage_result(
                m=min_m,
                sigma=seed_sigma,
                log_moneyness=log_moneyness,
                implied_vars=implied_vars,
                t=t,
                days_in_year=days_in_year,
                weights=weights,
                method="qls-stage1",
            )
        else:
            initial_guess = np.array([seed_m, seed_sigma], dtype=np.float64)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                res = optimize.minimize(
                    target_func,
                    initial_guess,
                    method="Powell",
                    bounds=[(min_m, max_m), (HARD_SIGMA_MIN, HARD_SIGMA_MAX)],
                    options={"maxiter": max(QLS_POWELL_MAXITER, maxiter * 20), "xtol": eps, "ftol": eps},
                )
            candidate = _quasi_explicit_stage_result(
                m=float(res.x[0]),
                sigma=float(res.x[1]),
                log_moneyness=log_moneyness,
                implied_vars=implied_vars,
                t=t,
                days_in_year=days_in_year,
                weights=weights,
                method="qls-stage1",
            )

        if best_result is None or candidate.objective < best_result.objective:
            best_result = candidate
        if candidate.hard_valid and (not candidate.boundary_hit) and candidate.objective <= eps:
            break

    if best_result is None:
        return _make_invalid_stage_result(method="qls-stage1")
    return best_result


def _direct_svi_fallback_fit(log_moneyness: np.array, implied_vars: np.array, t: float, days_in_year: int,
                             initial_params: List[float], use_weights: bool = True) -> _CalibrationStageResult:
    weights = _calibration_weights(log_moneyness=log_moneyness, implied_vars=implied_vars, t=t,
                                   use_weights=use_weights)
    target_vols = _vars_to_vols(implied_vars, t=t, days_in_year=days_in_year)
    min_m = float(np.min(log_moneyness))
    max_m = float(np.max(log_moneyness))
    max_a = max(float(np.max(implied_vars)), MIN_A_VALUE)
    m_span = max(max_m - min_m, 0.0)

    def unpack(raw_params: np.array) -> List[float]:
        raw_a, raw_b, raw_rho, raw_m, raw_sigma = [float(x) for x in raw_params]
        a = float(_softplus(np.array(raw_a)))
        sigma = float(SOFT_SIGMA_MIN + _softplus(np.array(raw_sigma)))
        rho = float(SOFT_RHO_LIMIT * np.tanh(raw_rho))
        b = float(_softplus(np.array(raw_b)))
        if m_span <= 0:
            m = min_m
        else:
            m = float(min_m + m_span * _safe_sigmoid(np.array(raw_m)))
        return [a, b, rho, m, sigma]

    def residuals(raw_params: np.array) -> np.array:
        a, b, rho, m, sigma = unpack(raw_params)
        model_vars = _svi_function(a, b, rho, m, sigma, np.exp(log_moneyness))
        model_vols = _vars_to_vols(model_vars, t=t, days_in_year=days_in_year)

        residual = np.sqrt(weights) * np.subtract(model_vols, target_vols)
        penalty_terms = [0.0, 0.0, 0.0]
        if a > max_a:
            penalty_terms[0] = float((a - max_a) * 100.0)
        max_b = _max_b_for_slice(t=t, days_in_year=days_in_year, rho=rho)
        if b > max_b:
            penalty_terms[1] = float((b - max_b) * 100.0)
        if sigma >= SOFT_SIGMA_MAX:
            penalty_terms[2] = float((sigma - SOFT_SIGMA_MAX) * 100.0)
        return np.concatenate([residual, np.asarray(penalty_terms, dtype=np.float64)])

    init_a, init_b, init_rho, init_m, init_sigma = [float(x) for x in initial_params]
    if not np.isfinite(init_a) or init_a <= 0:
        init_a = max(np.mean(implied_vars), MIN_A_VALUE)
    if not np.isfinite(init_b) or init_b <= 0:
        init_b = 0.01
    if not np.isfinite(init_rho):
        init_rho = 0.0
    if not np.isfinite(init_m):
        init_m = float((min_m + max_m) / 2.0)
    if not np.isfinite(init_sigma) or init_sigma <= SOFT_SIGMA_MIN:
        init_sigma = max(SOFT_SIGMA_MIN + 1e-3, 0.01)

    if m_span <= 0:
        raw_m = 0.0
    else:
        raw_m = _inverse_unit_interval((init_m - min_m) / m_span)

    stage1_guess = np.array(
        [
            _inverse_softplus(init_a),
            _inverse_softplus(init_b),
            _inverse_tanh(init_rho / SOFT_RHO_LIMIT),
            raw_m,
            _inverse_softplus(max(init_sigma - SOFT_SIGMA_MIN, 1e-12)),
        ],
        dtype=np.float64,
    )
    neutral_a = max(float(np.mean(implied_vars)), MIN_A_VALUE)
    seed_guesses = [
        stage1_guess,
        np.array(
            [
                _inverse_softplus(neutral_a),
                _inverse_softplus(0.02),
                0.0,
                0.0,
                _inverse_softplus(max(0.05 - SOFT_SIGMA_MIN, 1e-12)),
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                _inverse_softplus(neutral_a),
                _inverse_softplus(0.01),
                0.0,
                0.0,
                _inverse_softplus(max(0.01 - SOFT_SIGMA_MIN, 1e-12)),
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                _inverse_softplus(neutral_a),
                _inverse_softplus(0.02),
                _inverse_tanh(-0.5),
                0.0,
                _inverse_softplus(max(0.05 - SOFT_SIGMA_MIN, 1e-12)),
            ],
            dtype=np.float64,
        ),
    ]

    best_result: Optional[_CalibrationStageResult] = None
    for initial_guess in seed_guesses:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            res = optimize.least_squares(
                residuals,
                initial_guess,
                max_nfev=DIRECT_FALLBACK_MAXITER,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )

        params = unpack(np.asarray(res.x, dtype=np.float64))
        model_vars = _svi_function(params[0], params[1], params[2], params[3], params[4], np.exp(log_moneyness))
        objective = _weighted_svi_sqrt_error(implied_vars=implied_vars, model_vars=model_vars, weights=weights)
        vol_objective = _weighted_vol_sqrt_error(
            implied_vars=implied_vars,
            model_vars=model_vars,
            weights=weights,
            t=t,
            days_in_year=days_in_year,
        )
        candidate = _evaluate_stage_result(
            params=params,
            objective=objective,
            log_moneyness=log_moneyness,
            implied_vars=implied_vars,
            t=t,
            days_in_year=days_in_year,
            method="qls-direct-fallback",
            vol_objective=vol_objective,
        )
        if best_result is None or (
                candidate.vol_objective < best_result.vol_objective - 1e-12
                or (
                    abs(candidate.vol_objective - best_result.vol_objective) <= 1e-12
                    and candidate.objective < best_result.objective
                )
        ):
            best_result = candidate

    if best_result is None:
        return _make_invalid_stage_result(method="qls-direct-fallback")
    return best_result


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
    stats: Optional[Dict[str, int]] = None
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
                stage1 = _calibration_function_qls(
                    np.array(self.log_moneyness[idx]),
                    np.array(self.vars[idx]),
                    t,
                    self.days_in_year,
                    use_weights=True,
                )
                selected = stage1

                if stage1.boundary_hit:
                    _increment_stat(self.stats, "qls_boundary_retry_slices")
                    _increment_stat(self.stats, "qls_fallback_attempt_slices")
                    fallback = _direct_svi_fallback_fit(
                        log_moneyness=np.array(self.log_moneyness[idx]),
                        implied_vars=np.array(self.vars[idx]),
                        t=t,
                        days_in_year=self.days_in_year,
                        initial_params=stage1.params,
                        use_weights=True,
                    )
                    if fallback.hard_valid and (not fallback.boundary_hit) and (
                            fallback.objective <= stage1.objective * FALLBACK_MAX_ERROR_RATIO
                            or fallback.objective - stage1.objective <= FALLBACK_MAX_ABSOLUTE_ERROR_DELTA
                    ) and fallback.vol_objective <= stage1.vol_objective + 1e-4:
                        selected = fallback
                        _increment_stat(self.stats, "qls_fallback_success_slices")
                    else:
                        _increment_stat(self.stats, "qls_boundary_reject_slices")
                        logger.debug(
                            "Boundary-hit slice rejected for business_day=%s stage1_obj=%s fallback_obj=%s",
                            t,
                            stage1.objective,
                            fallback.objective,
                        )
                        continue
                else:
                    _increment_stat(self.stats, "qls_stage1_kept_slices")

                res_a, res_b, res_rho, res_m, res_sigma = selected.params
                a.append(float(res_a))
                b.append(float(res_b))
                rho.append(float(res_rho))
                m.append(float(res_m))
                sigma.append(float(res_sigma))
                ttm.append(t)
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
        return self.get_calibrated_vol_surface().implied_vol(
            forward=forward,
            strike=strike,
            expiration_date=expiration_date,
        )
