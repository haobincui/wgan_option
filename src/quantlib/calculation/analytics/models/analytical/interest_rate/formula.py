"""
formulas for interest rate derivative pricing
[Interest rate models, theory and practice]
"""
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, ncx2

from quantlib.calculation.analytics.models.math_tools.least_square import squared_error
from quantlib.calculation.analytics.models.math_tools.optimize_tool import optimize_tool
from quantlib.calculation.analytics.models.utils import double_is_zero
from quantlib.calculation.analytics.position.instruments.features import OptionType

_path = 10000


def caplet_black_price(forward_rate: float, strike_rate: float, vol: float,
                       bond_price: float, reset_tau: float,
                       option_tau: float, notional_amount: float = 1) -> float:
    """
    Cpl(t, T_1, T_2, X),
    t is the valuation time, T_1 is the option maturity, T_2 is the bond_ maturity, X is the strike rate
    :param forward_rate:
    :param strike_rate:
    :param vol:
    :param bond_price: P(t, T_2), T_2 is the bond maturity
    :param reset_tau: T_2 - T_1
    :param option_tau: T_1 - t
    :param notional_amount:
    :return:
    """
    if double_is_zero(vol * vol * option_tau):
        return max(forward_rate - strike_rate, 0) * bond_price * reset_tau * notional_amount

    d1 = (np.log(forward_rate / strike_rate) + 0.5 * vol * vol * option_tau) / (vol * np.sqrt(option_tau))
    d2 = d1 - vol * np.sqrt(option_tau)

    caplet_price = notional_amount * reset_tau * bond_price * (forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2))
    return caplet_price


def caplet_black_vol(caplet_price: float, forward_rate: float, strike_rate: float,
                     bond_price: float, reset_tau: float,
                     option_tau: float, notional_amount: float = 1) -> float:

    try:
        return brentq(
        lambda x: caplet_price - caplet_black_price(
            forward_rate, strike_rate, x, bond_price, reset_tau, option_tau, notional_amount), 0, 1)
    except Exception as e:
        def _target_func(param):
            vol = param
            price = caplet_black_price(forward_rate, strike_rate, vol, bond_price, reset_tau, option_tau, notional_amount)
            return squared_error(caplet_price, price)
        initial_guess = np.array([0.04])
        bound = np.array([(0.00001, 10)])
        res = optimize_tool(_target_func, initial_guess, bound)
        return float(*res.x)


# normal model
def caplet_bachelier_price():
    raise NotImplemented


def caplet_bachelier_vol():
    raise NotImplemented


# %% Vasicek
# dr(t) = k[\theta - r(t)]dt + \sigma dW(t)
def _vasicek_a(k: float, theta: float,
               b: float, sigma: float, tau: float) -> float:
    part1 = theta - sigma * sigma / (2 * k * k)
    part2 = b - tau - sigma * sigma / (4 * k) * b * b
    return np.exp(part1 * part2)


def _vasicek_b(k: float, tau: float) -> float:
    return 1 / k * (1 - np.exp(-k * tau))


def vasicek_bond_price(long_term_rate: float, mean_revert_speed: float,
                       short_rate: float, vol: float, tau: float) -> float:
    theta = long_term_rate
    k = mean_revert_speed
    sigma = vol
    b = _vasicek_b(k, tau)
    a = _vasicek_a(k, theta, b, sigma, tau)
    return a * np.exp(-b * short_rate)


def vasicek_bond_option_price(strike_price: float, option_type: OptionType,
                              mean_revert_speed: float,
                              long_term_rate: float,
                              initial_short_rate: float,
                              vol: float, tau_option: float, tau_bond: float,
                              bond_price_bond_maturity: Optional[float] = None,
                              bond_price_option_maturity: Optional[float] = None
                              ) -> float:
    """
    :param strike:  X
    :param option_type:
    :param long_term_rate: theta
    :param mean_revert_speed: k
    :param short_rate: r
    :param vol: sigma
    :param tau_option: T
    :param tau_bond: S
    :return:
    """
    omega = 1 if option_type == OptionType.CALL else -1
    k = mean_revert_speed
    phi = norm.cdf

    part1 = 1 - np.exp(-k * (tau_bond - tau_option))
    part2 = (1 - np.exp(-2 * k * tau_option)) / (2 * k)

    sigma_p = vol / k * part1 * np.sqrt(part2)

    bond_price_S = vasicek_bond_price(long_term_rate, mean_revert_speed, initial_short_rate, vol, tau_bond) \
        if bond_price_bond_maturity is None else bond_price_bond_maturity
    bond_price_T = vasicek_bond_price(long_term_rate, mean_revert_speed, initial_short_rate, vol, tau_option) \
        if bond_price_option_maturity is None else bond_price_option_maturity

    h = 1 / sigma_p * np.log(bond_price_S / (bond_price_T * strike_price)) + sigma_p / 2

    return omega * (bond_price_S * phi(omega * h) - strike_price * bond_price_T * phi(omega * (h - sigma_p)))


# %% Cir
# dr(t) = k(\theta - r(t))dt + \sigma \sqrt{r(t)}dW(t)

def _cir_h(k, sigma):
    return np.sqrt(k * k + 2 * sigma * sigma)


def _cir_a(k, theta, sigma, h, tau):
    upper = 2 * h * np.exp((k + h) * tau / 2)
    lower = 2 * h + (k + h) * (np.exp(tau * h) - 1)
    return np.power((upper / lower), 2 * k * theta / (sigma * sigma))


def _cir_b(k, theta, sigma, h, tau):
    upper = 2 * (np.exp(tau * h) - 1)
    lower = 2 * h + (k + h) * (np.exp(tau * h) - 1)
    return upper / lower


def _cir_validator(k, theta, sigma):
    if 2 * k * theta <= sigma * sigma:
        raise ValueError(f'2 k \\theta > \\sigma^2 for cir model, '
                         f'current k [{k}], \\theta [{theta}], and \\sigma [{sigma}]')


def cir_bond_price(long_term_rate: float, mean_revert_speed: float,
                   short_rate: float, vol: float, tau: float) -> float:
    k = mean_revert_speed
    theta = long_term_rate
    sigma = vol
    # _cir_validator(k, theta, sigma)

    h = _cir_h(k, sigma)
    a = _cir_a(k, theta, sigma, h, tau)
    b = _cir_b(k, theta, sigma, h, tau)

    return a * np.exp(-b * short_rate)


# def _get_cir_short_rate(bond_price: float, long_term_rate: float, mean_revert_speed: float,
#                         vol: float, tau: float) -> float:
#     return brentq(lambda x: bond_price - cir_bond_price(long_term_rate, mean_revert_speed, x, vol, tau), 0, 1)
#

def _cir_zbc(T, S, X, r_t, k, theta, sigma, h, bond_price_S, bond_price_T):
    x2 = ncx2.cdf
    rho = (2 * h) / (sigma * sigma * (np.exp(h * T) - 1))
    phi = (k + h) / (sigma * sigma)
    a = _cir_a(k, theta, sigma, h, (S - T))
    b = _cir_b(k, theta, sigma, h, (S - T))
    r_bar = np.log(a / X) / b

    v_1 = 2 * r_bar * (rho + phi + b)
    k_1 = 4 * k * theta / (sigma * sigma)
    lamda_1 = (2 * rho * rho * r_t * np.exp(h * T)) / (rho + phi + b)

    v_2 = 2 * r_bar * (rho + phi)
    k_2 = k_1
    lamda_2 = (2 * rho * rho * r_t * np.exp(h * T)) / (rho + phi)

    call_price = bond_price_S * x2(v_1, df=k_1, nc=lamda_1) - X * bond_price_T * x2(v_2, df=k_2, nc=lamda_2)
    return call_price


def cir_bond_option_price(strike_price: float, option_type: OptionType,
                          long_term_rate: float,
                          mean_revert_speed: float,
                          initial_short_rate: float,
                          vol: float, tau_option: float, tau_bond: float,
                          bond_price_bond_maturity: Optional[float] = None,
                          bond_price_option_maturity: Optional[float] = None,
                          close_form: bool = True
                          ) -> float:
    k = mean_revert_speed
    theta = long_term_rate
    sigma = vol
    if close_form:

        # _cir_validator(k, theta, sigma)

        bond_price_S = cir_bond_price(long_term_rate, mean_revert_speed, initial_short_rate, vol, tau_bond) \
            if bond_price_bond_maturity is None else bond_price_bond_maturity

        bond_price_T = cir_bond_price(long_term_rate, mean_revert_speed, initial_short_rate, vol, tau_option) \
            if bond_price_option_maturity is None else bond_price_option_maturity

        h = _cir_h(k, sigma)
        r_t = initial_short_rate
        x2 = ncx2

        call_price = _cir_zbc(tau_option, tau_bond, strike_price, r_t, k, theta, sigma, h, bond_price_S, bond_price_T)

        # call_price = bond_price_S * x2(v_1, k_1, lamda_1) - strike_price * bond_price_T * x2(v_2, k_2, lamda_2)

        if option_type == OptionType.CALL:
            return call_price
        else:
            return call_price - bond_price_S + strike_price * bond_price_T
    else:
        path = _path
        dt = 1 / 365
        n_timesteps = int(tau_option / dt)

        rates = np.zeros((path, n_timesteps))
        rates[:, 0] = initial_short_rate

        for t in range(1, n_timesteps):
            dW = np.random.normal(0, np.sqrt(dt), size=path)
            dr = mean_revert_speed * (long_term_rate - rates[:, t - 1]) * dt + vol * np.sqrt(rates[:, t - 1]) * dW
            dr[np.isnan(dr)] = 0
            dr[np.less(dr, 0)] = 0
            rates[:, t] = rates[:, t - 1] + dr

        simulated_rates = rates[:, -1].tolist()

        cir_bond_prices = [cir_bond_price(long_term_rate=long_term_rate,
                                          mean_revert_speed=mean_revert_speed,
                                          short_rate=rate,
                                          tau=tau_bond - tau_option,
                                          vol=vol) for rate in simulated_rates]
        # print(np.mean(cir_bond_prices))
        # call_price = np.maximum(np.array(cir_bond_prices) - strike_price, 0).mean()
        discount = cir_bond_price(long_term_rate, mean_revert_speed, initial_short_rate, vol, tau_bond - tau_option)

        if option_type == OptionType.CALL:
            return np.maximum(np.array(cir_bond_prices) - strike_price, 0).mean() * discount
        else:
            return np.maximum(strike_price - np.array(cir_bond_prices), 0).mean() * discount


# %% hull white extended vasicek
# dr(t) = [\vartphi(t) - a r(t)]dt + \sigma dW(t)

def _hull_white_extended_vasicek_b(a, tau):
    return 1 / a * (1 - np.exp(- a * tau))


def hull_white_extended_vasicek_option_price(strike_price: float, option_type: OptionType,
                                             mean_revert_speed: float,
                                             bond_price_bond_maturity: float, bond_price_option_maturity: float,
                                             vol: float, tau_option: float, tau_bond: float) -> float:
    # same as vasicek model
    bond_price_S = bond_price_bond_maturity
    bond_price_T = bond_price_option_maturity
    a = mean_revert_speed
    sigma = vol
    phi = norm.cdf
    b = _hull_white_extended_vasicek_b(a, (tau_bond - tau_option))

    sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * tau_option)) / (2 * a)) * b
    h = 1 / sigma_p * np.log(bond_price_S / (bond_price_T * strike_price)) + sigma_p / 2

    call_price = bond_price_S * phi(h) - strike_price * bond_price_T * phi(h - sigma_p)
    put_price = strike_price * bond_price_T * phi(- h + sigma_p) - bond_price_S * phi(-h)

    if option_type == OptionType.CALL:
        return call_price
    else:
        return put_price


# %% cir++ model
# r(t) = x(t) + phi(t)
# dx(t) = k (theta - x(t))dt + \sigma \sqrt{x(t)}dW(t)


def _cirpp_f(k, theta, sigma, h, tau, x_0):
    part_1_up = 2 * k * theta * (np.exp(tau * h) - 1)
    part_1_lo = 2 * h + (k * h) * (np.exp(tau * h) - 1)

    part_2_up = 4 * h * h * np.exp(tau * h)
    part_2_lo = 2 * h + (k + h) * (np.exp(tau * h) - 1)

    return part_1_up / part_1_lo + x_0 * part_2_up / (part_2_lo * part_2_lo)


def cirpp_bond_option_price(strike_price: float, option_type: OptionType,
                            initial_short_rate: float,
                            initial_x: float,
                            long_term_rate: float, mean_revert_speed: float,
                            bond_price_bond_maturity: float, bond_price_option_maturity: float,
                            vol: float, tau_option: float, tau_bond: float, tau_valuation_date,
                            market_bond_price_bond_maturity: float, market_bond_price_option_maturity: float,
                            market_bond_price_valuation_date: float) -> float:
    k = mean_revert_speed
    theta = long_term_rate
    sigma = vol
    bond_price_T = bond_price_option_maturity
    bond_price_S = bond_price_bond_maturity

    h = _cir_h(k, sigma)

    a_t = _cir_a(k, theta, sigma, h, tau_valuation_date)
    a_T = _cir_a(k, theta, sigma, h, tau_option)
    a_S = _cir_a(k, theta, sigma, h, tau_bond)

    b_t = _cir_b(k, theta, sigma, h, tau_valuation_date)
    b_T = _cir_b(k, theta, sigma, h, tau_option)
    b_S = _cir_b(k, theta, sigma, h, tau_bond)

    f_cir = _cirpp_f(k, theta, sigma, h, tau_valuation_date, initial_x)
    f_t = np.log(market_bond_price_valuation_date) / tau_valuation_date
    phi_cir = f_t - f_cir

    new_strike_up = market_bond_price_option_maturity * a_S * np.exp(- b_S * initial_x)
    new_strike_lo = market_bond_price_bond_maturity * a_T * np.exp(- b_T * initial_x)
    new_strike_price = strike_price * new_strike_up / new_strike_lo

    varphi = _cir_zbc(tau_option, tau_bond, new_strike_price, (initial_short_rate - phi_cir), k, theta, sigma, h,
                      bond_price_S, bond_price_T)

    call_price_part_1_up = market_bond_price_bond_maturity * a_t * np.exp(- b_t * initial_x)
    call_price_part_1_lo = market_bond_price_valuation_date * a_S * np.exp(-b_S * initial_x)

    call_price = call_price_part_1_up / call_price_part_1_lo * varphi

    if option_type == OptionType.CALL:
        return call_price
    else:
        return call_price - bond_price_S + strike_price * bond_price_T


# %% G2 model
# r(t) = x(t) + y(t)
# dx(t) = -a x(t)dt + \sigma dW_1 (t)
# dy(t) = -b y(t)dt + \eta dW_2 (t)
# dW_1(t)dW_2(t) = \rho dt

def _g2_sigma_square(T, S, a, b, sigma, eta, rho):
    part1 = sigma * sigma / (2 * a * a * a) \
            * (1 - np.exp(-a * (S - T))) ** 2 \
            * (1 - np.exp(-2 * a * (T)))

    part2 = eta * eta / (2 * b * b * b) \
            * (1 - np.exp(-b * (S - T))) ** 2 \
            * (1 - np.exp(-2 * b * (T)))

    part3 = 2 * rho * sigma * eta / (a * b * (a + b)) \
            * (1 - np.exp(-a * (S - T))) * (1 - np.exp(- b * (S - T))) \
            * (1 - np.exp(- (a + b) * (T)))

    return part1 + part2 + part3


def gaussian_2_bond_price(mean_revert_speed_x: float, sigma_x: float,
                          mean_revert_speed_y: float, sigma_y: float,
                          correlation: float,
                          tau_bond: float,
                          initial_x_t: float,
                          initial_y_t: float):
    r"""
    r(t) = x(t) + y(t),
    dx(t) = -ax(t) + \sigma dW_1 (t),
    dy(t) = -by(t) + \eta dW_2 (t),
    dW_1(t) dW_2(t) = \rho dt
    :param mean_revert_speed_x:
    :param sigma_x:
    :param mean_revert_speed_y:
    :param sigma_y:
    :param correlation:
    :param tau_bond:
    :param initial_x_t:
    :param initial_y_t:
    :return:
    """
    a = mean_revert_speed_x
    b = mean_revert_speed_y
    sigma = sigma_x
    eta = sigma_y
    rho = correlation
    x_t = initial_x_t
    y_t = initial_y_t

    def _exp_part(mean_revert_speed):
        return (np.exp(-mean_revert_speed * tau_bond) - 1) / mean_revert_speed

    def _mean():
        return - _exp_part(a) * x_t - _exp_part(b) * y_t

    def _variance(mean_revert_speed, vol):
        return (vol / mean_revert_speed) ** 2 \
               * (tau_bond + 2 / mean_revert_speed * np.exp(- mean_revert_speed * tau_bond)
                  - 1 / (2 * mean_revert_speed) * np.exp(- 2 * mean_revert_speed * tau_bond)
                  - 3 / (2 * mean_revert_speed))

    correlation_variance = 2 * rho * (sigma * eta) / (a * b) \
                           * (tau_bond + _exp_part(a) + _exp_part(b) - _exp_part(a + b))

    variance = _variance(a, sigma) + _variance(b, eta) + correlation_variance

    price = np.exp(_mean() + 1 / 2 * variance)
    return price


def gaussian_2_bond_option_price(mean_revert_speed_x: float, sigma_x: float,
                                 mean_revert_speed_y: float, sigma_y: float,
                                 correlation: float,
                                 tau_bond: float,
                                 tau_option: float,
                                 strike_price: float,
                                 option_type: OptionType,
                                 initial_x_t: Optional[float] = None,
                                 initial_y_t: Optional[float] = None,
                                 bond_price_bond_maturity: Optional[float] = None,
                                 bond_price_option_maturity: Optional[float] = None) -> float:
    T = tau_option
    S = tau_bond
    a = mean_revert_speed_x
    b = mean_revert_speed_y
    phi = norm.cdf

    sigma = sigma_x
    eta = sigma_y
    rho = correlation
    bond_S = bond_price_bond_maturity if bond_price_bond_maturity \
        else gaussian_2_bond_price(
        mean_revert_speed_x, sigma_x, mean_revert_speed_y, sigma_y, correlation, tau_bond, initial_x_t, initial_y_t
    )


    bond_T = bond_price_option_maturity if bond_price_option_maturity \
        else gaussian_2_bond_price(
        mean_revert_speed_x, sigma_x, mean_revert_speed_y, sigma_y, correlation, tau_option, initial_x_t, initial_y_t
    )

    # if close_form:
    Sigma = np.sqrt(_g2_sigma_square(T, S, a, b, sigma, eta, rho))

    # part_ln = np.log(bond_S / (strike_price * bond_T))

    # omega = 1 if option_type == OptionType.CALL else -1
    if option_type == OptionType.CALL:
        call = bond_S * phi(np.log(bond_S / (strike_price * bond_T)) / Sigma + 0.5 * Sigma) \
            - bond_T * strike_price * phi(np.log(bond_S / (strike_price * bond_T)) / Sigma - 0.5 * Sigma)
        return call
    else:
        put = - bond_S * phi(np.log((strike_price * bond_T) / bond_S) - 0.5 * Sigma) \
               + bond_T * strike_price * phi(np.log((strike_price * bond_T) / bond_S)/Sigma + 0.5 * Sigma)
        return put
    # else:
    #
    #     path = _path
    #     dt = 1 / 365
    #     n_timesteps = int(tau_option / dt)
    #
    #     xs = np.zeros((path, n_timesteps))
    #     xs[:, 0] = initial_x_t
    #
    #     ys = np.zeros((path, n_timesteps))
    #     ys[:, 0] = initial_y_t
    #
    #     for t in range(1, n_timesteps):
    #         dW_1 = np.random.normal(0, np.sqrt(dt), size=path)
    #         dx = mean_revert_speed_x * (- xs[:, t - 1]) * dt + sigma_x * dW_1
    #         dx[np.isnan(dx)] = 0
    #         xs[:, t] = xs[:, t - 1] + dx
    #
    #         dW_3 = np.random.normal(0, np.sqrt(dt), size=path)
    #         dy = mean_revert_speed_y * (- ys[:, t - 1]) * dt + sigma_y * (rho * dW_1 + np.sqrt(1 - rho * rho) * dW_3)
    #         dy[np.isnan(dy)] = 0
    #         # dr[np.less(dr, 0)] = 0
    #         ys[:, t] = ys[:, t - 1] + dy
    #
    #     simulated_xs = xs[:, -1].tolist()
    #     simulated_ys = ys[:, -1].tolist()
    #
    #     bond = [
    #         gaussian_2_bond_price(
    #             mean_revert_speed_x, sigma_x, mean_revert_speed_y, sigma_y,
    #             rho, tau_bond - tau_option, simulated_xs[i], simulated_ys[i]
    #         )
    #         for i in range(path)]
    #
    #     # bond_T = gaussian_2_bond_price(
    #     #     mean_revert_speed_x, sigma_x, mean_revert_speed_y, sigma_y, rho, tau_option, initial_x_t, initial_y_t
    #     # )
    #     if option_type == OptionType.CALL:
    #         trade_price = np.mean(np.maximum(np.subtract(bond,strike_price), 0)) * bond_T
    #     else:
    #         trade_price = np.mean(np.maximum(np.subtract(strike_price, bond), 0)) * bond_T
    #
    #     return trade_price


def nelson_siegel_zero_rate(beta_0: float, beta_1: float, beta_2: float, tau: int, bond_maturity: float):
    # nelson siegel 1987
    a = beta_0
    b = beta_1 + beta_2
    c = beta_2
    m_t = bond_maturity / tau
    r_t = a + b * (1 - np.exp(- m_t)) / m_t - c * np.exp(- m_t)
    return r_t


def nelson_siegel_svensson_forward_rate(beta_0: float, beta_1: float, beta_2: float, beta_3: float,
                                        tau_1: int, tau_2: int, bond_maturity: float):
    if beta_3 == 0:
        tau_2 = tau_1

    # svensson 1994
    m_t_1 = bond_maturity / tau_1
    m_t_2 = bond_maturity / tau_2

    factors = [1, np.exp(- m_t_1), m_t_1 * np.exp(-m_t_1), m_t_2 * np.exp(-m_t_2)]

    return float(np.sum(np.multiply([beta_0, beta_1, beta_2, beta_3], factors)))


def nelson_siegel_svensson_spot_rate(beta_0: float, beta_1: float, beta_2: float, beta_3: float,
                                     tau_1: int, tau_2: int, bond_maturity: float):
    # svensson 1994
    if beta_3 == 0:
        tau_2 = tau_1

    m_t_1 = bond_maturity / tau_1
    m_t_2 = bond_maturity / tau_2

    factors = [1, (1 - np.exp(- m_t_1)) / m_t_1,
               (1 - np.exp(-m_t_1)) / m_t_1 - np.exp(-m_t_1),
               (1 - np.exp(-m_t_2)) / m_t_2 - np.exp(-m_t_2)]

    return float(np.sum(np.multiply([beta_0, beta_1, beta_2, beta_3], factors)))

#
# def gaussian_bond_option_price(mean_revert_speed: float, sigma: float,
#                                  tau_bond: float,
#                                  tau_option: float,
#                                  strike_price: float,
#                                  option_type: OptionType,
#                                  initial_x_t: Optional[float] = None,
#                                  initial_y_t: Optional[float] = None,
#                                  bond_price_bond_maturity: Optional[float] = None,
#                                  bond_price_option_maturity: Optional[float] = None,
#                                  close_form: bool = True) -> float:
#     Sigma_square = sigma * sigma / (mean_revert_speed**3) \
#             * (1 - np.exp(-mean_revert_speed * (tau_bond - tau_option))) ** 2 \
#             * (1 - np.exp(-2 * mean_revert_speed * tau_option))
#     Sigma = np.sqrt(Sigma_square)
#
#     bond_S = bond_price_bond_maturity
#     bond_T = bond_price_option_maturity
#
#     phi = norm.cdf
#
#     call = bond_S * phi(np.log(bon))
