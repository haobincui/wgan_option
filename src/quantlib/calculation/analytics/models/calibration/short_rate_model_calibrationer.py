from typing import List, Tuple

import numpy as np

from quantlib.calculation.analytics.models.analytical.interest_rate.formula import (
    vasicek_bond_option_price,
    cir_bond_option_price,
    cirpp_bond_option_price,
    gaussian_2_bond_option_price,
)
from quantlib.calculation.analytics.models.math_tools.least_square import squared_error
from quantlib.calculation.analytics.models.math_tools.optimize_tool import optimize_tool
from quantlib.calculation.analytics.position.instruments.features import OptionType

_power = 0


def _weights(x, power):
    return np.power(x, power) / np.sum(np.power(x, power))


class ShortRateModelCalibrationer:

    @staticmethod
    def vasicek(strike_prices: List[float],
                taus_option: List[float],
                taus_bond: List[float],
                market_quote_bond_option_prices: List[float],
                bond_Ts: List[float] = None,
                bond_Ss: List[float] = None,
                option_type: OptionType = OptionType.PUT) -> Tuple:
        """
        calibration process for vasicek model,return (a, sigma)
        :return: a: mean revert speed, sigma: model volatility
        """
        weights = _weights(taus_option, _power)

        def _target_func(params):
            a = params[0]
            theta = params[1]
            sigma = params[2]
            r_t = params[3]

            vasicek_prices = [vasicek_bond_option_price(strike_price=strike_prices[i],
                                                        option_type=option_type,
                                                        mean_revert_speed=a,
                                                        initial_short_rate=r_t,
                                                        long_term_rate=theta,
                                                        vol=sigma,
                                                        tau_option=taus_option[i],
                                                        tau_bond=taus_bond[i],
                                                        bond_price_option_maturity=bond_Ts[i] if bond_Ts else None,
                                                        bond_price_bond_maturity=bond_Ss[i] if bond_Ss else None) for i
                              in range(len(taus_option))]
            return squared_error(market_quote_bond_option_prices, vasicek_prices, weights)

        initial_guess = np.array([0.9, 0.02, 0.048, 0.07])
        bounds = np.array([(0.001, 100), (0.001, 0.1), (0.01, 0.9), (0.01, 0.1)])

        res = optimize_tool(_target_func, initial_guess, bounds)
        a, theta, sigma, r_t = res.x
        return a, theta, sigma, r_t

    @staticmethod
    def cir(strike_prices: List[float],
            taus_option: List[float],
            taus_bond: List[float],
            market_quote_bond_option_prices: List[float],
            bond_Ts: List[float] = None,
            bond_Ss: List[float] = None,
            option_type: OptionType = OptionType.PUT) -> Tuple:
        """
        cir calibration, return (k, theta, sigma, initial short rate)
        :param strike_price:
        :param taus_option:
        :param taus_bond:
        :param market_quote_bond_option_prices:
        :param option_type:
        :param max_iter:
        :return: k: mean revert speed, theta: long-term mean, sigma, initial short rate
        """
        # weights = np.exp(market_quote_bond_option_prices) / np.exp(market_quote_bond_option_prices).sum()
        weights = _weights(taus_option, _power)

        def _target_func(params):
            k = params[0]
            theta = params[1]
            sigma = params[2]
            initial_short_rate = params[3]
            cir_prices = [cir_bond_option_price(strike_price=strike_prices[i],
                                                option_type=option_type,
                                                long_term_rate=theta, mean_revert_speed=k,
                                                initial_short_rate=initial_short_rate,
                                                vol=sigma,
                                                tau_option=taus_option[i],
                                                tau_bond=taus_bond[i],
                                                bond_price_bond_maturity=bond_Ss[i] if bond_Ss else None,
                                                bond_price_option_maturity=bond_Ts[i] if bond_Ts else None) for i in
                          range(len(taus_option))]

            return squared_error(market_quote_bond_option_prices, cir_prices, weights)

        initial_guess = np.array([1, 0.04, 0.06, 0.4])
        bounds = np.array([(0, 10), (0.01, 10), (0.01, 0.9), (0, 1)])

        res = optimize_tool(_target_func, initial_guess, bounds)
        k, theta, sigma, r_t = res.x

        return k, theta, sigma, r_t

    @staticmethod
    def cirpp(strike_price: float,
              bond_prices_bond_maturity: List[float],
              bond_prices_option_maturity: List[float],
              market_bond_prices_bond_maturity: List[float],
              market_bond_prices_option_maturity: List[float],
              market_bond_price_valuation_date: float,
              taus_option: List[float],
              taus_bond: List[float],
              tau_valuation_date: float,
              market_quote_bond_option_prices: List[float],
              option_type: OptionType = OptionType.PUT) -> Tuple:
        """
        cirpp calibration, return (k, theta, sigma, x_t)
        :param tau_valuation_date:
        :param strike_price:
        :param bond_prices_bond_maturity:
        :param bond_prices_option_maturity:
        :param market_bond_prices_bond_maturity:
        :param market_bond_prices_option_maturity:
        :param market_bond_price_valuation_date:
        :param taus_option:
        :param taus_bond:
        :param market_quote_bond_option_prices:
        :param option_type:
        :param max_iter:
        :return:
        """
        # weights = np.exp(market_quote_bond_option_prices) / np.exp(market_quote_bond_option_prices).sum()
        weights = _weights(taus_option, _power)

        def _target_func(params):
            k = params[0]
            theta = params[1]
            sigma = params[2]
            x_t = params[3]
            r_t = params[4]
            cirpp_prices = [cirpp_bond_option_price(strike_price=strike_price,
                                                    option_type=option_type,
                                                    long_term_rate=theta, mean_revert_speed=k,
                                                    initial_x=x_t,
                                                    initial_short_rate=r_t,
                                                    bond_price_bond_maturity=bond_prices_bond_maturity[i],
                                                    bond_price_option_maturity=bond_prices_option_maturity[i],
                                                    vol=sigma,
                                                    tau_option=taus_option[i],
                                                    tau_bond=taus_bond[i],
                                                    tau_valuation_date=tau_valuation_date,
                                                    market_bond_price_bond_maturity=market_bond_prices_bond_maturity[i],
                                                    market_bond_price_option_maturity=
                                                    market_bond_prices_option_maturity[i],
                                                    market_bond_price_valuation_date=market_bond_price_valuation_date,
                                                    ) for i in range(len(taus_option))]

            return squared_error(market_quote_bond_option_prices, cirpp_prices, weights)

        initial_guess = np.array([1, 0.04, 0.06, 0.4, 0.4])
        bounds = np.array([(0, 10), (0.01, 10), (0.01, 0.9), (0, 1), (0, 1)])

        res = optimize_tool(_target_func, initial_guess, bounds)
        k, theta, sigma, x_t, r_t = res.x
        return k, theta, sigma, x_t, r_t,

    @staticmethod
    def guassian_2(taus_bond: List[float],
                   taus_option: List[float],
                   strike_prices: List[float],
                   bond_Ss: List[float],
                   bond_Ts: List[float],
                   market_quote_bond_option_prices: List[float],
                   option_type: OptionType = OptionType.PUT) -> Tuple:
        """
        r(t) = x(t) + y(t),
        dx(t) = - ax(t) + \sigma dW_1(t),
        dy(t) = -by(t) + \eta dW_2(t),
        dW_1(t)dW_2(t) = \rho dt,
        return (a, \sigma, b, \eta, \rho),
        a: mean revert speed for x, \sigma: vol for x,
        b: mean revert speed for y, \eta: vol for y,
        \rho: correlation for x and y
        :return: (a, sigma, b, theta, rho)
        """
        weights = _weights(taus_option, _power)

        def _target_func(params):
            mean_revert_speed_x = params[0]
            vol_x = params[1]
            mean_revert_speed_y = params[2]
            vol_y = params[3]
            correlation = params[4]
            # initial_x_t = params[5]
            # initial_y_t = params[6]

            g2price = [gaussian_2_bond_option_price(mean_revert_speed_x=mean_revert_speed_x,
                                                    sigma_x=vol_x,
                                                    mean_revert_speed_y=mean_revert_speed_y,
                                                    sigma_y=vol_y,
                                                    correlation=correlation,
                                                    tau_bond=taus_bond[i],
                                                    tau_option=taus_option[i],
                                                    strike_price=strike_prices[i],
                                                    option_type=option_type,
                                                    # initial_x_t=initial_x_t,
                                                    # initial_y_t=initial_y_t,
                                                    bond_price_bond_maturity=bond_Ss[i],
                                                    bond_price_option_maturity=bond_Ts[i]
                       )
                       for i in range(len(taus_option))]

            return squared_error(market_quote_bond_option_prices, g2price, weights)

        # a, sigma, b, eta, rho
        initial_guess = np.array([0.11, 0.05, 0.11, 0.04, 0.8, ])
        bounds = np.array([(-10, 10), (0.001, 0.999), (-10, 10),
                           (0.001, 0.999), (-0.999, 0.999),
                           ])
        res = optimize_tool(_target_func, initial_guess, bounds)

        a, sigma, b, eta, rho = res.x
        return a, sigma, b, eta, rho
