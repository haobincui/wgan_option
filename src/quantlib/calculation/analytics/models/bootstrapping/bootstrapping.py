"""
calibration for yield curve
"""
import enum
from datetime import date
from typing import List

import numpy as np
from scipy.optimize import brentq

from quantlib.calculation.analytics.models.analytical.interest_rate.formula import caplet_black_price, caplet_black_vol
from quantlib.calculation.analytics.models.math_tools.least_square import squared_error
from quantlib.calculation.analytics.models.math_tools.optimize_tool import optimize_tool
from quantlib.calendar.daycount import act_365
from quantlib.calendar.schedule import Period, plus_period


class VolModelType(enum.Enum):
    Black = 0
    Normal = 1


class Bootstrapping:
    # @staticmethod
    # def get_cap_prices(cap_vols: Optional[List[float]],
    #                    caplet_prices: Optional[List[float]],
    #                    ):
    #     pass

    # @staticmethod
    # def get_caplet_vols_by_caplet_prices(caplet_prices: List[float],
    #                                      expiration_dates: List[date],
    #                                      valuation_date: date,
    #                                      strike_rate: float,
    #                                      forward_rates: List[float],
    #                                      discount_factors: List[float],
    #                                      reset_timedelta: Period,
    #                                      model_type: VolModelType = VolModelType.Black) -> List[float]:
    #     daycount = act_365
    #     cap_prices = []
    #     taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
    #     reset_dates = [plus_period(
    #         expiration_date, reset_timedelta
    #     ) for expiration_date in expiration_dates]
    #
    #     reset_taus = [daycount(
    #         expiration_date, reset_date
    #     ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]
    #
    #     caplet_vols = [caplet_black_vol(
    #         caplet_prices[i], forward_rates[i], strike_rate, discount_factors[i], reset_taus[i], taus[i]
    #     ) for i in range(len(caplet_prices))]
    #     return caplet_vols

    # @staticmethod
    # def get_cap_vols():
    #     pass

    @staticmethod
    def get_cap_prices_by_cap_vols(cap_vols: List[float],
                                   expiration_dates: List[date],
                                   valuation_date: date,
                                   strike_rate: float,
                                   forward_rates: List[float],
                                   discount_factors: List[float],
                                   reset_timedelta: Period,
                                   model_type: VolModelType = VolModelType.Black) -> List[float]:
        """
        get cap prices by cap vols
        :param cap_vols:
        :param expiration_dates:
        :param valuation_date:
        :param strike_rate:
        :param forward_rates:
        :param discount_factors:
        :param reset_timedelta:
        :param model_type:
        :return:
        """

        daycount = act_365
        cap_prices = []
        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        for idx in range(len(cap_vols)):
            cap_price = np.sum(
                [
                    caplet_black_price(
                        forward_rate=forward_rates[i],
                        strike_rate=strike_rate,
                        vol=cap_vols[idx],
                        bond_price=discount_factors[idx],
                        reset_tau=reset_taus[i],
                        option_tau=taus[i],
                    )
                    for i in range(idx + 1)
                ]
            )
            cap_prices.append(float(cap_price))

        return cap_prices

    @staticmethod
    def get_caplet_prices_by_cap_vols(cap_vols: List[float],
                                      expiration_dates: List[date],
                                      valuation_date: date,
                                      strike_rate: float,
                                      forward_rates: List[float],
                                      discount_factors: List[float],
                                      reset_timedelta: Period,
                                      model_type: VolModelType = VolModelType.Black) -> List[float]:
        """

        :param cap_vols:
        :param expiration_dates:
        :param valuation_date:
        :param strike_rate:
        :param forward_rates:
        :param discount_factors:
        :param reset_timedelta:
        :param model_type:
        :return:
        """
        daycount = act_365
        caplet_prices = []
        cap_prices = []
        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        for idx in range(len(cap_vols)):
            cap_price = np.sum(
                [
                    caplet_black_price(
                        forward_rate=forward_rates[i],
                        strike_rate=strike_rate,
                        vol=cap_vols[idx],
                        bond_price=discount_factors[idx],
                        reset_tau=reset_taus[i],
                        option_tau=taus[i],
                    )
                    for i in range(idx + 1)
                ]
            )
            cap_prices.append(cap_price)

            if idx == 0:  # first caplet trade_price = cap trade_price
                caplet_prices.append(float(cap_price))
            else:
                caplet_prices.append(cap_price - cap_prices[-2])

        return caplet_prices

    @staticmethod
    def get_bond_option_prices_by_cap_vols(cap_vols: List[float],
                                           expiration_dates: List[date],
                                           valuation_date: date,
                                           strike_rate: float,
                                           forward_rates: List[float],
                                           discount_factors: List[float],
                                           reset_timedelta: Period,
                                           model_type: VolModelType = VolModelType.Black) -> List[float]:
        """

        :param cap_vols:
        :param expiration_dates:
        :param valuation_date:
        :param strike_rate:
        :param forward_rates:
        :param discount_factors:
        :param reset_timedelta:
        :param model_type:
        :return:
        """
        daycount = act_365
        caplet_prices = Bootstrapping.get_caplet_prices_by_cap_vols(
            cap_vols, expiration_dates, valuation_date, strike_rate, forward_rates,
            discount_factors, reset_timedelta, model_type
        )
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        bond_prices = [caplet_prices[i] / (1 + reset_taus[i] * strike_rate) for i in range(len(cap_vols))]
        return bond_prices

    @staticmethod
    def _get_black_vols(caplet_prices: List[float],
                        expiration_dates: List[date],
                        valuation_date: date,
                        strike_rate: float,
                        forward_rates: List[float],
                        discount_factors: List[float],
                        reset_timedelta: Period,
                        model_type: VolModelType = VolModelType.Black) -> List[float]:
        daycount = act_365

        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        vols = [
            caplet_black_vol(
                caplet_price=caplet_prices[idx],
                forward_rate=forward_rates[idx],
                strike_rate=strike_rate,
                bond_price=discount_factors[idx],
                reset_tau=reset_taus[idx],
                option_tau=taus[idx],
            )
            for idx in range(len(caplet_prices))
        ]

        return vols

    @staticmethod
    def get_caplet_vols_by_cap_vols(cap_vols: List[float],
                                    expiration_dates: List[date],
                                    valuation_date: date,
                                    strike_rate: float,
                                    forward_rates: List[float],
                                    discount_factors: List[float],
                                    reset_timedelta: Period,
                                    model_type: VolModelType = VolModelType.Black) -> List[float]:
        caplet_prices = Bootstrapping.get_caplet_prices_by_cap_vols(
            cap_vols, expiration_dates, valuation_date, strike_rate,
            forward_rates, discount_factors, reset_timedelta, model_type
        )

        caplet_vols = Bootstrapping._get_black_vols(
            caplet_prices, expiration_dates, valuation_date, strike_rate,
            forward_rates, discount_factors, reset_timedelta, model_type
        )
        return caplet_vols

    @staticmethod
    def get_cap_vols_by_caplet_prices(caplet_prices: List[float],
                                      expiration_dates: List[date],
                                      valuation_date: date,
                                      strike_rate: float,
                                      forward_rates: List[float],
                                      discount_factors: List[float],
                                      reset_timedelta: Period,
                                      model_type: VolModelType = VolModelType.Black) -> List[float]:

        cap_prices = np.cumsum(caplet_prices)
        daycount = act_365
        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]


        vols = []
        for idx in range(len(cap_prices)):

            def target_func(vol):
                black_prices = [
                    caplet_black_price(
                        forward_rate=forward_rates[i],
                        strike_rate=strike_rate,
                        vol=vol,
                        bond_price=discount_factors[i],
                        reset_tau=reset_taus[i],
                        option_tau=taus[i],
                    )
                    for i in range(idx + 1)
                ]

                black_price = np.sum(black_prices)

                return squared_error(cap_prices[idx], black_price)

            try:
                vol = brentq(
                    lambda x: cap_prices[idx] - np.sum([caplet_black_price(
                        forward_rate=forward_rates[i],
                        strike_rate=strike_rate,
                        vol=x,
                        bond_price=discount_factors[i],
                        reset_tau=reset_taus[i],
                        option_tau=taus[i],
                    ) for i in range(idx + 1)]), 0, 1
                )
            except:
                initial_guess = np.array([0.4])
                bounds = np.array([(0.001, 10)])
                res = optimize_tool(target_func, initial_guess, bounds)
                vol = res.x

            vols.append(float(vol))

        return vols

    @staticmethod
    def get_bond_option_prices_by_caplet_prices(caplet_prices: List[float],
                                                expiration_dates: List[date],
                                                valuation_date: date,
                                                strike_rate: float,
                                                reset_timedelta: Period) -> List[float]:
        daycount = act_365
        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        bond_option_prices = [caplet_prices[i] / (1 + reset_taus[i] * strike_rate) for i in range(len(reset_taus))]
        return bond_option_prices

    @staticmethod
    def get_caplet_prices_by_bond_option_prices(bond_option_prices: List[float],
                                                expiration_dates: List[date],
                                                valuation_date: date,
                                                strike_rate: float,
                                                reset_timedelta: Period) -> List[float]:
        daycount = act_365
        taus = [daycount(valuation_date, expiration_date) for expiration_date in expiration_dates]
        reset_dates = [
            plus_period(expiration_date, reset_timedelta)
            for expiration_date in expiration_dates
        ]

        reset_taus = [daycount(
            expiration_date, reset_date
        ) for (expiration_date, reset_date) in zip(expiration_dates, reset_dates)]

        caplet_prices = [bond_option_price * (1 + reset_tau * strike_rate)
                         for bond_option_price, reset_tau in zip(bond_option_prices, reset_taus)]
        return caplet_prices
