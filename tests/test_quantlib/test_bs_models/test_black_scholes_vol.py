import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy import optimize
from scipy.optimize import brentq

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.models.math_tools.least_square import squared_error
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_price, black_scholes_vega, \
    black_scholes_implied_vol_torch, black_scholes_price_torch, black_scholes_implied_vol
from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaEuropean, \
    VanillaImpliedVolatility
from quantlib.calculation.analytics.position.pricer.config import blackscholes_default_config
from quantlib.calculation.analytics.position.pricer.pricer import BlackScholesValuationModelSingleAssetAnalytic


def _bs_price(*, strike, option_type, spot, vol, tau, r, q):
    return black_scholes_price(
        strike=strike,
        option_type=option_type,
        spot=spot,
        vol=vol,
        tau=tau,
        r=r,
        q=q,
    )


def _bs_vega(*, strike, option_type, spot, vol, tau, r, q):
    return black_scholes_vega(
        strike=strike,
        option_type=option_type,
        spot=spot,
        vol=vol,
        tau=tau,
        r=r,
        q=q,
    )


def _bs_iv(*, price, strike, option_type, spot, tau, r, q):
    return black_scholes_implied_vol(
        price=price,
        strike=strike,
        option_type=option_type,
        spot=spot,
        tau=tau,
        r=r,
        q=q,
    )


def _bs_price_torch(*, strike, option_type, spot, vol, tau, r, q, device):
    return black_scholes_price_torch(
        strike=strike,
        option_type=option_type,
        spot=spot,
        vol=vol,
        tau=tau,
        r=r,
        q=q,
        device=device,
    )


def _bs_iv_torch(*, price, strike, option_type, spot, tau, r, q, device):
    return black_scholes_implied_vol_torch(
        price=price,
        strike=strike,
        option_type=option_type,
        spot=spot,
        tau=tau,
        r=r,
        q=q,
        device=device,
    )


class TestBlackScholesVol(unittest.TestCase):

    def test_black_scholes_vol_brentq(self):
        spot = 100
        strike = 100
        r = 0.05
        q = 0.03
        option_type = OptionType.CALL
        tau = 20 / 252

        tau_1 = 10 / 252
        price_tau_1 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.4, tau=tau_1, r=r, q=q)
        res_tau_1 = brentq(lambda x: price_tau_1 - _bs_price(strike=strike, option_type=option_type, spot=spot, vol=x, tau=tau_1, r=r, q=q),
                           0.001, 10)
        self.assertAlmostEqual(0.4, res_tau_1, delta=1e-9)

        tau_2 = 200 / 252
        price_tau_2 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.4, tau=tau_2, r=r, q=q)
        res_tau_2 = brentq(lambda x: price_tau_2 - _bs_price(strike=strike, option_type=option_type, spot=spot, vol=x, tau=tau_2, r=r, q=q),
                           0.001, 10)
        self.assertAlmostEqual(0.4, res_tau_2, delta=1e-9)

        tau_3 = 3000 / 252
        price_tau_3 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.4, tau=tau_3, r=r, q=q)
        res_tau_3 = brentq(lambda x: price_tau_3 - _bs_price(strike=strike, option_type=option_type, spot=spot, vol=x, tau=tau_3, r=r, q=q),
                           0.001, 10)
        self.assertAlmostEqual(0.4, res_tau_3, delta=1e-9)

        strike_1 = 80
        price_strike_1 = _bs_price(strike=strike_1, option_type=option_type, spot=spot, vol=0.4, tau=tau, r=r, q=q)

        res_strike_1 = brentq(lambda x: price_strike_1 - _bs_price(strike=strike_1, option_type=option_type, spot=spot, vol=x, tau=tau, r=r, q=q),
                              0.001, 10)
        self.assertAlmostEqual(0.4, res_strike_1, delta=1e-9)

        strike_2 = 120
        price_strike_2 = _bs_price(strike=strike_2, option_type=option_type, spot=spot, vol=0.4, tau=tau, r=r, q=q)
        res_strike_2 = brentq(lambda x: price_strike_2 - _bs_price(strike=strike_2, option_type=option_type, spot=spot, vol=x, tau=tau, r=r, q=q),
                              0.001, 10)
        self.assertAlmostEqual(0.4, res_strike_2, delta=1e-9)

        strike_3 = 180
        price_strike_3 = _bs_price(strike=strike_3, option_type=option_type, spot=spot, vol=0.4, tau=tau, r=r, q=q)
        res_strike_3 = brentq(lambda x: price_strike_3 - _bs_price(strike=strike_3, option_type=option_type, spot=spot, vol=x, tau=tau, r=r, q=q),
                              0, 1)
        self.assertAlmostEqual(0.4, res_strike_3, delta=1e-9)

    def test_black_scholes_vol_newton(self):
        def find_vol(price, strike, option_type, spot, tau, r, q):
            def _target_func(param):
                current_vol = param
                current_price = _bs_price(
                    strike=strike,
                    option_type=option_type,
                    spot=spot,
                    vol=current_vol,
                    tau=tau,
                    r=r,
                    q=q,
                )
                return squared_error(price, current_price)

            def _gradian(param):
                current_vol = param
                return -2 * (
                    price
                    - _bs_price(
                        strike=strike,
                        option_type=option_type,
                        spot=spot,
                        vol=current_vol,
                        tau=tau,
                        r=r,
                        q=q,
                    )
                ) * _bs_vega(
                    strike=strike,
                    option_type=option_type,
                    spot=spot,
                    vol=current_vol,
                    tau=tau,
                    r=r,
                    q=q,
                )

            initial_guess = np.array([1])

            res = optimize.fsolve(_target_func, initial_guess,
                                  fprime=_gradian,
                                  xtol=1e-6)

            return float(*res)

        spot = 100
        strike = 100
        r = 0.05
        q = 0.03
        option_type = OptionType.CALL
        tau = 20 / 252
        price = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=9, tau=tau, r=r, q=q)

        res_tau = find_vol(price, strike, option_type, spot, tau, r, q)

        self.assertAlmostEqual(9, res_tau, delta=1e-6)

        tau_1 = 10 / 252
        price_tau_1 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.1, tau=tau_1, r=r, q=q)
        res_tau_1 = find_vol(price_tau_1, strike, option_type, spot, tau_1, r, q)
        self.assertAlmostEqual(0.1, res_tau_1, delta=1e-6)

        tau_2 = 80 / 252
        price_tau_2 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.4, tau=tau_2, r=r, q=q)
        res_tau_2 = find_vol(price_tau_2, strike, option_type, spot, tau_2, r, q)
        self.assertAlmostEqual(0.4, res_tau_2, delta=1e-6)

        tau_3 = 3000 / 252
        price_tau_3 = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=0.9, tau=tau_3, r=r, q=q)
        res_tau_3 = find_vol(price_tau_3, strike, option_type, spot, tau_3, r, q)
        self.assertAlmostEqual(0.9, res_tau_3, delta=1e-6)

        strike_1 = 80
        price_strike_1 = _bs_price(strike=strike_1, option_type=option_type, spot=spot, vol=0.6, tau=tau, r=r, q=q)
        res_strike_1 = find_vol(price_strike_1, strike_1, option_type, spot, tau, r, q)
        self.assertAlmostEqual(0.6, res_strike_1, delta=1e-6)

        strike_2 = 120
        price_strike_2 = _bs_price(strike=strike_2, option_type=option_type, spot=spot, vol=0.1, tau=tau, r=r, q=q)
        res_strike_2 = find_vol(price_strike_2, strike_2, option_type, spot, tau, r, q)
        self.assertAlmostEqual(0.1, res_strike_2, delta=1e-6)

        strike_3 = 180
        price_strike_3 = _bs_price(strike=strike_3, option_type=option_type, spot=spot, vol=0.4, tau=tau, r=r, q=q)
        res_strike_3 = find_vol(price_strike_3, strike_3, option_type, spot, tau, r, q)
        self.assertAlmostEqual(0.4, res_strike_3, delta=1e-6)

    def test_black_scholes_find_vol(self):
        spot = 100
        strike = 100
        r = 0.05
        q = 0.03
        option_type = OptionType.CALL
        valuation_date = date(2023, 10, 9)
        expiration_date = date(2027, 10, 13)
        config = blackscholes_default_config

        def _get_recalc_vol(target_vol):
            instrument_eur = VanillaEuropean(underlying='test',
                                             strike=strike,
                                             option_type=option_type,
                                             expiration_date=expiration_date,
                                             delivery_date=expiration_date)

            pricer_vol = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument_eur,
                                                                       spot=spot,
                                                                       vol=target_vol,
                                                                       r=r,
                                                                       q=q,
                                                                       quantity=1,
                                                                       valuation_date=valuation_date,
                                                                       config=config)

            instrument = VanillaImpliedVolatility(underlying='test',
                                                  price=pricer_vol.pv(),
                                                  strike=strike,
                                                  option_type=option_type,
                                                  expiration_date=expiration_date,
                                                  delivery_date=expiration_date)

            pricer = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument,
                                                                   spot=spot,
                                                                   vol=0.1,
                                                                   r=r,
                                                                   q=q,
                                                                   quantity=1,
                                                                   valuation_date=valuation_date,
                                                                   config=config)

            return pricer.pv()

        target_vols = [0.1, 0.3, 0.8, 1.8, 7]

        res_vols = [_get_recalc_vol(target_vol) for target_vol in target_vols]
        for vol, res_vol in zip(target_vols, res_vols):
            self.assertAlmostEqual(vol, res_vol, delta=1e-5)

    def test_black_scholes_find_vol_tau(self):
        spot = 100
        r = 0.05
        q = 0.03
        option_type = [OptionType.CALL]
        valuation_date = date(2023, 10, 9)

        config = blackscholes_default_config

        def _get_recalc_vol(expiration_date, strike, target_vol):
            instrument_eur = VanillaEuropean(underlying='test',
                                             strike=strike,
                                             option_type=option_type,
                                             expiration_date=expiration_date,
                                             delivery_date=expiration_date)

            pricer_vol = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument_eur,
                                                                       spot=spot,
                                                                       vol=target_vol,
                                                                       r=r,
                                                                       q=q,
                                                                       quantity=1,
                                                                       valuation_date=valuation_date,
                                                                       config=config)

            instrument = VanillaImpliedVolatility(underlying='test',
                                                  price=pricer_vol.pv(),
                                                  strike=strike,
                                                  option_type=option_type,
                                                  expiration_date=expiration_date,
                                                  delivery_date=expiration_date)

            pricer = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument,
                                                                   spot=spot,
                                                                   vol=0.1,
                                                                   r=r,
                                                                   q=q,
                                                                   quantity=1,
                                                                   valuation_date=valuation_date,
                                                                   config=config)

            return pricer.pv()

        expiration_dates = [date(2023, 12, 1), date(2024, 11, 10), date(2030, 5, 1), date(2040, 1, 6), date(2045, 1, 9)]
        target_vol = 0.4

        res_vols_dates = [_get_recalc_vol(expiration_date, 100, target_vol) for expiration_date in expiration_dates]
        for vol in res_vols_dates:
            self.assertAlmostEqual(target_vol, vol, delta=1e-6)

        strikes = [50, 80, 100, 120, 180]
        res_vols_strikes = [_get_recalc_vol(expiration_dates[0], strike, target_vol) for strike in strikes]
        for vol in res_vols_strikes:
            self.assertAlmostEqual(target_vol, vol, delta=1e-6)

        res_vols_strikes_dates = [_get_recalc_vol(expiration_date, strike, target_vol)
                                  for expiration_date, strike in zip(expiration_dates, strikes)]
        for vol in res_vols_strikes_dates:
            self.assertAlmostEqual(target_vol, vol, delta=1e-6)

    def test_black_scholes_vol_torch(self):
        find_vol = black_scholes_implied_vol_torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)

        spot = 100
        spot = torch.tensor([spot], device=device)
        strike = 100
        strike = torch.tensor([strike], device=device)
        r = 0.05
        r = torch.tensor([r], device=device)
        q = 0.03
        q = torch.tensor([q], device=device)
        option_type = [OptionType.CALL]
        tau = 20 / 252
        tau = torch.tensor([tau], device=device)

        price = _bs_price_torch(
            strike=strike,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([9], device=device),
            tau=tau,
            r=r,
            q=q,
            device=device,
        )

        res_tau = find_vol(price, strike, option_type, spot, tau, r, q, device).cpu().numpy()

        self.assertAlmostEqual(9, res_tau, delta=1e-5)

        tau_1 = 10 / 252
        tau_1 = torch.tensor([tau_1], device=device)
        price_tau_1 = _bs_price_torch(
            strike=strike,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([0.1], device=device),
            tau=tau_1,
            r=r,
            q=q,
            device=device,
        )
        res_tau_1 = find_vol(price_tau_1, strike, option_type, spot, tau_1, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.1, res_tau_1, delta=1e-5)

        tau_2 = 80 / 252
        tau_2 = torch.tensor([tau_2], device=device)
        price_tau_2 = _bs_price_torch(
            strike=strike,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([0.4], device=device),
            tau=tau_2,
            r=r,
            q=q,
            device=device,
        )
        res_tau_2 = find_vol(price_tau_2, strike, option_type, spot, tau_2, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.4, res_tau_2, delta=1e-5)

        tau_3 = 3000 / 252
        tau_3 = torch.tensor([tau_3], device=device)
        price_tau_3 = _bs_price_torch(
            strike=strike,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([0.9], device=device),
            tau=tau_3,
            r=r,
            q=q,
            device=device,
        )
        res_tau_3 = find_vol(price_tau_3, strike, option_type, spot, tau_3, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.9, res_tau_3, delta=1e-5)

        strike_1 = 80
        strike_1 = torch.tensor([strike_1], device=device)
        price_strike_1 = _bs_price_torch(
            strike=strike_1,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([0.6], device=device),
            tau=tau,
            r=r,
            q=q,
            device=device,
        )
        res_strike_1 = find_vol(price_strike_1, strike_1, option_type, spot, tau, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.6, res_strike_1, delta=1e-5)

        strike_2 = 120 / 100
        strike_2 = torch.tensor([strike_2], device=device)
        price_strike_2 = _bs_price_torch(
            strike=strike_2,
            option_type=option_type,
            spot=spot / 100,
            vol=torch.tensor([0.1], device=device),
            tau=tau,
            r=r,
            q=q,
            device=device,
        )
        res_strike_2 = find_vol(price_strike_2, strike_2, option_type, spot / 100, tau, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.1, res_strike_2, delta=1e-1)

        strike_3 = 180
        strike_3 = torch.tensor([strike_3], device=device)
        price_strike_3 = _bs_price_torch(
            strike=strike_3,
            option_type=option_type,
            spot=spot,
            vol=torch.tensor([0.4], device=device),
            tau=tau,
            r=r,
            q=q,
            device=device,
        )
        res_strike_3 = find_vol(price_strike_3, strike_3, option_type, spot, tau, r, q, device).cpu().numpy()
        self.assertAlmostEqual(0.4, res_strike_3, delta=1e-5)

    def test_black_scholes_vol_torch_nan(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        nan = float('nan')

        prices = [nan, nan, nan, 0.0800, 0.1100, 0.1100, 0.1100, 0.1300, 0.1300, 0.1300, 0.0800, 0.0800,
                  0.0800, nan, 0.0800, nan]
        prices_torch = torch.tensor(prices, dtype=torch.float64, device=device)

        taus = [nan, nan, nan, 10.2612, 10.2611, 10.2611, 10.2611, 10.2610, 10.2610, 10.2610, 10.2608,
                10.2608, 10.2608, nan, 10.2608, nan]

        taus_torch = torch.tensor(taus, dtype=torch.float64, device=device)

        strikes = [i / 100 for i in
                   [nan, nan, nan, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000,
                    99.5000, 99.5000, nan, 99.5000, nan]]
        strikes_torch = torch.tensor(strikes, dtype=torch.float64, device=device)

        spots = [100 / 100]
        spots_torch = torch.tensor(spots, dtype=torch.float64, device=device)

        r = [0.05]
        r_torch = torch.tensor(r, dtype=torch.float64, device=device)
        q = [0.05]
        q_torch = torch.tensor(q, dtype=torch.float64, device=device)

        option_type_call = OptionType.CALL
        option_type_call_torch = torch.tensor([option_type_call.value], dtype=torch.bool, device=device)
        option_type_put = OptionType.PUT
        option_type_put_torch = torch.tensor([option_type_put.value], dtype=torch.bool, device=device)

        call_vols_torch = _bs_iv_torch(
            price=prices_torch,
            strike=strikes_torch,
            option_type=option_type_call_torch,
            spot=spots_torch,
            tau=taus_torch,
            r=r_torch,
            q=q_torch,
            device=device,
        )

        put_vols_torch = _bs_iv_torch(
            price=prices_torch,
            strike=strikes_torch,
            option_type=option_type_put_torch,
            spot=spots_torch,
            tau=taus_torch,
            r=r_torch,
            q=q_torch,
            device=device,
        )

        target_call_vols = torch.tensor([nan, nan, nan, 0.1033, 0.1434, 0.1434, 0.1434, 0.1705, 0.1705,
                                         0.1705, 0.1033, 0.1033, 0.1033, nan, 0.1033, nan], device=device)
        target_put_vols = torch.tensor([nan, nan, nan, 0.1073, 0.1475, 0.1475, 0.1475, 0.1746, 0.1746,
                                        0.1746, 0.1073, 0.1073, 0.1073, nan, 0.1073, nan], device=device)

        call_close = torch.isclose(target_call_vols, call_vols_torch, atol=1e-4) | \
                     (torch.isnan(target_call_vols) & torch.isnan(call_vols_torch))
        put_close = torch.isclose(target_put_vols, put_vols_torch, atol=1e-4) | \
                    (torch.isnan(target_put_vols) & torch.isnan(put_vols_torch))

        self.assertTrue(torch.all(call_close))
        self.assertTrue(torch.all(put_close))

    def test_black_scholes_vol_torch_cases(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        prices = [0.0800, 0.1100, 0.1100, 0.1100, 0.1300, 0.1300, 0.1300, 0.0800, 0.0800,
                  0.0800, 0.0800]

        prices_torch = torch.tensor(prices, dtype=torch.float64, device=device)

        taus = [10.2612, 10.2611, 10.2611, 10.2611, 10.2610, 10.2610, 10.2610, 10.2608,
                10.2608, 10.2608, 10.2608]

        taus_torch = torch.tensor(taus, dtype=torch.float64, device=device)

        strikes = [i / 100 for i in [99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000, 99.5000,
                                     99.5000, 99.5000, 99.5000]]

        strikes_torch = torch.tensor(strikes, dtype=torch.float64, device=device)

        spots = [100 / 100]
        spots_torch = torch.tensor(spots, dtype=torch.float64, device=device)

        r = [0.05]
        r_torch = torch.tensor(r, dtype=torch.float64, device=device)
        q = [0.05]
        q_torch = torch.tensor(q, dtype=torch.float64, device=device)

        option_type_call = OptionType.CALL
        option_type_call_torch = torch.tensor([option_type_call.value], dtype=torch.bool, device=device)
        option_type_put = OptionType.PUT
        option_type_put_torch = torch.tensor([option_type_put.value], dtype=torch.bool, device=device)

        call_vols_torch = _bs_iv_torch(
            price=prices_torch,
            strike=strikes_torch,
            option_type=option_type_call_torch,
            spot=spots_torch,
            tau=taus_torch,
            r=r_torch,
            q=q_torch,
            device=device,
        )
        call_vols = [
            _bs_iv(
                price=prices[i],
                strike=strikes[i],
                option_type=option_type_call,
                spot=spots[0],
                tau=taus[i],
                r=r[0],
                q=q[0],
            )
            for i in range(len(prices))
        ]

        put_vols_torch = _bs_iv_torch(
            price=prices_torch,
            strike=strikes_torch,
            option_type=option_type_put_torch,
            spot=spots_torch,
            tau=taus_torch,
            r=r_torch,
            q=q_torch,
            device=device,
        )
        put_vols = [
            _bs_iv(
                price=prices[i],
                strike=strikes[i],
                option_type=option_type_put,
                spot=spots[0],
                tau=taus[i],
                r=r[0],
                q=q[0],
            )
            for i in range(len(prices))
        ]

        for i in range(len(prices)):
            self.assertAlmostEqual(call_vols[i], call_vols_torch.cpu().numpy()[i], delta=1e-6)

            self.assertAlmostEqual(put_vols[i], put_vols_torch.cpu().numpy()[i], delta=1e-6)
