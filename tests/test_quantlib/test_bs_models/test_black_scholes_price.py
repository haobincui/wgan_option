import sys
import unittest
from datetime import date
from pathlib import Path

import torch

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_price, \
    black_scholes_price_torch
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaEuropean
from quantlib.calculation.analytics.position.pricer.config import blackscholes_default_config
from quantlib.calculation.analytics.position.pricer.pricer import BlackScholesValuationModelSingleAssetAnalytic

from quantlib.calendar.daycount import act_365


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


class TestBlackScholesPrice(unittest.TestCase):

    def test_vanilla_european_price(self):
        spot = 100
        strike = 100
        vol = 0.5
        r = 0.05
        q = 0.03
        option_type = OptionType.CALL
        valuation_date = date(2022, 1, 1)
        expiration_date = date(2023, 1, 1)
        day_count = act_365
        config = blackscholes_default_config
        requests = [RiskMeasure.PV, RiskMeasure.DELTA, RiskMeasure.GAMMA]

        instrument = VanillaEuropean(underlying='test',
                                     strike=strike,
                                     option_type=option_type,
                                     expiration_date=expiration_date,
                                     delivery_date=expiration_date)

        pricer = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument,
                                                               spot=spot,
                                                               vol=vol,
                                                               r=r,
                                                               q=q,
                                                               quantity=1,
                                                               valuation_date=valuation_date,
                                                               config=config)

        res = pricer.calc_many(requests)
        target_res = {
            RiskMeasure.PV: 19.637645572391815,
            RiskMeasure.DELTA: 0.5947559298898284,
            RiskMeasure.GAMMA: 0.007563321669557865,

        }
        for request in requests:
            self.assertAlmostEqual(target_res[request], res[request], delta=1e-14)

    def test_black_scholes_density(self):
        spot = 100
        strike = 100
        vol = 0.5
        r = 0.05
        q = 0.03
        option_type = OptionType.CALL
        valuation_date = date(2023, 10, 9)
        expiration_date = date(2023, 10, 13)
        config = blackscholes_default_config
        requests = [RiskMeasure.DENSITY]

        instrument = VanillaEuropean(underlying='test',
                                     strike=strike,
                                     option_type=option_type,
                                     expiration_date=expiration_date,
                                     delivery_date=expiration_date)

        pricer = BlackScholesValuationModelSingleAssetAnalytic(instrument=instrument,
                                                               spot=spot,
                                                               vol=vol,
                                                               r=r,
                                                               q=q,
                                                               quantity=1,
                                                               valuation_date=valuation_date,
                                                               config=config)

        res_1 = pricer.calc_many(requests)

        tau = 4 / 250

        kp = _bs_price(strike=strike + config.dk, option_type=option_type, spot=spot, vol=vol, tau=tau, r=r, q=q)
        k = _bs_price(strike=strike, option_type=option_type, spot=spot, vol=vol, tau=tau, r=r, q=q)
        km = _bs_price(strike=strike - config.dk, option_type=option_type, spot=spot, vol=vol, tau=tau, r=r, q=q)

        density = (kp - 2 * k + km) / (config.dk ** 2)

        self.assertAlmostEqual(res_1[RiskMeasure.DENSITY], density, delta=1e-14)


    def test_black_scholes_torch(self):
        spot = 100
        strike = 100
        vol = 0.5
        r = 0.05
        q = 0.03
        tau = 10/252

        target_price = _bs_price(
            strike=strike,
            option_type=OptionType.CALL,
            spot=spot,
            vol=vol,
            tau=tau,
            r=r,
            q=q,
        )
        strike = torch.Tensor([strike])
        spot = torch.Tensor([spot])
        vol = torch.Tensor([vol])
        tau = torch.Tensor([tau])
        r = torch.Tensor([r])
        q = torch.Tensor([q])
        device = 'cpu'
        torch_price = _bs_price_torch(
            strike=strike,
            option_type=[OptionType.CALL],
            spot=spot,
            vol=vol,
            tau=tau,
            r=r,
            q=q,
            device=device,
        )

        self.assertAlmostEqual(target_price, torch_price, delta=1e-4)


    def test_black_scholes_torch_tensor_array(self):
        spots = 100
        strikes = [90, 100, 110, 120]
        vol = [0.3, 0.4, 0.5, 0.6]
        r = 0.05
        q = 0.05
        tau = [5/252, 10/252, 50/252, 250/252]

        prices = [
            _bs_price(
                strike=strike,
                option_type=OptionType.CALL,
                spot=spots,
                vol=vol_value,
                tau=t,
                r=r,
                q=q,
            )
            for strike, vol_value, t in zip(strikes, vol, tau)
        ]

        strikes = torch.Tensor(strikes)
        spots = torch.Tensor([spots])
        vol = torch.Tensor(vol)
        tau = torch.Tensor(tau)
        r = torch.Tensor([r])
        q = torch.Tensor([q])
        device = 'cpu'
        torch_prices = _bs_price_torch(
            strike=strikes,
            option_type=[OptionType.CALL],
            spot=spots,
            vol=vol,
            tau=tau,
            r=r,
            q=q,
            device=device,
        ).cpu().numpy()

        for c_price, t_price in zip(prices, torch_prices):
            self.assertAlmostEqual(c_price, t_price, delta=1e-5)







