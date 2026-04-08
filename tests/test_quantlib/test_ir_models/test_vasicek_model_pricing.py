import sys
import unittest
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.models.analytical.interest_rate.formula import vasicek_bond_price, vasicek_bond_option_price
from quantlib.calculation.analytics.position.instruments.features import OptionType


class TestVasicekModelPricing(unittest.TestCase):

    def test_vasicek_bond_price(self):
        mean_revert_speed = 5
        long_term_rate = 0.03
        vol = 0.05
        tau = 20/252
        short_rate = 0.025
        price = vasicek_bond_price(long_term_rate, mean_revert_speed, short_rate, vol, tau)
        self.assertAlmostEqual(price, 0.9979493782968007, delta = 1e-14)


    def test_vasicek_call_option_price(self):
        mean_revert_speed = 5
        long_term_rate = 0.03
        vol = 0.05
        tau_option = 5/252
        tau_bond = 25/252
        short_rate = 0.025
        option_type = OptionType.CALL
        strike_price = 95 / 100
        price = vasicek_bond_option_price(
            strike_price=strike_price,
            option_type=option_type,
            mean_revert_speed=long_term_rate,
            long_term_rate=mean_revert_speed,
            initial_short_rate=short_rate,
            vol=vol,
            tau_option=tau_option,
            tau_bond=tau_bond,
        )
        self.assertAlmostEqual(price, 0.04677906633865181, delta = 1e-14)




