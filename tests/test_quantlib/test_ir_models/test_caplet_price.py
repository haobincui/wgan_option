import sys
import unittest
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.models.analytical.interest_rate.formula import caplet_black_price, caplet_black_vol


class TestCapletPrice(unittest.TestCase):

    def test_caplet_price(self):
        forward_rate = 0.04
        strike_rate = 0.045
        option_tau = 1
        vol = 0.02
        bound_price = 10
        notional_amount = 1000000
        price = caplet_black_price(
            forward_rate=forward_rate,
            strike_rate=strike_rate,
            vol=vol,
            bond_price=bound_price,
            reset_tau=option_tau,
            option_tau=notional_amount,
        )
        target_price = 0.4
        self.assertAlmostEqual(price, target_price, delta= 1e-14)

    def test_caplet_vol(self):
        forward_rate = 0.04
        strike_rate = 0.045
        option_tau = 1
        target_vol = 0.02
        notional_amount = 1000000
        bound_price = 10
        # caplet_price = 0.02
        caplet_price = caplet_black_price(
            forward_rate=forward_rate,
            strike_rate=strike_rate,
            vol=target_vol,
            bond_price=bound_price,
            reset_tau=option_tau,
            option_tau=option_tau,
            notional_amount=notional_amount,
        )

        vol = caplet_black_vol(
            caplet_price=caplet_price,
            forward_rate=forward_rate,
            strike_rate=strike_rate,
            bond_price=bound_price,
            reset_tau=option_tau,
            option_tau=option_tau,
            notional_amount=notional_amount,
        )
        self.assertAlmostEqual(target_vol, vol, delta=1e-6)
