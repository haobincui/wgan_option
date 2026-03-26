import sys
import unittest
from datetime import date
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.models.analytical.interest_rate.formula import vasicek_bond_option_price, \
    cir_bond_option_price, cirpp_bond_option_price
from quantlib.calculation.analytics.position.instruments.features import OptionType


class TestShortRateModelZBOptionPrice(unittest.TestCase):
    """
    test for short rate model pricing formulas
    """

    def test_vasicek_model_price(self):
        mean_revert_speed = 0.05
        option_maturity = date(2023, 7, 18)
        bond_maturity = date(2023, 9, 20)

        bond_price_bond_maturity = 0.977505
        bond_price_option_maturity = 0.986878

        option_type = OptionType.CALL
        strike_price = bond_price_option_maturity

        vol = 40 / 100
        valuation_date = date(2023, 4, 18)

        tau_option = (option_maturity - valuation_date).days / 365
        tau_bond = (bond_maturity - valuation_date).days / 365

        price = vasicek_bond_option_price(strike_price, option_type, mean_revert_speed,
                                          bond_price_bond_maturity, bond_price_option_maturity,
                                          vol, tau_option, tau_bond)
        self.assertAlmostEqual(price, 56.975462170218776, delta=1e-6)



    def test_cir_price(self):
        mean_revert_speed = 0.05
        option_maturity = date(2023, 7, 18)
        bond_maturity = date(2023, 9, 20)

        bond_price_bond_maturity = 0.977505
        bond_price_option_maturity = 0.986878

        option_type = OptionType.CALL
        strike_price = 0.95

        vol = 20 / 100
        valuation_date = date(2023, 4, 18)
        long_term_rate = 4 / 100

        tau_option = (option_maturity - valuation_date).days / 365
        tau_bond = (bond_maturity - valuation_date).days / 365
        r_t = 3 / 100

        price = cir_bond_option_price(strike_price, option_type,
                                      long_term_rate, mean_revert_speed, r_t,
                                      bond_price_bond_maturity, bond_price_option_maturity,
                                      vol, tau_option, tau_bond)
        self.assertAlmostEqual(price, 0.0, delta=1e-6)

    def test_cirpp_price(self):
        mean_revert_speed = 3
        option_maturity = date(2023, 7, 18)
        bond_maturity = date(2023, 9, 20)

        bond_price_bond_maturity = 0.977505
        bond_price_option_maturity = 0.986878

        option_type = OptionType.CALL
        strike_price = bond_price_option_maturity

        vol = 40 / 100
        valuation_date = date(2023, 4, 18)
        long_term_rate = 3.5 / 100

        tau_option = (option_maturity - valuation_date).days / 365
        tau_bond = (bond_maturity - valuation_date).days / 365

        tau_valuation_date = 0
        market_bond_price_bond_maturity = bond_price_bond_maturity
        market_bond_price_option_maturity = bond_price_option_maturity
        market_bond_price_valuation_date = 1
        initial_short_rate = 0

        price = cirpp_bond_option_price(strike_price=strike_price,
                                        option_type=option_type,
                                        long_term_rate=long_term_rate,
                                        mean_revert_speed=mean_revert_speed,
                                        vol=vol,
                                        tau_option=tau_option,
                                        tau_bond=tau_bond,
                                        tau_valuation_date=tau_valuation_date,
                                        market_bond_price_bond_maturity=market_bond_price_bond_maturity,
                                        market_bond_price_option_maturity=market_bond_price_option_maturity,
                                        market_bond_price_valuation_date=market_bond_price_valuation_date,
                                        initial_short_rate=initial_short_rate,
                                        bond_price_option_maturity=bond_price_option_maturity,
                                        bond_price_bond_maturity=bond_price_bond_maturity,
                                        initial_x=initial_short_rate)

        # print(price)

        # TODO: check the trade_price
    def test_g2_price(self):
        mean_revert_speed = 3
        option_maturity = date(2023, 7, 18)
        bond_maturity = date(2023, 9, 20)

        bond_price_bond_maturity = 0.977505
        bond_price_option_maturity = 0.986878

        option_type = OptionType.CALL
        strike_price = bond_price_option_maturity

        vol = 40 / 100
        valuation_date = date(2023, 4, 18)
        long_term_rate = 3.5 / 100

        tau_option = (option_maturity - valuation_date).days / 365
        tau_bond = (bond_maturity - valuation_date).days / 365

        tau_valuation_date = 0

        vol_1 = vol_2 = vol
        mean_revert_speed_1 = mean_revert_speed_2 = mean_revert_speed
        correlation = 1

        # trade_price = gaussian_2_bond_option_price(mean_revert_speed_1, vol_1,
        #                                      mean_revert_speed_2, vol_2,
        #                                      correlation,
        #                                      tau_bond, tau_option, tau_valuation_date,
        #                                      strike_price, option_type,
        #                                      bond_price_bond_maturity,
        #                                      bond_price_option_maturity)
        # print(price)

        # TODO: check the trade_price
