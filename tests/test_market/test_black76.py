from __future__ import annotations

import unittest

from wgan_option.market.black76 import (
    black76_implied_vol,
    black76_no_arbitrage_bounds,
    black76_price,
)


class Black76Tests(unittest.TestCase):
    def test_price_and_implied_vol_round_trip(self) -> None:
        for option_type, strike in (("CALL", 112.0), ("PUT", 116.0)):
            expected_vol = 0.183
            price = black76_price(
                futures_price=114.0,
                strike=strike,
                tau=45.0 / 365.0,
                volatility=expected_vol,
                discount_factor=0.992,
                option_type=option_type,
            )
            actual_vol = black76_implied_vol(
                price=price,
                futures_price=114.0,
                strike=strike,
                tau=45.0 / 365.0,
                discount_factor=0.992,
                option_type=option_type,
            )
            self.assertAlmostEqual(actual_vol, expected_vol, places=10)

    def test_no_arbitrage_bounds_are_enforced(self) -> None:
        lower, upper = black76_no_arbitrage_bounds(
            futures_price=114.0,
            strike=110.0,
            discount_factor=0.99,
            option_type="CALL",
        )
        self.assertAlmostEqual(lower, 3.96)
        self.assertAlmostEqual(upper, 112.86)
        with self.assertRaisesRegex(ValueError, "violates Black-76 bounds"):
            black76_implied_vol(
                price=3.0,
                futures_price=114.0,
                strike=110.0,
                tau=30.0 / 365.0,
                discount_factor=0.99,
                option_type="CALL",
            )

    def test_zero_vol_returns_discounted_intrinsic(self) -> None:
        self.assertAlmostEqual(
            black76_price(
                futures_price=114.0,
                strike=110.0,
                tau=0.5,
                volatility=0.0,
                discount_factor=0.98,
                option_type="CALL",
            ),
            3.92,
        )


if __name__ == "__main__":
    unittest.main()
