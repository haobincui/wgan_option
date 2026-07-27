from __future__ import annotations

import unittest
from datetime import date

from wgan_option.market.treasury_options import (
    cme_treasury_option_last_trading_date,
    cme_treasury_option_last_trading_datetime,
    resolve_contract_year,
    resolve_ty_option_contract_dates,
)


class TreasuryOptionConventionTests(unittest.TestCase):
    def test_rule_19a_monthly_dates(self) -> None:
        self.assertEqual(
            cme_treasury_option_last_trading_date(2022, 3),
            date(2022, 2, 18),
        )
        self.assertEqual(
            cme_treasury_option_last_trading_date(2023, 3),
            date(2023, 2, 24),
        )
        self.assertEqual(
            cme_treasury_option_last_trading_date(2023, 12),
            date(2023, 11, 24),
        )

    def test_chicago_dst_is_converted_to_utc(self) -> None:
        winter = cme_treasury_option_last_trading_datetime(2023, 3)
        summer = cme_treasury_option_last_trading_datetime(2023, 9)
        self.assertEqual(winter.hour, 22)
        self.assertEqual(summer.hour, 21)

    def test_year_resolution_depends_only_on_trade_and_symbol(self) -> None:
        self.assertEqual(
            resolve_contract_year(2, trade_date=date(2022, 1, 3), contract_month=3),
            2022,
        )
        self.assertEqual(
            resolve_contract_year(3, trade_date=date(2022, 12, 15), contract_month=3),
            2023,
        )
        self.assertEqual(
            resolve_contract_year(23, trade_date=date(2022, 12, 15), contract_month=3),
            2023,
        )

    def test_ty_serial_option_maps_to_quarterly_future(self) -> None:
        contract = resolve_ty_option_contract_dates(
            option_month_code="B",
            option_year_code="3",
            trade_date=date(2023, 1, 10),
        )
        self.assertEqual(contract.named_month, 2)
        self.assertEqual(contract.named_year, 2023)
        self.assertEqual(contract.underlying_future_month_code, "H")
        self.assertEqual(contract.underlying_future_year, 2023)

    def test_unknown_option_month_code_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported TY option month"):
            resolve_ty_option_contract_dates(
                option_month_code="Y",
                option_year_code="3",
                trade_date=date(2023, 1, 10),
            )


if __name__ == "__main__":
    unittest.main()
