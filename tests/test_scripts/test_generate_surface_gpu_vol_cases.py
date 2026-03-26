import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_data.contract_handler.contract_type import ContractType  # noqa: E402
from market_data.contract_handler.future_contract import FutureContract  # noqa: E402
from market_data.contract_handler.option_contract import OptionContract  # noqa: E402
from market_data.contract_handler.utils import ContractTerminationRule  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from quantlib.calculation.analytics.models.analytical.equity.formula import (  # noqa: E402
    black_scholes_price,
)
from scripts.generate_surface.common.minute_svi_common import (  # noqa: E402
    _make_expiry_dt_utc,
    _tau_years_from_trade_to_expiry,
)
from scripts.generate_surface.surface_cpu.generate_minute_svi_params import (  # noqa: E402
    _compute_implied_vols_cpu,
)
from scripts.generate_surface.surface_gpu.generate_minute_svi_params import (  # noqa: E402
    ContractMeta,
    MinuteOptionCandidate,
    _compute_implied_vols_gpu,
)


class TestGenerateSurfaceGpuVolCases(unittest.TestCase):
    DEVICE = torch.device("cpu")

    def _make_candidate(
        self,
        option_contract_id: str,
        future_contract_id: str,
        spot: float,
        price: float,
        trade_ts: Optional[datetime] = None,
    ) -> MinuteOptionCandidate:
        trade_ts = trade_ts or datetime(2025, 11, 14, 18, 3, tzinfo=timezone.utc)

        option_contract = OptionContract(option_contract_id, ContractType.Option)
        future_contract = FutureContract(future_contract_id, ContractType.Future)
        if option_contract.get_underlying() != future_contract.get_underlying():
            raise ValueError(
                f"Underlying mismatch: option={option_contract_id}, future={future_contract_id}"
            )

        calendar = usd_calendar()
        expiry_date = option_contract.get_contract_maturity_dates_by_contract_id(
            data_date=trade_ts.date(),
            calendars=[calendar],
            termination_rule=ContractTerminationRule.EndOfMonth,
        )
        expiry_dt_utc = _make_expiry_dt_utc(expiry_date)
        tau = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc)
        strike = float(option_contract.get_strike())
        option_type = option_contract.get_option_type()
        business_days = int(calendar.count_business_days(trade_ts.date(), expiry_date, False, True))

        return MinuteOptionCandidate(
            meta=ContractMeta(
                contract_type="option",
                underlying=future_contract.get_underlying(),
                strike=strike,
                option_type=option_type,
                expiry_date=expiry_date,
                expiry_dt_utc=expiry_dt_utc,
            ),
            price=float(price),
            weight=5.0,
            strike=strike,
            spot=float(spot),
            tau=float(tau),
            business_days=business_days,
        )

    def _make_candidate_from_contract_ids(
        self,
        option_contract_id: str,
        future_contract_id: str,
        spot: float,
        target_vol: float,
        trade_ts: Optional[datetime] = None,
    ) -> MinuteOptionCandidate:
        trade_ts = trade_ts or datetime(2025, 11, 14, 18, 3, tzinfo=timezone.utc)
        option_contract = OptionContract(option_contract_id, ContractType.Option)
        expiry_date = option_contract.get_contract_maturity_dates_by_contract_id(
            data_date=trade_ts.date(),
            calendars=[usd_calendar()],
            termination_rule=ContractTerminationRule.EndOfMonth,
        )
        expiry_dt_utc = _make_expiry_dt_utc(expiry_date)
        tau = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc)
        price = float(
            black_scholes_price(
                strike=float(option_contract.get_strike()),
                option_type=option_contract.get_option_type(),
                spot=spot,
                vol=target_vol,
                tau=tau,
                r=0.0,
                q=0.0,
            )
        )
        return self._make_candidate(
            option_contract_id=option_contract_id,
            future_contract_id=future_contract_id,
            spot=spot,
            price=price,
            trade_ts=trade_ts,
        )

    def _make_candidate_from_market_quote(
        self,
        option_contract_id: str,
        future_contract_id: str,
        spot: float,
        price: float,
        trade_ts: datetime,
    ) -> MinuteOptionCandidate:
        return self._make_candidate(
            option_contract_id=option_contract_id,
            future_contract_id=future_contract_id,
            spot=spot,
            price=price,
            trade_ts=trade_ts,
        )

    def _market_cases(self):
        return [
            {
                "option_contract_id": "TY10025R3",
                "future_contract_id": "TYM3",
                "spot": 115.273438,
                "price": 0.007,
                "trade_ts": datetime(2023, 5, 1, 0, 2, 47, tzinfo=timezone.utc),
                "expected_vol": 0.1287814081,
            },
            {
                "option_contract_id": "TY990V3",
                "future_contract_id": "TYV3",
                "spot": 113.585938,
                "price": 2.5,
                "trade_ts": datetime(2023, 5, 31, 20, 59, 49, tzinfo=timezone.utc),
                "expected_vol": 0.2808476686,
            },
            {
                "option_contract_id": "TY1015S3",
                "future_contract_id": "TYN3",
                "spot": 112.179688,
                "price": 0.007,
                "trade_ts": datetime(2023, 6, 21, 14, 0, 2, tzinfo=timezone.utc),
                "expected_vol": 0.1175748110,
            },
        ]

    def test_gpu_implied_vol_put_round_trips_from_ty_option_and_future_price(self):
        candidate = self._make_candidate_from_contract_ids(
            option_contract_id="TY110O26",
            future_contract_id="TYZ5",
            spot=112.5934,
            target_vol=0.27,
        )

        gpu_vols = _compute_implied_vols_gpu([candidate], device=self.DEVICE)

        self.assertEqual(len(gpu_vols), 1)
        self.assertIsNotNone(gpu_vols[0])
        self.assertAlmostEqual(gpu_vols[0], 0.27, places=3)

    def test_gpu_implied_vol_call_round_trips_from_ty_option_and_future_price(self):
        candidate = self._make_candidate_from_contract_ids(
            option_contract_id="TY115C26",
            future_contract_id="TYZ5",
            spot=112.5934,
            target_vol=0.31,
        )

        gpu_vols = _compute_implied_vols_gpu([candidate], device=self.DEVICE)

        self.assertEqual(len(gpu_vols), 1)
        self.assertIsNotNone(gpu_vols[0])
        self.assertAlmostEqual(gpu_vols[0], 0.31, places=3)

    def test_gpu_implied_vol_batch_supports_multiple_contract_ids(self):
        candidates = [
            self._make_candidate_from_contract_ids(
                option_contract_id="TY110O26",
                future_contract_id="TYZ5",
                spot=112.5934,
                target_vol=0.27,
            ),
            self._make_candidate_from_contract_ids(
                option_contract_id="TY115C26",
                future_contract_id="TYZ5",
                spot=112.5934,
                target_vol=0.31,
            ),
        ]

        gpu_vols = _compute_implied_vols_gpu(candidates, device=self.DEVICE)

        self.assertEqual(len(gpu_vols), 2)
        self.assertIsNotNone(gpu_vols[0])
        self.assertIsNotNone(gpu_vols[1])
        self.assertAlmostEqual(gpu_vols[0], 0.27, places=3)
        self.assertAlmostEqual(gpu_vols[1], 0.31, places=3)

    def test_gpu_implied_vol_returns_none_for_invalid_option_price(self):
        candidate = self._make_candidate_from_contract_ids(
            option_contract_id="TY110O26",
            future_contract_id="TYZ5",
            spot=112.5934,
            target_vol=0.27,
        )
        invalid_candidate = MinuteOptionCandidate(
            meta=candidate.meta,
            price=float("nan"),
            weight=candidate.weight,
            strike=candidate.strike,
            spot=candidate.spot,
            tau=candidate.tau,
            business_days=candidate.business_days,
        )

        gpu_vols = _compute_implied_vols_gpu([invalid_candidate], device=self.DEVICE)

        self.assertEqual(gpu_vols, [None])

    def test_gpu_implied_vol_market_cases_match_expected_vols(self):
        for case in self._market_cases():
            with self.subTest(option_contract_id=case["option_contract_id"]):
                candidate = self._make_candidate_from_market_quote(
                    option_contract_id=case["option_contract_id"],
                    future_contract_id=case["future_contract_id"],
                    spot=case["spot"],
                    price=case["price"],
                    trade_ts=case["trade_ts"],
                )
                gpu_vols = _compute_implied_vols_gpu([candidate], device=self.DEVICE)

                self.assertEqual(len(gpu_vols), 1)
                self.assertIsNotNone(gpu_vols[0])
                self.assertAlmostEqual(gpu_vols[0], case["expected_vol"], delta=1e-5)

    def test_gpu_implied_vol_market_cases_match_cpu_solver(self):
        candidates = [
            self._make_candidate_from_market_quote(
                option_contract_id=case["option_contract_id"],
                future_contract_id=case["future_contract_id"],
                spot=case["spot"],
                price=case["price"],
                trade_ts=case["trade_ts"],
            )
            for case in self._market_cases()
        ]

        cpu_vols = _compute_implied_vols_cpu(candidates)
        gpu_vols = _compute_implied_vols_gpu(candidates, device=self.DEVICE)

        self.assertEqual(len(cpu_vols), len(gpu_vols))
        for cpu_vol, gpu_vol in zip(cpu_vols, gpu_vols):
            self.assertIsNotNone(cpu_vol)
            self.assertIsNotNone(gpu_vol)
            self.assertAlmostEqual(cpu_vol, gpu_vol, delta=1e-6)

    def test_gpu_implied_vol_mixed_batch_preserves_none_for_invalid_fields(self):
        valid_candidate = self._make_candidate_from_market_quote(
            option_contract_id="TY10025R3",
            future_contract_id="TYM3",
            spot=115.273438,
            price=0.007,
            trade_ts=datetime(2023, 5, 1, 0, 2, 47, tzinfo=timezone.utc),
        )
        standalone_valid_vol = _compute_implied_vols_gpu([valid_candidate], device=self.DEVICE)[0]
        mixed_candidates = [
            valid_candidate,
            MinuteOptionCandidate(
                meta=valid_candidate.meta,
                price=float("nan"),
                weight=valid_candidate.weight,
                strike=valid_candidate.strike,
                spot=valid_candidate.spot,
                tau=valid_candidate.tau,
                business_days=valid_candidate.business_days,
            ),
            MinuteOptionCandidate(
                meta=valid_candidate.meta,
                price=valid_candidate.price,
                weight=valid_candidate.weight,
                strike=valid_candidate.strike,
                spot=valid_candidate.spot,
                tau=float("nan"),
                business_days=valid_candidate.business_days,
            ),
            MinuteOptionCandidate(
                meta=valid_candidate.meta,
                price=valid_candidate.price,
                weight=valid_candidate.weight,
                strike=valid_candidate.strike,
                spot=float("nan"),
                tau=valid_candidate.tau,
                business_days=valid_candidate.business_days,
            ),
            MinuteOptionCandidate(
                meta=valid_candidate.meta,
                price=valid_candidate.price,
                weight=valid_candidate.weight,
                strike=float("nan"),
                spot=valid_candidate.spot,
                tau=valid_candidate.tau,
                business_days=valid_candidate.business_days,
            ),
        ]

        gpu_vols = _compute_implied_vols_gpu(mixed_candidates, device=self.DEVICE)

        self.assertEqual(len(gpu_vols), 5)
        self.assertIsNotNone(standalone_valid_vol)
        self.assertIsNotNone(gpu_vols[0])
        self.assertAlmostEqual(gpu_vols[0], standalone_valid_vol, delta=1e-5)
        self.assertEqual(gpu_vols[1:], [None, None, None, None])


if __name__ == "__main__":
    unittest.main()
