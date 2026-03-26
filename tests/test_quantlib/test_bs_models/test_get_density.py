import sys
import unittest
from datetime import datetime, time
from pathlib import Path

import torch

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlib.calculation.analytics.models.analytical.equity.formula import black_scholes_price, \
    black_scholes_implied_vol_torch, black_scholes_implied_vol, black_scholes_price_torch
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calendar.daycount import bus_250_gbp
from quantlib.risk_engine.implied_distribution.get_density import get_density_torch, get_density


def get_tau(quote_time: datetime, maturity_time: datetime, daycount) -> float:
    base_tau = daycount(quote_time.date(), maturity_time.date())
    seconds_per_day = 86400.0

    quote_fraction = (
        quote_time - datetime.combine(quote_time.date(), time(0, 0, 0))
    ).total_seconds() / seconds_per_day
    maturity_fraction = (
        maturity_time - datetime.combine(maturity_time.date(), time(0, 0, 0))
    ).total_seconds() / seconds_per_day

    days_in_year = float(getattr(daycount, "days_in_year", 365.0))
    return float(base_tau + (maturity_fraction - quote_fraction) / days_in_year)


class TestGetDensity(unittest.TestCase):
    def test_get_density_torch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        strike = 100
        option_type = OptionType.CALL
        spot = 100
        tau = 1
        r = 0.05
        q = 0.03
        vol = 0.5
        price = black_scholes_price(strike, option_type, spot, vol, tau, r, q)
        kp = black_scholes_price(strike + 0.0001, option_type, spot, vol, tau, r, q)
        kd = black_scholes_price(strike - 0.0001, option_type, spot, vol, tau, r, q)
        cpu_result = (kp - 2 * price + kd) / 0.0001 ** 2

        price = torch.tensor([price], dtype=torch.float64, device=device)
        strike = torch.tensor(strike, dtype=torch.float64, device=device)
        spot = torch.tensor([spot], dtype=torch.float64, device=device)
        tau = torch.tensor([tau], dtype=torch.float64, device=device)
        r = torch.tensor([r], dtype=torch.float64, device=device)
        q = torch.tensor([q], dtype=torch.float64, device=device)

        # option_type = torch.tensor([option_type.value], dtype=torch.bool)

        gpu_res = get_density_torch(price, strike, [option_type], spot, tau, r, q, device).cpu().numpy()
        self.assertAlmostEqual(cpu_result, gpu_res, delta=1e-5)

    def test_density_vol_time(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        strike = 100
        strike_torch = torch.tensor([strike], dtype=torch.float64, device=device)
        spot = 100
        spot_torch = torch.tensor([spot], dtype=torch.float64, device=device)
        quote_time = datetime(2023, 11, 14, 9, 19, 45)
        maturity = datetime(2034, 1, 31, 20, 0, 0)
        price = 0.9
        price_torch = torch.tensor([price], dtype=torch.float64, device=device)
        option_type = OptionType.CALL
        option_type_torch = torch.tensor([option_type.value], dtype=torch.bool, device=device)
        daycount = bus_250_gbp

        expiration_date = maturity

        tau = get_tau(quote_time, maturity, daycount)
        tau_torch = torch.tensor([tau], dtype=torch.float64, device=device)

        vol = black_scholes_implied_vol(price, strike, option_type, spot, tau, 0.05, 0.05)
        vol_torch = black_scholes_implied_vol_torch(
            price_torch, strike_torch, option_type_torch, spot_torch, tau_torch,
            torch.tensor([0.05], dtype=torch.float64, device=device),
            torch.tensor([0.05], dtype=torch.float64, device=device),
            device
        )

        self.assertAlmostEqual(vol, vol_torch, delta=1e-4)

        price = black_scholes_price(strike, option_type, spot, vol, tau, 0.05, 0.05)
        price_torch = black_scholes_price_torch(
            strike_torch,
            option_type_torch,
            spot_torch,
            torch.tensor([vol], dtype=torch.float64, device=device),
            tau_torch,
            torch.tensor([0.05], dtype=torch.float64, device=device),
            torch.tensor([0.05], dtype=torch.float64, device=device),
            device
        ).cpu()

        self.assertAlmostEqual(price, price_torch, delta=1e-4)

        density = get_density(price, strike, option_type, spot, tau, 0.05, 0.05)

        density_torch = get_density_torch(
            torch.tensor([price], dtype=torch.float64, device=device),
            strike_torch,
            option_type_torch,
            spot_torch,
            tau_torch,
            torch.tensor([0.05], device=device),
            torch.tensor([0.05], device=device),
            device
        ).cpu()
        self.assertAlmostEqual(density, density_torch, delta=1e-4)
