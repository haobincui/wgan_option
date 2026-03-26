import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.generate_surface.main as surface_main  # noqa: E402
from quantlib.calculation.analytics.position.instruments.features import OptionType  # noqa: E402
from scripts.generate_surface.surface_cpu.generate_minute_svi_params import (  # noqa: E402
    _compute_implied_vols_cpu,
)
from scripts.generate_surface.surface_gpu.generate_minute_svi_params import (  # noqa: E402
    ContractMeta,
    MinuteOptionCandidate,
    _compute_implied_vols_gpu,
)


class TestGenerateSurfaceMain(unittest.TestCase):
    def test_main_dispatches_cpu_minute_job(self):
        mock_cpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "minute-svi": {
                    "cpu": mock_cpu_main,
                    "gpu": surface_main.MINUTE_COMMANDS["minute-svi"]["gpu"],
                }
            },
        ):
            surface_main.main(["minute-svi", "--device", "cpu", "--max-files", "1"])
        mock_cpu_main.assert_called_once_with(["--max-files", "1"])

    @patch("scripts.generate_surface.dispatch.torch.cuda.is_available", return_value=True)
    def test_main_dispatches_gpu_window_job(self, _mock_cuda):
        mock_gpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "minute-svi-window": {
                    "cpu": surface_main.MINUTE_COMMANDS["minute-svi-window"]["cpu"],
                    "gpu": mock_gpu_main,
                }
            },
        ):
            surface_main.main(["minute-svi-window", "--device", "gpu", "--max-files", "1"])
        mock_gpu_main.assert_called_once_with(["--max-files", "1"])

    def test_main_dispatches_cpu_excel_job(self):
        mock_cpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "minute-svi-excel": {
                    "cpu": mock_cpu_main,
                    "gpu": surface_main.MINUTE_COMMANDS["minute-svi-excel"]["gpu"],
                }
            },
        ):
            surface_main.main(["minute-svi-excel", "--device", "cpu", "--window-minutes", "5"])
        mock_cpu_main.assert_called_once_with(["--window-minutes", "5"])

    @patch("scripts.generate_surface.main.daily_surface_main")
    def test_main_dispatches_daily_surface_job(self, mock_daily_main):
        surface_main.main(["daily-surface", "--output-dir", "outputs/demo"])
        mock_daily_main.assert_called_once_with(["--output-dir", "outputs/demo"])

    @patch("scripts.generate_surface.dispatch.torch.cuda.is_available", return_value=False)
    def test_main_raises_when_gpu_requested_without_cuda(self, _mock_cuda):
        mock_gpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "minute-svi": {
                    "cpu": surface_main.MINUTE_COMMANDS["minute-svi"]["cpu"],
                    "gpu": mock_gpu_main,
                }
            },
        ):
            with self.assertRaises(RuntimeError):
                surface_main.main(["minute-svi", "--device", "gpu", "--max-files", "1"])
        mock_gpu_main.assert_not_called()


class TestGpuImpliedVolHelper(unittest.TestCase):
    def test_gpu_batch_helper_matches_cpu_solver(self):
        candidates = [
            MinuteOptionCandidate(
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=110.0,
                    option_type=OptionType.PUT,
                    expiry_date=date(2026, 3, 31),
                ),
                price=0.328125,
                weight=5.0,
                strike=110.0,
                spot=112.5934,
                tau=0.376,
                business_days=94,
            ),
            MinuteOptionCandidate(
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=113.0,
                    option_type=OptionType.CALL,
                    expiry_date=date(2026, 3, 31),
                ),
                price=0.171875,
                weight=4.0,
                strike=113.0,
                spot=112.5934,
                tau=0.376,
                business_days=94,
            ),
        ]

        cpu_vols = _compute_implied_vols_cpu(candidates)
        gpu_vols = _compute_implied_vols_gpu(candidates, device=torch.device("cpu"))

        self.assertEqual(len(cpu_vols), len(gpu_vols))
        for cpu_vol, gpu_vol in zip(cpu_vols, gpu_vols):
            self.assertIsNotNone(cpu_vol)
            self.assertIsNotNone(gpu_vol)
            self.assertAlmostEqual(cpu_vol, gpu_vol, places=3)


if __name__ == "__main__":
    unittest.main()
