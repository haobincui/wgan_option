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
from scripts.generate_surface.backend.surface_cpu.all import (  # noqa: E402
    _compute_implied_vols_cpu,
)
from scripts.generate_surface.backend.surface_gpu.all import (  # noqa: E402
    ContractMeta,
    MinuteOptionCandidate,
    _compute_implied_vols_gpu,
)


class TestGenerateSurfaceMain(unittest.TestCase):
    def test_main_dispatches_cpu_all_job(self):
        mock_cpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "all": {
                    "cpu": mock_cpu_main,
                    "gpu": surface_main.MINUTE_COMMANDS["all"]["gpu"],
                }
            },
        ):
            surface_main.main(
                [
                    "generate_surface",
                    "--device",
                    "cpu",
                    "--model",
                    "sabr",
                    "--data_range",
                    "all",
                    "--max-files",
                    "1",
                ]
            )
        mock_cpu_main.assert_called_once_with(
            ["--model", "sabr", "--data_range", "all", "--max-files", "1"]
        )

    def test_main_dispatches_cpu_all_job_with_raw_model(self):
        mock_cpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "all": {
                    "cpu": mock_cpu_main,
                    "gpu": surface_main.MINUTE_COMMANDS["all"]["gpu"],
                }
            },
        ):
            surface_main.main(
                [
                    "generate_surface",
                    "--device",
                    "cpu",
                    "--model",
                    "raw",
                    "--data_range",
                    "all",
                    "--max-files",
                    "1",
                ]
            )
        mock_cpu_main.assert_called_once_with(
            ["--model", "raw", "--data_range", "all", "--max-files", "1"]
        )

    @patch("scripts.generate_surface.dispatch.torch.cuda.is_available", return_value=True)
    def test_main_dispatches_gpu_window_job(self, _mock_cuda):
        mock_gpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "window": {
                    "cpu": surface_main.MINUTE_COMMANDS["window"]["cpu"],
                    "gpu": mock_gpu_main,
                }
            },
        ):
            surface_main.main(
                [
                    "generate_surface",
                    "--device",
                    "gpu",
                    "--data_range",
                    "window",
                    "--max-files",
                    "1",
                ]
            )
        mock_gpu_main.assert_called_once_with(["--data_range", "window", "--max-files", "1"])

    def test_main_dispatches_cpu_excel_job(self):
        mock_cpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "excel": {
                    "cpu": mock_cpu_main,
                    "gpu": surface_main.MINUTE_COMMANDS["excel"]["gpu"],
                }
            },
        ):
            surface_main.main(
                [
                    "generate_surface",
                    "--device",
                    "cpu",
                    "--data_range",
                    "excel",
                    "--window-minutes",
                    "5",
                ]
            )
        mock_cpu_main.assert_called_once_with(
            ["--data_range", "excel", "--window-minutes", "5"]
        )

    @patch("scripts.generate_surface.dispatch.torch.cuda.is_available", return_value=False)
    def test_main_raises_when_gpu_requested_without_cuda(self, _mock_cuda):
        mock_gpu_main = Mock()
        with patch.dict(
            surface_main.MINUTE_COMMANDS,
            {
                "all": {
                    "cpu": surface_main.MINUTE_COMMANDS["all"]["cpu"],
                    "gpu": mock_gpu_main,
                }
            },
        ):
            with self.assertRaises(RuntimeError):
                surface_main.main(
                    ["generate_surface", "--device", "gpu", "--data_range", "all", "--max-files", "1"]
                )
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
