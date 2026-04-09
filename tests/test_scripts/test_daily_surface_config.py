import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_surface import daily_surface  # noqa: E402
from scripts.generate_surface.daily_surface import _load_daily_surface_config, _parse_args  # noqa: E402


class TestDailySurfaceConfig(unittest.TestCase):
    def test_load_daily_surface_config_expands_shared_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      raw_root: data/raw/shared
                      daily_surface:
                        input_glob: ${raw_root}/**/*.csv.gz
                        output_dir: outputs/daily
                        save_daily_csv: false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_daily_surface_config(str(config_path))

            self.assertEqual(loaded["input_glob"], "data/raw/shared/**/*.csv.gz")
            self.assertEqual(loaded["output_dir"], "outputs/daily")
            self.assertFalse(loaded["save_daily_csv"])

    def test_load_daily_surface_config_reads_nested_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      daily_surface:
                        output_dir: outputs/daily
                        save_daily_csv: false
                        npz_name: custom_surface_stack.npz
                        strike_bins: 24
                        maturity_bins: 18
                        moneyness_min: 0.6
                        moneyness_max: 1.4
                        maturity_min_days: 3
                        maturity_max_days: 540
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_daily_surface_config(str(config_path))

            self.assertEqual(loaded["input_glob"], "data/raw/shared/**/*.csv.gz")
            self.assertEqual(loaded["output_dir"], "outputs/daily")
            self.assertFalse(loaded["save_daily_csv"])
            self.assertEqual(loaded["npz_name"], "custom_surface_stack.npz")
            self.assertEqual(loaded["strike_bins"], 24)
            self.assertEqual(loaded["maturity_bins"], 18)
            self.assertEqual(loaded["moneyness_min"], 0.6)
            self.assertEqual(loaded["moneyness_max"], 1.4)
            self.assertEqual(loaded["maturity_min_days"], 3)
            self.assertEqual(loaded["maturity_max_days"], 540)

    def test_parse_args_uses_config_defaults_and_allows_cli_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      daily_surface:
                        output_dir: outputs/daily
                        save_daily_csv: false
                        npz_name: custom_surface_stack.npz
                        strike_bins: 24
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = _parse_args([
                "--config",
                str(config_path),
                "--strike-bins",
                "30",
                "--save-daily-csv",
            ])

            self.assertEqual(args.input_glob, "data/raw/shared/**/*.csv.gz")
            self.assertEqual(args.output_dir, "outputs/daily")
            self.assertEqual(args.npz_name, "custom_surface_stack.npz")
            self.assertEqual(args.strike_bins, 30)
            self.assertTrue(args.save_daily_csv)

    def test_run_is_thin_wrapper_around_common_runtime(self):
        args = object()
        expected = Path("outputs/vol_surface/demo.npz")

        with patch("scripts.generate_surface.daily_surface.run_daily_surface_job", return_value=expected) as run_job:
            actual = daily_surface.run(args)

        self.assertEqual(actual, expected)
        run_job.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
