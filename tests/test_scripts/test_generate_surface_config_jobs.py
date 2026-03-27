import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.generate_surface.main as surface_main  # noqa: E402
from scripts.generate_surface.common.minute_svi_common import _load_minute_svi_config  # noqa: E402
from scripts.generate_surface.common.minute_svi_excel_common import _parse_args as parse_excel_args  # noqa: E402
from scripts.generate_surface.common.minute_svi_window_common import _parse_args as parse_window_args  # noqa: E402


class TestGenerateSurfaceConfigDrivenJob(unittest.TestCase):
    def test_main_dispatches_configured_job_without_explicit_subcommand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      job: minute-svi-excel
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

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
                surface_main.main([
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--window-minutes",
                    "5",
                ])

            mock_cpu_main.assert_called_once_with([
                "--config",
                str(config_path),
                "--window-minutes",
                "5",
            ])

    def test_main_raises_for_unknown_job_in_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      job: unsupported-job
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                surface_main.main(["--config", str(config_path)])


class TestWindowAndExcelConfigParsing(unittest.TestCase):
    def test_minute_svi_config_expands_output_dir_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      minute_svi:
                        output_dir: data/processed
                        expiration_time_utc: "20:00:00"
                        input_glob: ${option_data_glob}
                        output_json: output_dir/window.json
                        log_file: ${output_dir}/minute_svi.log
                        precalib_csv: ${output_dir}/minute_svi_precalib.csv
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_minute_svi_config(str(config_path))

            self.assertEqual(loaded["input_glob"], "data/raw/shared/**/*.csv.gz")
            self.assertEqual(loaded["output_dir"], "data/processed")
            self.assertEqual(loaded["expiration_time_utc"], "20:00:00")
            self.assertEqual(loaded["output_json"], "data/processed/window.json")
            self.assertEqual(loaded["log_file"], "data/processed/minute_svi.log")
            self.assertEqual(loaded["precalib_csv"], "data/processed/minute_svi_precalib.csv")

    def test_window_args_read_defaults_from_default_yaml_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      minute_svi:
                        output_dir: data/processed
                        expiration_time_utc: "19:45:30"
                        output_json: ${output_dir}/window.json
                        log_file: output_dir/window.log
                      minute_svi_window:
                        target_datetimes:
                          - 2026-03-09T14:35:00Z
                          - 2026-03-09T14:36:00Z
                        target_datetimes_file: data/raw/targets.txt
                        window_minutes: 7
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = parse_window_args(["--config", str(config_path)])

            self.assertEqual(args.input_glob, "data/raw/shared/**/*.csv.gz")
            self.assertEqual(args.output_json, "data/processed/window.json")
            self.assertEqual(args.log_file, "data/processed/window.log")
            self.assertEqual(args.expiration_time_utc, "19:45:30")
            self.assertEqual(
                args.target_datetimes,
                ["2026-03-09T14:35:00Z", "2026-03-09T14:36:00Z"],
            )
            self.assertEqual(args.target_datetimes_file, "data/raw/targets.txt")
            self.assertEqual(args.window_minutes, 7)

    def test_excel_args_read_defaults_from_default_yaml_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      minute_svi:
                        output_dir: data/processed
                        expiration_time_utc: "20:00:00"
                        output_json: output_dir/excel.json
                        precalib_csv: ${output_dir}/excel_precalib.csv
                      minute_svi_excel:
                        target_xlsx: data/raw/custom.xlsx
                        sheet_name: Events
                        date_column: PublishDate
                        time_column: EventTime
                        source_timezone: Europe/London
                        max_target_datetimes: 12
                        window_minutes: 9
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = parse_excel_args(["--config", str(config_path)])

            self.assertEqual(args.output_json, "data/processed/excel.json")
            self.assertEqual(args.precalib_csv, "data/processed/excel_precalib.csv")
            self.assertEqual(args.expiration_time_utc, "20:00:00")
            self.assertEqual(args.target_xlsx, "data/raw/custom.xlsx")
            self.assertEqual(args.sheet_name, "Events")
            self.assertEqual(args.date_column, "PublishDate")
            self.assertEqual(args.time_column, "EventTime")
            self.assertEqual(args.source_timezone, "Europe/London")
            self.assertEqual(args.max_target_datetimes, 12)
            self.assertEqual(args.window_minutes, 9)


if __name__ == "__main__":
    unittest.main()
