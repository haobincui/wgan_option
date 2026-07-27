import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scripts.generate_surface.main as surface_main  # noqa: E402
from wgan_option.surface_generation.common.config_utils import load_yaml_mapping  # noqa: E402
from wgan_option.surface_generation.data_helperd.all import (  # noqa: E402
    _load_generate_surface_config,
    _parse_args as parse_generate_surface_args,
)
from wgan_option.surface_generation.data_helperd.excel import _parse_args as parse_excel_args  # noqa: E402
from wgan_option.surface_generation.data_helperd.window import _parse_args as parse_window_args  # noqa: E402


class TestGenerateSurfaceConfigDrivenJob(unittest.TestCase):
    def test_main_dispatches_configured_job_without_explicit_subcommand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      job: generate_surface
                      generate_surface:
                        model: svi
                        data_range: excel
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

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

    def test_main_requires_surface_builder_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text("job: generate_surface\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "top-level `surface_builder` mapping"):
                surface_main.main(["--config", str(config_path)])


class TestGenerateSurfaceConfigParsing(unittest.TestCase):
    def test_checked_in_surface_builder_configs_match_names(self):
        cases = [
            ("configs/surface_builder/svi/generate_surface-svi-all.yaml", "svi", "all"),
            ("configs/surface_builder/svi/generate_surface-svi-window.yaml", "svi", "window"),
            ("configs/surface_builder/svi/generate_surface-svi-excel.yaml", "svi", "excel"),
            ("configs/surface_builder/sabr/generate_surface-sabr-all.yaml", "sabr", "all"),
            ("configs/surface_builder/sabr/generate_surface-sabr-window.yaml", "sabr", "window"),
            ("configs/surface_builder/sabr/generate_surface-sabr-excel.yaml", "sabr", "excel"),
            ("configs/surface_builder/cubic/generate_surface-cubic-all.yaml", "cubic", "all"),
            ("configs/surface_builder/cubic/generate_surface-cubic-window.yaml", "cubic", "window"),
            ("configs/surface_builder/cubic/generate_surface-cubic-excel.yaml", "cubic", "excel"),
            ("configs/surface_builder/raw/generate_surface-raw-all.yaml", "raw", "all"),
            ("configs/surface_builder/raw/generate_surface-raw-window.yaml", "raw", "window"),
            ("configs/surface_builder/raw/generate_surface-raw-excel.yaml", "raw", "excel"),
            (
                "configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml",
                "raw",
                "excel",
            ),
            (
                "configs/surface_builder/raw/generate_surface-raw-excel-pipeline.yaml",
                "raw",
                "excel",
            ),
        ]

        for config_path, expected_model, expected_data_range in cases:
            with self.subTest(config_path=config_path):
                _, payload = load_yaml_mapping(config_path)
                self.assertEqual(payload["surface_builder"]["job"], "generate_surface")

                loaded = _load_generate_surface_config(config_path)
                self.assertEqual(loaded["model"], expected_model)
                self.assertEqual(loaded["data_range"], expected_data_range)

    def test_generate_surface_config_requires_nested_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "surface_builder.generate_surface"):
                _load_generate_surface_config(str(config_path))

    def test_generate_surface_config_expands_output_dir_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      generate_surface:
                        model: svi
                        data_range: window
                        output_dir: data/processed
                        expiration_time_utc: "20:00:00"
                        max_precalib_iv: 2.5
                        input_glob: ${option_data_glob}
                        output_json: output_dir/window.json
                        log_file: ${output_dir}/surface.log
                        precalib_csv: ${output_dir}/surface-precalib.csv
                        resolved_config_path: ${output_dir}/surface-resolved_config.yaml
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_generate_surface_config(str(config_path))

            self.assertEqual(loaded["input_glob"], "data/raw/shared/**/*.csv.gz")
            self.assertEqual(loaded["output_dir"], "data/processed")
            self.assertEqual(loaded["expiration_time_utc"], "20:00:00")
            self.assertEqual(loaded["max_precalib_iv"], 2.5)
            self.assertEqual(loaded["output_json"], "data/processed/window.json")
            self.assertEqual(loaded["log_file"], "data/processed/surface.log")
            self.assertEqual(loaded["precalib_csv"], "data/processed/surface-precalib.csv")
            self.assertEqual(loaded["resolved_config_path"], "data/processed/surface-resolved_config.yaml")

    def test_generate_surface_config_expands_model_data_range_and_run_ts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      generate_surface:
                        model: svi
                        data_range: all
                        run_ts: baseline-run
                        output_dir: data/processed/${model}-${data_range}/${run_ts}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_generate_surface_config(
                str(config_path),
                runtime_overrides={
                    "model": "cubic",
                    "data_range": "excel",
                    "run_ts": "experiment-01",
                },
            )

            self.assertEqual(loaded["model"], "cubic")
            self.assertEqual(loaded["data_range"], "excel")
            self.assertEqual(loaded["run_ts"], "experiment-01")
            self.assertEqual(loaded["output_dir"], "data/processed/cubic-excel/experiment-01")
            self.assertEqual(
                loaded["output_json"],
                "data/processed/cubic-excel/experiment-01/surface-cubic-excel.json",
            )

    def test_generate_surface_args_reject_non_boolean_yaml_values(self):
        for raw_value in ['"false"', "0"]:
            with self.subTest(raw_value=raw_value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = Path(tmpdir) / "surface.yaml"
                    config_path.write_text(
                        textwrap.dedent(
                            f"""
                            surface_builder:
                              generate_surface:
                                save_precalib_csv: {raw_value}
                            """
                        ).strip()
                        + "\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, "YAML boolean"):
                        parse_generate_surface_args(["--config", str(config_path)])

    def test_window_args_read_defaults_from_unified_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      option_data_glob: data/raw/shared/**/*.csv.gz
                      generate_surface:
                        model: svi
                        data_range: window
                        output_dir: data/processed
                        expiration_time_utc: "19:45:30"
                        max_precalib_iv: 2.75
                        output_json: ${output_dir}/surface-window.json
                        log_file: ${output_dir}/surface-window.log
                        target_datetimes:
                          - 2026-03-09T14:35:00Z
                          - 2026-03-09T14:36:00Z
                        target_datetimes_file: data/raw/targets.txt
                        window_minutes: 7
                        calibration_workers: 8
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = parse_window_args(["--config", str(config_path)])

            self.assertEqual(args.input_glob, "data/raw/shared/**/*.csv.gz")
            self.assertEqual(args.output_json, "data/processed/surface-window.json")
            self.assertEqual(args.log_file, "data/processed/surface-window.log")
            self.assertEqual(args.expiration_time_utc, "19:45:30")
            self.assertEqual(args.max_precalib_iv, 2.75)
            self.assertEqual(args.data_range, "window")
            self.assertEqual(
                args.target_datetimes,
                ["2026-03-09T14:35:00Z", "2026-03-09T14:36:00Z"],
            )
            self.assertEqual(args.target_datetimes_file, "data/raw/targets.txt")
            self.assertEqual(args.window_minutes, 7)
            self.assertEqual(args.calibration_workers, 8)

    def test_generate_surface_args_track_model_data_range_run_ts_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      generate_surface:
                        model: svi
                        data_range: all
                        output_dir: data/processed/${model}-${data_range}/${run_ts}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = parse_generate_surface_args(
                [
                    "--config",
                    str(config_path),
                    "--model",
                    "sabr",
                    "--data_range",
                    "window",
                    "--run-ts",
                    "20260409-010203",
                ]
            )

            self.assertEqual(args.model, "sabr")
            self.assertEqual(args.data_range, "window")
            self.assertEqual(args.run_ts, "20260409-010203")
            self.assertEqual(args.output_dir, "data/processed/sabr-window/20260409-010203")
            self.assertEqual(
                args.output_json,
                "data/processed/sabr-window/20260409-010203/surface-sabr-window.json",
            )

    def test_excel_args_read_defaults_from_unified_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "surface.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    surface_builder:
                      generate_surface:
                        model: raw
                        data_range: excel
                        output_dir: data/processed
                        expiration_time_utc: "20:00:00"
                        max_precalib_iv: 1.8
                        output_json: ${output_dir}/surface-excel.json
                        precalib_csv: ${output_dir}/surface-excel-precalib.csv
                        target_xlsx: data/raw/custom.xlsx
                        sheet_name: Events
                        date_column: PublishDate
                        time_column: EventTime
                        source_timezone: Europe/London
                        publication_availability_lag_minutes: 2
                        max_target_datetimes: 12
                        window_minutes: 9
                        min_expiries_per_minute: 2
                        calibration_workers: 6
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = parse_excel_args(["--config", str(config_path)])

            self.assertEqual(args.output_json, "data/processed/surface-excel.json")
            self.assertEqual(args.precalib_csv, "data/processed/surface-excel-precalib.csv")
            self.assertEqual(args.expiration_time_utc, "20:00:00")
            self.assertEqual(args.max_precalib_iv, 1.8)
            self.assertEqual(args.data_range, "excel")
            self.assertEqual(args.target_xlsx, "data/raw/custom.xlsx")
            self.assertEqual(args.sheet_name, "Events")
            self.assertEqual(args.date_column, "PublishDate")
            self.assertEqual(args.time_column, "EventTime")
            self.assertEqual(args.source_timezone, "Europe/London")
            self.assertEqual(args.publication_availability_lag_minutes, 2)
            self.assertEqual(args.max_target_datetimes, 12)
            self.assertEqual(args.window_minutes, 9)
            self.assertEqual(args.calibration_workers, 6)


if __name__ == "__main__":
    unittest.main()
