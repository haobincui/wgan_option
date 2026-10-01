"""Coverage and fail-closed tests for the Chapter 3 binding generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.rq123.build_chapter3_bootstrap_bindings import (
    DEFAULT_OUTPUT,
    DEFAULT_TEX,
    DEFAULT_VALUES,
    F4_CAPACITY_ANALYSIS_KIND,
    F4_CAPACITY_PAIR_METRICS,
    F4_CAPACITY_SUMMARY_CSV,
    F4_CAPACITY_SUMMARY_JSON,
    F4_CAPACITY_TABLE_LABEL,
    F4_CAPACITY_TABLE_TEX,
    LEGACY_CAPACITY_TABLE_LABEL,
    PASSTHROUGH_LABELS,
    REQUIRED_JOBS,
    REQUIRED_RESULT_TABLES,
    _serialized,
    build_payload,
)
from scripts.rq123.verify_chapter3_bootstrap_bindings import verify_bindings


class Chapter3BindingBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = build_payload()

    def test_checked_in_bindings_are_generated_and_verify(self):
        self.assertEqual(
            DEFAULT_OUTPUT.read_text(encoding="utf-8"),
            _serialized(self.payload),
        )
        qa = verify_bindings(DEFAULT_OUTPUT)
        self.assertTrue(qa["passed"])
        self.assertEqual(qa["bound_jobs"], sorted(REQUIRED_JOBS))
        self.assertEqual(qa["unchanged_passthrough_tables"], 6)

    def test_complete_scope_covers_all_formal_jobs_and_target_tables(self):
        self.assertEqual(self.payload["status"], "complete")
        self.assertEqual(
            self.payload["scope"]["excluded_anchor_range"]["start"],
            r"\label{subsec:ch3:rq4_conditional_text_value}",
        )
        self.assertEqual(
            set(self.payload["coverage"]["bound_jobs"]), set(REQUIRED_JOBS)
        )
        bound_labels = {
            item["anchor"]["value"]
            for item in self.payload["bindings"]
            if item["anchor"]["kind"] == "latex_label"
        }
        self.assertTrue(set(REQUIRED_RESULT_TABLES).issubset(bound_labels))
        self.assertGreaterEqual(self.payload["coverage"]["numerical_bindings"], 450)
        self.assertEqual(
            tuple(item["label"] for item in self.payload["passthrough_table_hashes"]),
            PASSTHROUGH_LABELS,
        )
        self.assertEqual(len(self.payload["external_table_changes"]), 2)

    def test_every_atom_has_an_explicit_resolvable_lineage(self):
        values = json.loads(DEFAULT_VALUES.read_text(encoding="utf-8"))
        for group in self.payload["bindings"]:
            self.assertTrue(group["values"], group["binding_id"])
            for atom in group["values"]:
                self.assertNotEqual("source" in atom, "source_expression" in atom)
                sources = (
                    [atom["source"]]
                    if "source" in atom
                    else atom["source_expression"]["operands"]
                )
                for source in sources:
                    record = values["jobs"][source["job_id"]][source["collection"]][
                        source["item_id"]
                    ]
                    self.assertIn(source["field"], record)

    def test_a_changed_table_literal_fails_generation(self):
        tex = DEFAULT_TEX.read_text(encoding="utf-8")
        old = r"\rqonebootcell{0.001602081}"
        self.assertEqual(tex.count(old), 1)
        changed = tex.replace(old, r"\rqonebootcell{0.001602082}", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chapter3.tex"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Formal literal.*absent"):
                build_payload(path, DEFAULT_VALUES, DEFAULT_OUTPUT)

    def test_external_current_hashes_match_exact_tables(self):
        tex = DEFAULT_TEX.read_text(encoding="utf-8")
        tables = re.findall(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL
        )
        for item in self.payload["external_table_changes"]:
            matching = [
                table
                for table in tables
                if r"\label{" + item["label"] + "}" in table
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(
                hashlib.sha256(matching[0].encode("utf-8")).hexdigest(),
                item["current_sha256"],
            )

    def _legacy_fixture(self, directory: Path) -> tuple[Path, Path]:
        tex = DEFAULT_TEX.read_text(encoding="utf-8")
        tables = re.findall(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL
        )
        active = [
            table
            for table in tables
            if r"\label{" + F4_CAPACITY_TABLE_LABEL + "}" in table
        ]
        self.assertEqual(len(active), 1)
        legacy_table = "\n".join(
            (
                r"\begin{table}[htbp]",
                r"\centering",
                r"\begin{tabular}{lr}",
                "Capacity & MAE \\\\",
                "Legacy & 0.0010000000 \\\\",
                r"\end{tabular}",
                r"\caption{Superseded capacity diagnostics.}",
                rf"\label{{{LEGACY_CAPACITY_TABLE_LABEL}}}",
                r"\end{table}",
            )
        )
        tex_path = directory / "chapter3_legacy.tex"
        tex_path.write_text(
            tex.replace(active[0], legacy_table, 1), encoding="utf-8"
        )

        existing = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        external = existing["external_table_changes"]
        capacity_index = next(
            index
            for index, record in enumerate(external)
            if record["label"]
            in {LEGACY_CAPACITY_TABLE_LABEL, F4_CAPACITY_TABLE_LABEL}
        )
        capacity = external[capacity_index]
        if capacity["label"] == F4_CAPACITY_TABLE_LABEL:
            supersedes = capacity["supersedes"]
            baseline_sha256 = supersedes["baseline_sha256"]
        else:
            baseline_sha256 = capacity["baseline_sha256"]
        external[capacity_index] = {
            "label": LEGACY_CAPACITY_TABLE_LABEL,
            "baseline_sha256": baseline_sha256,
            "current_sha256": hashlib.sha256(
                legacy_table.encode("utf-8")
            ).hexdigest(),
            "status": "user_authorized_descriptive_update_verified",
            "numerical_source_audit": (
                "Synthetic legacy fixture for migration testing; not a reported "
                "result."
            ),
        }
        existing_path = directory / "legacy_bindings.json"
        existing_path.write_text(_serialized(existing), encoding="utf-8")
        return tex_path, existing_path

    def _f4_migration_fixture(self, directory: Path) -> tuple[Path, Path, dict]:
        legacy_tex_path, legacy_existing_path = self._legacy_fixture(directory)
        tex = legacy_tex_path.read_text(encoding="utf-8")
        tables = re.findall(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL
        )
        legacy = [
            table
            for table in tables
            if r"\label{" + LEGACY_CAPACITY_TABLE_LABEL + "}" in table
        ]
        self.assertEqual(len(legacy), 1)
        generated_table = "\n".join(
            (
                r"\begin{table}[htbp]",
                r"\centering",
                r"\begin{tabular}{lr}",
                "Capacity & MAE \\\\",
                "c32 & 0.0010000000 \\\\",
                r"\end{tabular}",
                r"\caption{F4 capacity robustness test.}",
                rf"\label{{{F4_CAPACITY_TABLE_LABEL}}}",
                r"\end{table}",
            )
        )
        tex_path = directory / "chapter3.tex"
        tex_path.write_text(
            tex.replace(legacy[0], generated_table, 1), encoding="utf-8"
        )

        analysis_dir = directory / "analysis"
        analysis_dir.mkdir()
        input_path = directory / "evaluation_pair_metrics.csv.gz"
        input_path.write_bytes(b"evaluation-pair-source")
        pair_path = analysis_dir / F4_CAPACITY_PAIR_METRICS
        pair_path.write_bytes(b"frozen-analysis-pairs")
        summary_csv_path = analysis_dir / F4_CAPACITY_SUMMARY_CSV
        summary_csv_path.write_text("capacity_id\nc08\n", encoding="utf-8")
        latex_path = analysis_dir / F4_CAPACITY_TABLE_TEX
        latex_path.write_text(generated_table + "\n", encoding="utf-8")

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        summary = {
            "schema_version": 1,
            "kind": F4_CAPACITY_ANALYSIS_KIND,
            "fold": "f4_2023q4",
            "architectures": ["film_cnn", "pure_cnn"],
            "capacity_ids": ["c08", "c12", "c16", "c24", "c32", "c48"],
            "current_capacity_id": "c32",
            "seeds": [42, 202, 404],
            "pair_count": 143,
            "session_count": 45,
            "seed_pair_rows_per_architecture_capacity": 429,
            "pair_metric_rows": 5148,
            "aggregation": "equal_seed_mean_of_within_seed_pair_mae_v1",
            "improvement_formula": (
                "100*(1-mae_capacity/mae_c32)_within_architecture"
            ),
            "table_label": F4_CAPACITY_TABLE_LABEL,
            "input_pair_metrics_path": str(input_path),
            "input_pair_metrics_sha256": digest(input_path),
            "artifacts": {
                F4_CAPACITY_PAIR_METRICS: digest(pair_path),
                F4_CAPACITY_SUMMARY_CSV: digest(summary_csv_path),
                F4_CAPACITY_TABLE_TEX: digest(latex_path),
            },
            "rows": [
                {"capacity_id": capacity_id}
                for capacity_id in ("c08", "c12", "c16", "c24", "c32", "c48")
            ],
        }
        (analysis_dir / F4_CAPACITY_SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        payload = build_payload(
            tex_path,
            DEFAULT_VALUES,
            legacy_existing_path,
            f4_capacity_analysis_dir=analysis_dir,
        )
        return tex_path, analysis_dir, payload

    def test_f4_external_record_nests_legacy_and_verifies_sources(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            _, _, payload = self._f4_migration_fixture(directory)
            records = payload["external_table_changes"]
            self.assertEqual(len(records), 2)
            capacity = records[0]
            self.assertEqual(capacity["label"], F4_CAPACITY_TABLE_LABEL)
            self.assertEqual(
                capacity["supersedes"]["label"], LEGACY_CAPACITY_TABLE_LABEL
            )
            self.assertEqual(capacity["supersedes"]["status"], "superseded")
            self.assertNotIn(
                LEGACY_CAPACITY_TABLE_LABEL, [row["label"] for row in records]
            )
            binding_path = directory / "bindings.json"
            binding_path.write_text(_serialized(payload), encoding="utf-8")
            qa = verify_bindings(binding_path)
            self.assertEqual(qa["verified_external_tables"], 2)
            self.assertEqual(qa["superseded_external_tables"], 1)

    def test_f4_builder_waits_for_artifacts_until_new_label_is_active(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            missing = Path(directory_name) / "not-created"
            with self.assertRaisesRegex(ValueError, "Missing F4 capacity summary JSON"):
                build_payload(
                    DEFAULT_TEX,
                    DEFAULT_VALUES,
                    DEFAULT_OUTPUT,
                    f4_capacity_analysis_dir=missing,
                )

            legacy_tex_path, legacy_existing_path = self._legacy_fixture(directory)
            legacy = build_payload(
                legacy_tex_path,
                DEFAULT_VALUES,
                legacy_existing_path,
                f4_capacity_analysis_dir=missing,
            )
            self.assertEqual(
                legacy["external_table_changes"][0]["label"],
                LEGACY_CAPACITY_TABLE_LABEL,
            )

    def test_f4_external_verification_fails_closed_on_table_and_source_drift(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            tex_path, analysis_dir, payload = self._f4_migration_fixture(directory)
            binding_path = directory / "bindings.json"
            binding_path.write_text(_serialized(payload), encoding="utf-8")

            original_tex = tex_path.read_text(encoding="utf-8")
            tex_path.write_text(
                original_tex.replace("0.0010000000", "0.0010000001", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Externally tracked table changed"):
                verify_bindings(binding_path)
            tex_path.write_text(original_tex, encoding="utf-8")

            latex_path = analysis_dir / F4_CAPACITY_TABLE_TEX
            original_latex = latex_path.read_text(encoding="utf-8")
            latex_path.write_text(original_latex + "% drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash drift"):
                verify_bindings(binding_path)
            latex_path.write_text(original_latex, encoding="utf-8")

            tex_path.write_text(
                original_tex + rf"\n\label{{{LEGACY_CAPACITY_TABLE_LABEL}}}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy capacity label"):
                verify_bindings(binding_path)


if __name__ == "__main__":
    unittest.main()
