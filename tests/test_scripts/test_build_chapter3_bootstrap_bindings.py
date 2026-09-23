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


if __name__ == "__main__":
    unittest.main()
