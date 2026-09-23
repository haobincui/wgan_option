"""Protect explicit source mapping and the final manuscript's binding coverage."""

from __future__ import annotations

import json
import unittest

from scripts.rq123.build_chapter3_bootstrap_bindings import (
    DEFAULT_OUTPUT,
    REQUIRED_JOBS,
    REQUIRED_RESULT_TABLES,
    BindingBuilder,
    _expression_value,
    build_payload,
    contrast,
    fixed,
)
from scripts.rq123.verify_chapter3_bootstrap_bindings import verify_bindings


class Chapter3BindingBuilderTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "jobs": {
                "job": {
                    "contrasts": {
                        "focal": {"reported_p": 0.04},
                        "unrelated": {"reported_p": 0.03},
                    }
                }
            }
        }
        self.tex = (
            "\\begin{table}\n"
            "Model & 0.0400 \\\\\n"
            "Other & 0.0300 \\\\\n"
            "\\label{results}\n"
            "\\end{table}\n"
            "Our focal result has p = 0.0400. End of finding.\n"
        )
        self.source = contrast("job", "focal", "reported_p")

    def test_row_keeps_the_explicit_source_identity(self):
        builder = BindingBuilder(self.tex, self.values)
        builder.row("focal", "results", "Model", [(self.source, fixed(4))])
        binding = builder.bindings[0]
        self.assertEqual(binding["values"][0]["source"], self.source)
        self.assertEqual(binding["values"][0]["literal"], "0.0400")
        self.assertTrue(binding["expected_substring"].startswith("Model &"))

    def test_unrelated_matching_number_cannot_replace_the_named_source(self):
        tex = self.tex.replace("Model & 0.0400", "Model & 0.0300")
        builder = BindingBuilder(tex, self.values)
        with self.assertRaisesRegex(ValueError, "Formal literal.*absent"):
            builder.row("focal", "results", "Model", [(self.source, fixed(4))])

    def test_ambiguous_rows_and_duplicate_ids_fail_closed(self):
        builder = BindingBuilder(self.tex.replace("Other &", "Model &"), self.values)
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            builder.row("focal", "results", "Model", [(self.source, fixed(4))])
        builder = BindingBuilder(self.tex, self.values)
        builder.row("focal", "results", "Model", [(self.source, fixed(4))])
        with self.assertRaisesRegex(ValueError, "Duplicate binding id"):
            builder.row("focal", "results", "Other", [(self.source, fixed(4))])

    def test_prose_requires_unique_explicit_boundaries(self):
        anchor = {"kind": "unique_text", "value": "Our focal result", "window_lines": 0}
        builder = BindingBuilder(self.tex, self.values)
        builder.prose(
            "finding", anchor, "Our focal result", "End of finding.",
            [(self.source, fixed(4))],
        )
        self.assertEqual(len(builder.bindings), 1)
        ambiguous = BindingBuilder(self.tex + "Our focal result again.", self.values)
        with self.assertRaisesRegex(ValueError, "one prose start marker"):
            ambiguous.prose(
                "finding", anchor, "Our focal result", "End of finding.",
                [(self.source, fixed(4))],
            )

    def test_derived_displays_use_named_unrounded_values(self):
        other = contrast("job", "unrelated", "reported_p")
        expression = {"operator": "difference", "operands": [self.source, other]}
        self.assertAlmostEqual(_expression_value(self.values, expression), 0.01)
        expression = {"operator": "negate", "operands": [self.source]}
        self.assertEqual(_expression_value(self.values, expression), -0.04)
        expression["operands"].append(other)
        with self.assertRaisesRegex(ValueError, "Unsupported source expression"):
            _expression_value(self.values, expression)

    def test_final_manuscript_has_complete_current_source_bindings(self):
        payload = build_payload()
        saved = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(payload, saved)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(set(payload["coverage"]["bound_jobs"]), set(REQUIRED_JOBS))
        labels = {
            binding["anchor"]["value"]
            for binding in payload["bindings"]
            if binding["anchor"]["kind"] == "latex_label"
        }
        self.assertTrue(set(REQUIRED_RESULT_TABLES) <= labels)
        self.assertEqual(len(payload["passthrough_table_hashes"]), 6)
        self.assertGreaterEqual(payload["coverage"]["numerical_bindings"], 401)
        self.assertTrue(verify_bindings()["passed"])


if __name__ == "__main__":
    unittest.main()
