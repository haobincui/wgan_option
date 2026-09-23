"""Fail-closed manuscript-to-result linkage checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.rq123.verify_chapter3_bootstrap_bindings import _value, render, verify_bindings


class Chapter3BindingTests(unittest.TestCase):
    def test_validation_change_is_negative_of_improvement(self):
        source = {"job_id": "j", "collection": "ratios", "item_id": "later_over_earlier",
                  "field": "observed_improvement_percent"}
        values = {"jobs": {"j": {"ratios": {"later_over_earlier": {
            "observed_improvement_percent": .125}}}}}
        binding = {"source_expression": {"operator": "negate", "operands": [source]}}
        self.assertEqual(_value(values, binding), -.125)
        binding["source_expression"]["operands"] = [source, source]
        with self.assertRaisesRegex(ValueError, "one source operand"):
            _value(values, binding)

    def test_decimal_scientific_and_star_contracts(self):
        self.assertEqual(render(.0012345, {"style": "fixed", "digits": 4}), "0.0012")
        self.assertEqual(render(-.0000716092, {"style": "latex_scientific", "digits": 4}),
                         "-7.161\\times10^{-5}")
        self.assertEqual(render(.00234, {"style": "significant", "digits": 2,
                                      "explicit_plus": True, "math_mode": True}), "$+0.0023$")
        self.assertEqual(render("", {"style": "stars"}), "")
        with self.assertRaises(ValueError):
            render(float("nan"), {"style": "fixed"})

    def _case(self, base: Path):
        tex = base / "chapter.tex"
        values = base / "values.json"
        binding_path = base / "bindings.json"
        tex.write_text("\\label{table}\nModel & 0.0400 \\\\\n"
                       "RQ4 starts\nOld result\nRobustness starts\n", encoding="utf-8")
        values.write_text(json.dumps({
            "kind": "chapter3_shared_market_panel_bootstrap_v2",
            "jobs": {"j": {"contrasts": {"c": {"reported_p": .04}}}},
        }), encoding="utf-8")
        payload = {
            "schema_version": 1, "kind": "chapter3_bootstrap_tex_bindings",
            "status": "complete",
            "tex_path": str(tex), "values_path": str(values),
            "scope": {"excluded_anchor_range": {
                "start": "RQ4 starts", "end": "Robustness starts"}},
            "bindings": [{"binding_id": "table_row", "anchor": {
                "kind": "latex_label", "value": "table", "window_lines": 2},
                "expected_substring": "Model & 0.0400", "values": [{
                    "literal": "0.0400", "source": {
                        "job_id": "j", "collection": "contrasts", "item_id": "c",
                        "field": "reported_p"}, "format": {"style": "fixed", "digits": 4},
                }]}],
        }
        binding_path.write_text(json.dumps(payload), encoding="utf-8")
        return binding_path, payload, tex, values

    def test_result_and_text_are_both_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, tex, _ = self._case(Path(directory))
            qa = verify_bindings(path)
            self.assertTrue(qa["passed"])
            self.assertEqual(qa["numerical_bindings"], 1)
            tex.write_text(tex.read_text().replace("0.0400", "0.0300"))
            with self.assertRaisesRegex(ValueError, "Chapter text drift"):
                verify_bindings(path)

    def test_edited_binding_cannot_hide_changed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, tex, _ = self._case(Path(directory))
            tex.write_text(tex.read_text().replace("0.0400", "0.0300"))
            payload["bindings"][0]["expected_substring"] = "Model & 0.0300"
            payload["bindings"][0]["values"][0]["literal"] = "0.0300"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Numerical drift"):
                verify_bindings(path)

    def test_rq4_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, _, _ = self._case(Path(directory))
            payload["bindings"][0]["anchor"] = {
                "kind": "unique_text", "value": "Old result", "window_lines": 3}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "RQ4"):
                verify_bindings(path)

    def test_missing_bindings_and_v1_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload, _, values = self._case(Path(directory))
            values.write_text(json.dumps({"kind": "v1"}))
            with self.assertRaisesRegex(ValueError, "corrected v2"):
                verify_bindings(path)
            payload["bindings"] = []
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_bindings(path)


if __name__ == "__main__":
    unittest.main()
