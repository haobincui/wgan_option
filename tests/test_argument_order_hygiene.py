import ast
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "venv", "build", "dist", "__pycache__"}
TARGET_DIRS = ("src", "scripts", "tests")

FORMULA_DEFINING_FILES = {
    "_svi_function": "src/quantlib/vol_surface/algo/svi_algo.py",
    "black_scholes_implied_vol": "src/quantlib/calculation/analytics/models/analytical/equity/formula.py",
    "black_scholes_implied_vol_torch": "src/quantlib/calculation/analytics/models/analytical/equity/formula.py",
    "black_scholes_price": "src/quantlib/calculation/analytics/models/analytical/equity/formula.py",
    "black_scholes_price_torch": "src/quantlib/calculation/analytics/models/analytical/equity/formula.py",
    "caplet_black_price": "src/quantlib/calculation/analytics/models/analytical/interest_rate/formula.py",
    "caplet_black_vol": "src/quantlib/calculation/analytics/models/analytical/interest_rate/formula.py",
    "cir_bond_option_price": "src/quantlib/calculation/analytics/models/analytical/interest_rate/formula.py",
    "vasicek_bond_option_price": "src/quantlib/calculation/analytics/models/analytical/interest_rate/formula.py",
}

KEYWORD_ONLY_CONSTRUCTORS = {
    "BlackScholesScenarioDefinitionSingleAsset",
    "ConstVolSurface",
    "DigitalCash",
    "Forward",
    "FuturesContract",
    "ImpliedDistributionByDate",
    "ImpliedDistributionByTenor",
    "InterpolatedImpliedVolSurface",
    "InterpolatedPercentStrikeImpliedVolSurface",
    "InterpolatedPercentStrikeImpliedVolSurfaceByTenors",
    "SpotContract",
    "SviCalibrationGatheral",
    "SviCalibrationQuasiExplicit",
    "VanillaEuropean",
    "VanillaImpliedVolatility",
}


def _iter_repo_python_files():
    for directory in TARGET_DIRS:
        base = ROOT_DIR / directory
        for path in base.rglob("*.py"):
            if any(part in IGNORED_PARTS for part in path.parts):
                continue
            yield path


def _call_name(node: ast.Call):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class TestArgumentOrderHygiene(unittest.TestCase):
    def test_argument_order_rules(self):
        violations = []

        for path in _iter_repo_python_files():
            rel_path = path.relative_to(ROOT_DIR).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                violations.append(f"{rel_path}:{exc.lineno}: failed to parse for hygiene audit: {exc.msg}")
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                name = _call_name(node)
                if name is None:
                    continue

                if name == "count_business_days" and len(node.args) > 2:
                    violations.append(
                        f"{rel_path}:{node.lineno}: count_business_days must use keyword args for include_start/include_end"
                    )
                    continue

                if name == "plus_period" and len(node.args) > 2:
                    violations.append(
                        f"{rel_path}:{node.lineno}: plus_period must use keyword args after start and period"
                    )
                    continue

                if name == "generate_schedule_simple" and len(node.args) > 3:
                    violations.append(
                        f"{rel_path}:{node.lineno}: generate_schedule_simple must use keyword args after start, end, and freq"
                    )
                    continue

                if name == "DayCountBusN" and len(node.args) > 3:
                    violations.append(
                        f"{rel_path}:{node.lineno}: DayCountBusN must use keyword args for include_start/include_end"
                    )
                    continue

                if name in KEYWORD_ONLY_CONSTRUCTORS and node.args:
                    violations.append(
                        f"{rel_path}:{node.lineno}: {name} must be constructed with keyword args"
                    )
                    continue

                defining_file = FORMULA_DEFINING_FILES.get(name)
                if defining_file is not None and rel_path != defining_file and node.args:
                    violations.append(
                        f"{rel_path}:{node.lineno}: {name} must use keyword args outside {defining_file}"
                    )

        self.assertFalse(violations, "Argument-order hygiene violations found:\n" + "\n".join(violations))

    def test_compatibility_cleanup_rules(self):
        violations = []

        config_utils_path = ROOT_DIR / "src/wgan_option/surface_generation/common/config_utils.py"
        config_utils_text = config_utils_path.read_text(encoding="utf-8")
        if 'raw_data.get("surface_builder", raw_data)' in config_utils_text:
            violations.append(
                "src/wgan_option/surface_generation/common/config_utils.py: "
                "flat-root `surface_builder` fallback is not allowed"
            )
        if "if section is None:\n        defaults = dict(root_supported)" in config_utils_text:
            violations.append(
                "src/wgan_option/surface_generation/common/config_utils.py: "
                "missing section must not fall back to root-supported defaults"
            )

        strict_bool_files = [
            ROOT_DIR / "src/wgan_option/config_parsing.py",
            config_utils_path,
        ]
        banned_alias_tokens = ('"yes"', '"no"', '"on"', '"off"', '"1"', '"0"', '"y"', '"n"')
        for path in strict_bool_files:
            text = path.read_text(encoding="utf-8")
            for token in banned_alias_tokens:
                if token in text:
                    violations.append(
                        f"{path.relative_to(ROOT_DIR).as_posix()}: strict boolean parsing must not accept alias token {token}"
                    )

        daycount_text = (ROOT_DIR / "src/quantlib/calendar/daycount.py").read_text(encoding="utf-8")
        if "*args" in daycount_text or "**kwargs" in daycount_text:
            violations.append(
                "src/quantlib/calendar/daycount.py: canonicalized constructor APIs must not reintroduce vararg compatibility shims"
            )

        self.assertFalse(violations, "Compatibility-cleanup hygiene violations found:\n" + "\n".join(violations))
