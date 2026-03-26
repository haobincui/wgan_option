import sys
import unittest
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").exists())
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_data.contract_handler.utils import has_number_before_letter

class TestNumberBeforeText(unittest.TestCase):
    def test_has_number_before_text(self):
        option_contract_ids = ['FLG10000O20', 'FLG12750N0', 'FLG8500L3',
                               'FLG10025O0', 'FLG12750A30', 'FLG8500D3',
                               'TY80D24', 'TY80D4', 'TY100D4', 'TY100D24',
                               'TY8175D24', 'TY8175D4', 'TY10075D24', 'TY10075D4']

        for option_contract_id in option_contract_ids:
            self.assertTrue(has_number_before_letter(option_contract_id))

        future_contract_ids = ['FLGO20', 'FLGN0', 'FLGL3',
                               'FLGO0', 'FLGA30', 'FLGD3',
                               'TYD24', 'TYD4', 'TYD4', 'TYD24',
                               'TYD24', 'TYD4', 'TYD24', 'TYD4']

        for future_contract_id in future_contract_ids:
            self.assertFalse(has_number_before_letter(future_contract_id))
