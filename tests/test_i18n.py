import json
import pathlib
import re
import unittest

from tools.Brain import ACTIONS

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestTranslations(unittest.TestCase):
    def test_every_action_and_activity_shown_is_translated_in_every_language(self):
        """An action's name is shown on the insights panel, an activity in the inspector."""
        shown = {action.upper() for action in ACTIONS}
        shown |= set(re.findall(r'last_action = "([^"]*[^" ])"', (ROOT / "tools" / "Simulation.py").read_text(encoding="utf-8")))
        catalogs = sorted((ROOT / "i18n").glob("*.json"))
        self.assertEqual(len(catalogs), 11)
        for catalog in catalogs:
            keys = {key.lower() for key in json.loads(catalog.read_text(encoding="utf-8"))}  # looked up regardless of case
            missing = sorted(text for text in shown if text.lower() not in keys)
            self.assertEqual(missing, [], catalog.name)


if __name__ == "__main__":
    unittest.main()
