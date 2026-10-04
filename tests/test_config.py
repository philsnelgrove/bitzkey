import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import config.Globals


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config.Globals)

    def test_files_are_written_to_the_project_folder_from_source(self):
        project = Path(config.Globals.__file__).resolve().parent.parent

        self.assertEqual(config.Globals.Config.APP_DIR, project)

    def test_files_are_written_beside_the_packaged_executable(self):
        executable = Path("C:/Games/Bitzkey/Bitzkey.exe")

        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(executable)):
            importlib.reload(config.Globals)

        self.assertEqual(config.Globals.Config.APP_DIR, executable.resolve().parent)


if __name__ == "__main__":
    unittest.main()
