import tempfile
import unittest
from pathlib import Path

from tools import Rules
from tools import WorldReport
from tools.Simulation import Simulation
from tools.WorldFile import save_world


class WorldReportTests(unittest.TestCase):
    def tearDown(self):
        Rules.apply({})

    def test_a_save_is_reported_on_whole(self):
        world = Simulation(30, 30, population=60, seed=7)
        for _ in range(30):
            world.advance()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report-test.bzworld"
            save_world(world, path)
            report = WorldReport.read_save(path)
        self.assertEqual(report.name, "report-test.bzworld")
        self.assertFalse(report.live)
        self.assertEqual(report.state["turn"], world.turn)
        self.assertEqual(set(report.herds), {"grazer", "omnivore", "predator"})
        sections = WorldReport.analyse(report)
        self.assertEqual(len(sections), 10)
        self.assertEqual(sections[0][0], "Summary")
        text = WorldReport.as_text(sections)
        self.assertIn("== Herds & migration ==", text)
        self.assertIn("BITZKEY BRAIN REPORT", text)
        self.assertNotIn("\t", text)  # its tables lined up in spaces

    def test_minds_count_runaway_values_and_jammed_neurons(self):
        world = Simulation(30, 30, population=40, seed=3)
        for _ in range(10):
            world.advance()
        minds = WorldReport._minds(world)["minds"]
        self.assertGreater(sum(mind["thinking"] for mind in minds.values()), 0)
        for mind in minds.values():
            self.assertLessEqual(mind["jammed"], mind["neurons"])
            self.assertLessEqual(mind["runaway"], mind["valued"])
            self.assertLessEqual(mind["value_median"], mind["value_max"])

    def test_a_probe_address_is_a_port_host_port_or_a_whole_address(self):
        with self.assertRaisesRegex(RuntimeError, "give the probe as host:port"):
            WorldReport.Probe("not a port")
        with self.assertRaisesRegex(RuntimeError, "no probe answers at 127.0.0.1:1"):
            WorldReport.Probe("127.0.0.1:1")


if __name__ == "__main__":
    unittest.main()
