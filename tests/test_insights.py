import unittest

from tools.Insights import SCENARIOS, Insights, brain_report
from tools.Simulation import Simulation, decode_dna


class InsightsTests(unittest.TestCase):
    def test_rate_counts_chances_taken(self):
        insights = Insights()
        traits = decode_dna(0)
        for turn in range(10):
            insights.record("grazer", traits, ("breed", "rest"), "breed" if turn < 3 else "rest")

        self.assertEqual(insights.rate("breed", "grazer"), (0.3, 10.0))
        self.assertEqual(insights.rate("breed", "grazer", "fertility", "low"), (0.3, 10.0))
        self.assertEqual(insights.rate("attack", "grazer"), (None, 0.0))

    def test_extremes_rank_trait_groups(self):
        insights = Insights()
        keen, reluctant = decode_dna(15 << 16), decode_dna(0)
        for turn in range(10):
            insights.record("grazer", keen, ("breed", "rest"), "breed")
            insights.record("grazer", reluctant, ("breed", "rest"), "rest")

        self.assertEqual(insights.extremes("breed", 1, lowest=True)[0][:2], ("low-fertility grazers", 0.0))
        self.assertEqual(insights.extremes("breed", 1, lowest=False)[0][1], 1.0)

    def test_old_decisions_fade(self):
        insights = Insights()
        insights.record("grazer", decode_dna(0), ("rest",), "rest")
        insights.decay()

        self.assertLess(insights.offered[("grazer", None, None, "rest")], 1.0)

    def test_report_probes_every_scenario(self):
        simulation = Simulation(width=20, height=20, population=40, seed=2)

        report = brain_report(simulation.turn, simulation.living, simulation.insights)

        for title, *_ in SCENARIOS:
            self.assertIn(title, report)
        self.assertIn("Least keen to breed", report)


if __name__ == "__main__":
    unittest.main()
