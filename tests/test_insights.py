import unittest

from tools.Insights import SCENARIOS, TRAITS, Insights, brain_report
from tools.Simulation import Simulation
from unittest.mock import patch


from tools.Simulation import EIGHT_TURN_DAYS

def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, SMALL_HIDING=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0, **EIGHT_TURN_DAYS)  # and 360-turn years, no sleeping, spans in eight-turn days
    _grown.start()


def tearDownModule():
    _grown.stop()


def profile(**values: float) -> dict[str, float]:
    return {trait: values.get(trait, 0.0) for trait in TRAITS}


class InsightsTests(unittest.TestCase):
    def test_rate_counts_chances_taken(self):
        insights = Insights()
        for turn in range(10):
            insights.record("grazer", profile(), ("breed", "rest"), "breed" if turn < 3 else "rest")

        self.assertEqual(insights.rate("breed", "grazer"), (0.3, 10.0))
        self.assertEqual(insights.rate("breed", "grazer", "libido", "low"), (0.3, 10.0))
        self.assertEqual(insights.rate("attack", "grazer"), (None, 0.0))

    def test_extremes_rank_trait_groups(self):
        insights = Insights()
        for _ in range(10):
            insights.record("grazer", profile(libido=6.6), ("breed", "rest"), "breed")
            insights.record("grazer", profile(), ("breed", "rest"), "rest")

        self.assertEqual(insights.extremes("breed", 1, lowest=True)[0][:2], ("low-libido grazers", 0.0))
        self.assertEqual(insights.extremes("breed", 1, lowest=False)[0][:2], ("high-libido grazers", 1.0))

    def test_old_decisions_fade(self):
        insights = Insights()
        insights.record("grazer", profile(), ("rest",), "rest")
        insights.decay()

        self.assertLess(insights.offered[("grazer", None, None, "rest")], 1.0)

    def test_report_probes_every_scenario_and_temperament(self):
        simulation = Simulation(width=20, height=20, population=40, seed=2)
        for _ in range(5):
            simulation.advance()

        report = brain_report(simulation.turn, simulation.living, simulation.insights)

        for title, *_ in SCENARIOS:
            self.assertIn(title, report)
        self.assertIn("Least keen to breed", report)
        self.assertIn("Temperament shaped by experience", report)


if __name__ == "__main__":
    unittest.main()
