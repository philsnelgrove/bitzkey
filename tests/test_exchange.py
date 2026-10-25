import json
import random
import unittest

from tools.Brain import Brain
from tools.Exchange import Imported, export_bitzkeys, import_bitzkeys
from tools.Genome import TEMPERAMENT, convert_v1, convert_v2, encode_dna, make_dna
from tools.Simulation import Bitzkey
from unittest.mock import patch


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0)  # and 360-turn years, no sleeping
    _grown.start()


def tearDownModule():
    _grown.stop()


def bitzkey(dna: int, brain: Brain | None = None) -> Bitzkey:
    return Bitzkey(0, dna, 0, 0, brain=brain)


class ExchangeTests(unittest.TestCase):
    def test_a_plant_exports_as_its_three_hex_colour_words(self):
        dna = (0x0A1B2C3D << 64) | (0x7421FF40 << 32) | 0x637E4B01

        self.assertEqual(export_bitzkeys([bitzkey(dna)]),
                         '{"version": 4, "ring": "7421FF40", "dot": "637E4B01", "social": "0A1B2C3D"}')

    def test_an_animal_exports_its_current_temperament(self):
        animal = bitzkey(encode_dna("grazer", fear=4))
        animal.mind["fear"] = 5.87654

        record = json.loads(export_bitzkeys([animal]))

        self.assertEqual(list(record["mind"]), list(TEMPERAMENT))
        self.assertEqual(record["mind"]["fear"], 5.877)

    def test_temperament_imports_back_kept_within_range(self):
        text = json.dumps({"version": 3, "ring": "00000001", "dot": "00000000", "social": "00000000",
                           "mind": {"fear": 6.5, "libido": 99, "telepathy": 3, "loyalty": 2}})

        self.assertEqual(import_bitzkeys(text), [Imported(1 << 32, mind={"fear": 6.5, "libido": 7.0, "loyalty": 2.0})])

    def test_a_thinking_bitzkey_exports_its_brain_too(self):
        brain = Brain.random(random.Random(3))

        record = json.loads(export_bitzkeys([bitzkey(encode_dna("predator"), brain)]))

        self.assertEqual(record["brain"], brain.to_record())

    def test_exports_import_back_with_their_brains(self):
        rng = random.Random(8)
        trained = Brain.random(rng)
        first, second, third = (make_dna(1, rng) for _ in range(3))
        text = export_bitzkeys([bitzkey(first, trained), bitzkey(second)]) + "\n" + export_bitzkeys([bitzkey(third)])

        imported = import_bitzkeys(text)

        self.assertEqual([item.dna for item in imported], [first, second, third])
        self.assertEqual(imported[0].brain, Brain.from_record(trained.to_record()))
        self.assertEqual((imported[1].brain, imported[2].brain), (None, None))
        self.assertIsNotNone(imported[1].mind)

    def test_hand_written_records_import(self):
        self.assertEqual(import_bitzkeys('{"bitzkeys": [{"version": 3, "ring": "#00000001", "dot": "abcdef01", '
                                         '"social": "00000002"}]}'),
                         [Imported((2 << 64) | (1 << 32) | 0xABCDEF01)])
        self.assertEqual(import_bitzkeys('{"version": 3, "ring": "00000001", "dot": "00000000", "social": "00000000", '
                                         '"brain": null}'),
                         [Imported(1 << 32)])
        self.assertEqual(import_bitzkeys("  "), [])

    def test_version_two_records_are_converted_keeping_their_social_temperament(self):
        imported = import_bitzkeys('{"version": 2, "ring": "00000001", "dot": "00000000", "mind": {"empathy": 5.5}}')

        self.assertEqual(imported, [Imported(convert_v2(1 << 32), mind={"empathy": 5.5})])

    def test_version_one_records_are_converted_with_their_temperament(self):
        imported = import_bitzkeys('{"ring": "000000F1", "dot": "000A0000", "mind": {"fear": 15, "libido": 3}}')

        self.assertEqual(imported, [Imported(convert_v1((0xF1 << 32) | 0x000A0000), mind={"fear": 7.0, "libido": 1.4})])

    def test_unreadable_imports_say_what_is_wrong(self):
        broken_brain = json.dumps({"ring": "00000001", "dot": "00000000", "brain": {"senses": []}})
        for text, problem in (('{"ring": "12", "dot": "00000000"}', "eight hex digits"),
                              ('{"ring": "00000000"', "Not valid JSON"),
                              ('[{"colour": "red"}]', 'with "ring" and "dot"'),
                              (broken_brain, "00000001/00000000: The brain is incomplete"),
                              ('{"ring": "00000001", "dot": "0", "mind": 4}', "eight hex digits"),
                              ('{"ring": "00000001", "dot": "00000000", "mind": 4}', "must map trait names"),
                              ('{"ring": "00000001", "dot": "00000000", "mind": {"fear": "lots"}}',
                               '"fear" must be a number')):
            with self.assertRaisesRegex(ValueError, problem):
                import_bitzkeys(text)


if __name__ == "__main__":
    unittest.main()
