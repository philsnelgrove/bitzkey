import json
import random
import unittest

from tools.Brain import Brain
from tools.Exchange import Imported, export_bitzkeys, import_bitzkeys
from tools.Genome import MIND_TRAITS, convert_v1, encode_dna, make_dna
from tools.Simulation import Bitzkey


def bitzkey(dna: int, brain: Brain | None = None) -> Bitzkey:
    return Bitzkey(0, dna, 0, 0, brain=brain)


class ExchangeTests(unittest.TestCase):
    def test_a_plant_exports_as_its_two_hex_colour_words(self):
        dna = (0x7421FF40 << 32) | 0x637E4B01

        self.assertEqual(export_bitzkeys([bitzkey(dna)]), '{"version": 2, "ring": "7421FF40", "dot": "637E4B01"}')

    def test_an_animal_exports_its_current_temperament(self):
        animal = bitzkey(encode_dna("grazer", fear=4))
        animal.mind["fear"] = 5.87654

        record = json.loads(export_bitzkeys([animal]))

        self.assertEqual(list(record["mind"]), list(MIND_TRAITS))
        self.assertEqual(record["mind"]["fear"], 5.877)

    def test_temperament_imports_back_kept_within_range(self):
        text = json.dumps({"version": 2, "ring": "00000001", "dot": "00000000",
                           "mind": {"fear": 6.5, "libido": 99, "telepathy": 3}})

        self.assertEqual(import_bitzkeys(text), [Imported(1 << 32, mind={"fear": 6.5, "libido": 7.0})])

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
        self.assertEqual(import_bitzkeys('{"bitzkeys": [{"version": 2, "ring": "#00000001", "dot": "abcdef01"}]}'),
                         [Imported((1 << 32) | 0xABCDEF01)])
        self.assertEqual(import_bitzkeys('{"version": 2, "ring": "00000001", "dot": "00000000", "brain": null}'),
                         [Imported(1 << 32)])
        self.assertEqual(import_bitzkeys("  "), [])

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
