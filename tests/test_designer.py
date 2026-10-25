import random
import unittest

from display.Designer import FIELDS, DesignerForm
from display.Setup import SetupForm
from tools.Genome import (BODY_TRAITS, MIND_TRAITS, SOCIAL_TRAITS, body_word, decode_body, decode_climate, decode_mind,
                          decode_neurodivergence, decode_social, encode_dna, mind_word, social_word)


def field_index(kind: str, name: str) -> int:
    return FIELDS.index((kind, name))


class DesignerFormTests(unittest.TestCase):
    def test_every_trait_is_exposed_once(self):
        self.assertEqual([name for kind, name in FIELDS if kind == "mind"], list(MIND_TRAITS))
        self.assertEqual([name for kind, name in FIELDS if kind == "body"], list(BODY_TRAITS))
        self.assertEqual([name for kind, name in FIELDS if kind == "social"], [*SOCIAL_TRAITS, "neurodivergence", "terrain"])
        self.assertEqual([name for kind, name in FIELDS if kind == "hex"], ["ring", "dot", "social"])
        self.assertEqual([name for kind, name in FIELDS if kind == "climate"], ["hardiness", "temperature"])

    def test_the_climate_rows_edit_the_top_bits_of_each_word(self):
        form = DesignerForm(encode_dna("grazer", temperature=0, size=5))
        form.select(field_index("climate", "temperature"))
        form.adjust(10)
        form.select(field_index("climate", "hardiness"))
        form.type_text("2")
        self.assertEqual(decode_climate(form.dna), (3, False, True))
        self.assertEqual(decode_body(form.dna)["size"], 5)

    def test_editing_a_trait_changes_only_its_own_bits(self):
        form = DesignerForm(encode_dna("grazer", fear=3, size=7))

        form.select(field_index("mind", "fear"))
        form.adjust(4)
        form.select(field_index("body", "size"))
        form.adjust(-20)

        self.assertEqual(decode_mind(form.dna)["fear"], 7)
        self.assertEqual(decode_body(form.dna)["size"], 0)
        self.assertEqual(form.dna, encode_dna("grazer", fear=7))

    def test_the_social_rows_and_code_edit_the_top_word(self):
        form = DesignerForm(encode_dna("grazer", size=4))
        form.select(field_index("social", "trust"))
        form.adjust(5)
        form.select(field_index("social", "neurodivergence"))
        form.adjust(12)  # it runs to 15
        self.assertEqual((decode_social(form.dna)["trust"], decode_neurodivergence(form.dna)), (5, 12))
        form.select(field_index("hex", "social"))
        form.type_text("0000ABCD")
        self.assertEqual((social_word(form.dna), decode_body(form.dna)["size"]), (0xABCD, 4))

    def test_species_runs_through_all_eight_values(self):
        form = DesignerForm(encode_dna("plant"))
        form.select(field_index("body", "species"))

        seen = []
        for _ in range(8):
            seen.append(decode_body(form.dna)["species"])
            form.adjust(1)

        self.assertEqual(seen[:5], ["plant", "grazer", "omnivore", "predator", "plant"])
        self.assertEqual(form.value(("body", "species")), 7)

    def test_typed_values_are_kept_within_range(self):
        form = DesignerForm(encode_dna("grazer"))
        form.select(field_index("mind", "libido"))

        form.type_text("5")
        self.assertEqual(decode_mind(form.dna)["libido"], 5)
        form.type_text("9")
        self.assertEqual(decode_mind(form.dna)["libido"], 7)
        form.backspace()
        self.assertEqual(decode_mind(form.dna)["libido"], 0)

    def test_typing_a_hex_code_sets_every_trait_in_that_word(self):
        form = DesignerForm(encode_dna("grazer"))
        form.select(field_index("hex", "dot"))

        form.type_text("12345")
        self.assertEqual(form.code("dot"), "12345")
        self.assertEqual(mind_word(form.dna), 0)
        form.type_text("abc")

        self.assertEqual(mind_word(form.dna), 0x12345ABC)
        self.assertEqual(decode_mind(form.dna)["aggression"], 0xABC & 0b111)
        self.assertEqual(body_word(form.dna), body_word(encode_dna("grazer")))

    def test_an_unfinished_hex_code_is_dropped_when_moving_on(self):
        form = DesignerForm(encode_dna("grazer"))
        form.select(field_index("hex", "ring"))
        form.type_text("FFFF")

        form.move(1)

        self.assertEqual(form.dna, encode_dna("grazer"))
        self.assertEqual(form.code("ring"), f"{body_word(encode_dna('grazer')):08X}")

    def test_random_designs_vary(self):
        form = DesignerForm(0)
        rng = random.Random(3)

        designs = set()
        for _ in range(5):
            form.randomise(rng)
            designs.add(form.dna)

        self.assertEqual(len(designs), 5)


class ApplyingDesignsTests(unittest.TestCase):
    def test_each_design_joins_the_bottom_of_the_import_list(self):
        form = SetupForm({"width": 100, "height": 100, "population": 90, "grass_density": 5, "seed": None})
        form.paste('{"version": 3, "ring": "00000001", "dot": "00000000", "social": "00000000"}')
        designs = [encode_dna("predator", size=7, weapons=4), encode_dna("grazer", fear=2)]

        for dna in designs:
            form.add_design(dna)

        self.assertEqual([item.dna for item in form.values()["imported"]], [1 << 32] + designs)
        self.assertTrue(form.import_text.endswith('"dot": "00000400", "social": "00000000"}'))  # fear 2 is the fourth trait, bits 9-11
        self.assertEqual(len(form.import_text.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
