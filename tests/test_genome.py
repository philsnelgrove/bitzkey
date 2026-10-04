import random
import unittest
from unittest.mock import patch

from tools.Genome import (BODY_TRAITS, MIND_TRAITS, TRAIT_MAX, body_word, color_from_word, convert_v1, cross,
                          decode_body, decode_mind, encode_dna, genome_from_record, genome_record, make_dna,
                          mind_word)


class GenomeTests(unittest.TestCase):
    def test_mind_lives_in_low_word_and_body_in_high_word(self):
        dna = encode_dna("predator", fear=4, intellect=7, tolerance=5, size=6, fertility=3, carrion=2)

        self.assertEqual(mind_word(dna), (4 << 12) | (7 << 21) | (5 << 27))
        self.assertEqual(body_word(dna), 3 | (6 << 3) | (3 << 21) | (2 << 27))

    def test_every_word_holds_ten_three_bit_traits(self):
        self.assertEqual((len(MIND_TRAITS), len(BODY_TRAITS), TRAIT_MAX), (10, 10, 7))
        everything = encode_dna("predator", **{trait: TRAIT_MAX for trait in MIND_TRAITS + BODY_TRAITS[1:]})
        self.assertEqual(mind_word(everything) >> 30, 0)  # the top two bits carry no trait

    def test_traits_outside_their_range_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 to 7"):
            encode_dna("grazer", fear=8)

    def test_encoding_round_trips_every_trait(self):
        traits = {trait: index % TRAIT_MAX + 1 for index, trait in enumerate(MIND_TRAITS + BODY_TRAITS[1:])}

        dna = encode_dna("omnivore", **traits)

        self.assertEqual(decode_mind(dna), {trait: traits[trait] for trait in MIND_TRAITS})
        self.assertEqual(decode_body(dna), {"species": "omnivore",
                                            **{trait: traits[trait] for trait in BODY_TRAITS[1:]}})

    def test_unknown_trait_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_dna("grazer", wings=3)

    def test_each_word_sets_its_own_colour(self):
        dna = (0x00FF0000 << 32) | 0xFF000000

        self.assertEqual(color_from_word(mind_word(dna)), (0, 255, 255))
        self.assertEqual(color_from_word(body_word(dna)), (255, 0, 255))

    def test_random_genome_keeps_requested_species(self):
        rng = random.Random(1)

        self.assertTrue(all(decode_body(make_dna(2, rng))["species"] == "omnivore" for _ in range(20)))

    def test_child_of_identical_parents_differs_by_at_most_one_bit_per_word(self):
        rng = random.Random(5)
        parent = make_dna(1, rng)

        for _ in range(50):
            child = cross(parent, parent, rng)
            self.assertLessEqual((mind_word(child) ^ mind_word(parent)).bit_count(), 1)
            self.assertLessEqual((body_word(child) ^ body_word(parent)).bit_count(), 1)

    def test_child_dot_is_the_average_of_its_parents_dot_colours(self):
        rng = random.Random(2)
        first, second = 0x10204080, 0x30604020

        with patch("tools.Genome.MUTATION_CHANCE", 0):
            child = cross(first, second, rng)

        self.assertEqual(mind_word(child), 0x20404050)

    def test_odd_halves_round_either_way(self):
        rng = random.Random(2)

        with patch("tools.Genome.MUTATION_CHANCE", 0):
            bytes_seen = {mind_word(cross(0x00, 0x01, rng)) for _ in range(40)}

        self.assertEqual(bytes_seen, {0x00, 0x01})

    def test_child_ring_takes_each_bit_from_a_parent(self):
        rng = random.Random(4)
        first, second = make_dna(1, rng), make_dna(1, rng)
        agreed = ~(body_word(first) ^ body_word(second)) & 0xFFFFFFFF

        with patch("tools.Genome.MUTATION_CHANCE", 0):
            for _ in range(20):
                child = cross(first, second, rng)
                self.assertEqual(body_word(child) & agreed, body_word(first) & agreed)

    def test_genome_is_recorded_as_its_version_and_two_hex_colour_words(self):
        dna = (0x7421FF41 << 32) | 0x637E4B01

        self.assertEqual(genome_record(dna), {"version": 2, "ring": "7421FF41", "dot": "637E4B01"})
        self.assertEqual(genome_from_record(genome_record(dna)), dna)
        self.assertEqual(genome_from_record({"version": 2, "ring": "#00000001", "dot": "abcdef01"}),
                         (1 << 32) | 0xABCDEF01)

    def test_version_one_genomes_are_converted(self):
        old = ((6 | (15 << 4) | (8 << 28)) << 32) | (10 << 16)  # omnivore, size 15, fertility 8; fear 10

        dna = genome_from_record({"ring": f"{body_word(old):08X}", "dot": f"{mind_word(old):08X}"})

        self.assertEqual(dna, convert_v1(old))
        self.assertEqual(decode_body(dna), {"species": "omnivore", "size": 7, "strength": 0, "armour": 0,
                                            "weapons": 0, "speed": 0, "sight": 0, "fertility": 4, "litter": 0,
                                            "carrion": 0})
        self.assertEqual((decode_mind(dna)["fear"], decode_mind(dna)["tolerance"], decode_mind(dna)["empathy"]),
                         (5, 3, 0))

    def test_unknown_genome_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "version 3"):
            genome_from_record({"version": 3, "ring": "00000001", "dot": "00000000"})

    def test_bad_hex_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eight hex digits"):
            genome_from_record({"ring": "12", "dot": "00000000"})


if __name__ == "__main__":
    unittest.main()
