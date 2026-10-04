import random
import unittest
from unittest.mock import patch

from tools.Genome import (BODY_TRAITS, MIND_TRAITS, body_word, color_from_word, cross, decode_body,
                          decode_mind, encode_dna, genome_from_record, genome_record, make_dna, mind_word)


class GenomeTests(unittest.TestCase):
    def test_mind_lives_in_low_word_and_body_in_high_word(self):
        dna = encode_dna("predator", fear=9, intellect=15, size=12, fertility=3)

        self.assertEqual(mind_word(dna), (9 << 16) | (15 << 28))
        self.assertEqual(body_word(dna), 3 | (12 << 4) | (3 << 28))

    def test_encoding_round_trips_every_trait(self):
        traits = {trait: index + 1 for index, trait in enumerate(MIND_TRAITS + BODY_TRAITS[1:])}

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

    def test_genome_is_recorded_as_its_two_hex_colour_words(self):
        dna = (0x7421FF41 << 32) | 0x637E4B01

        self.assertEqual(genome_record(dna), {"ring": "7421FF41", "dot": "637E4B01"})
        self.assertEqual(genome_from_record(genome_record(dna)), dna)
        self.assertEqual(genome_from_record({"ring": "#00000001", "dot": "abcdef01"}), (1 << 32) | 0xABCDEF01)

    def test_bad_hex_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eight hex digits"):
            genome_from_record({"ring": "12", "dot": "00000000"})


if __name__ == "__main__":
    unittest.main()
