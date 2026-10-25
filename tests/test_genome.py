import random
import unittest
from unittest.mock import patch

from tools.Genome import (BODY_TRAITS, MIND_MUTATION_CHANCE, MIND_MUTATION_ROLLS, MIND_TRAITS, SOCIAL_TRAITS, SPARE_BITS,
                          TEMPERAMENT, TEMPERATURES, TRAIT_MAX, body_word, color_from_word, convert_v1, convert_v2, cross,
                          decode_body, decode_climate, decode_mind, decode_neurodivergence, decode_social,
                          decode_temperament, encode_dna, genome_from_record, genome_record, inherit_mind, make_dna,
                          mind_word, social_difference, social_word, with_terrain)


class GenomeTests(unittest.TestCase):
    def test_mind_lives_in_low_word_body_in_the_next_and_social_background_in_the_top(self):
        dna = encode_dna("predator", fear=4, intellect=7, moodiness=5, size=6, fertility=3, carrion=2, tolerance=5,
                         kinship=1, neurodivergence=9)

        self.assertEqual(mind_word(dna), (4 << 9) | (7 << 18) | (5 << 27))
        self.assertEqual(body_word(dna), 3 | (6 << 3) | (3 << 21) | (2 << 27) | (1 << 30))  # a medium home
        self.assertEqual(social_word(dna), (5 << 6) | (1 << 18) | (9 << 21))
        self.assertEqual(dna >> 96, 0)

    def test_the_dot_and_ring_hold_ten_traits_and_the_social_background_seven_and_neurodivergence(self):
        self.assertEqual((len(MIND_TRAITS), len(BODY_TRAITS), len(SOCIAL_TRAITS), TRAIT_MAX), (10, 10, 7, 7))
        self.assertEqual(TEMPERAMENT, MIND_TRAITS + SOCIAL_TRAITS)
        everything = encode_dna("predator", **{trait: TRAIT_MAX for trait in TEMPERAMENT + BODY_TRAITS[1:]},
                                neurodivergence=15)
        self.assertEqual(mind_word(everything) >> 30, 0)  # the top two bits carry no trait
        self.assertEqual(social_word(everything) >> 25, 0)  # nor the social background's top seven
        self.assertEqual(decode_neurodivergence(everything), 15)
        with self.assertRaisesRegex(ValueError, "neurodivergence to 15"):
            encode_dna("grazer", neurodivergence=16)

    def test_the_climate_genes_hold_the_top_bits_of_each_word(self):
        self.assertEqual(TEMPERATURES, ("hot", "medium", "cold", "very cold"))
        dna = encode_dna("grazer", temperature=3, cold_tolerance=True, size=7)
        self.assertEqual(decode_climate(dna), (3, False, True))
        self.assertEqual((body_word(dna) >> 30, mind_word(dna) >> 30), (3, 2))
        self.assertEqual(decode_body(dna)["size"], 7)  # the traits are untouched
        self.assertEqual(decode_climate(encode_dna("plant", heat_tolerance=True)), (1, True, False))
        with self.assertRaisesRegex(ValueError, "0 to 3"):
            encode_dna("grazer", temperature=4)

    def test_a_child_inherits_its_tolerances_together_from_one_parent(self):
        hardy = encode_dna("grazer", heat_tolerance=True, cold_tolerance=True)
        tender = encode_dna("grazer", cold_tolerance=True)
        rng = random.Random(5)
        tolerances = {decode_climate(cross(hardy, tender, rng))[1:] for _ in range(60)}
        self.assertEqual(tolerances, {(False, False), (False, True)})  # both flags cancel out, or only cold

    def test_both_tolerance_flags_cancel_out(self):
        self.assertEqual(decode_climate(encode_dna("grazer", heat_tolerance=True, cold_tolerance=True)), (1, False, False))

    def test_traits_outside_their_range_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 to 7"):
            encode_dna("grazer", fear=8)

    def test_encoding_round_trips_every_trait(self):
        traits = {trait: index % TRAIT_MAX + 1 for index, trait in enumerate(TEMPERAMENT + BODY_TRAITS[1:])}

        dna = encode_dna("omnivore", **traits)

        self.assertEqual(decode_mind(dna), {trait: traits[trait] for trait in MIND_TRAITS})
        self.assertEqual(decode_social(dna), {trait: traits[trait] for trait in SOCIAL_TRAITS})
        self.assertEqual(decode_temperament(dna), {trait: traits[trait] for trait in TEMPERAMENT})
        self.assertEqual(decode_body(dna), {"species": "omnivore",
                                            **{trait: traits[trait] for trait in BODY_TRAITS[1:]}})

    def test_unknown_trait_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_dna("grazer", wings=3)

    def test_each_word_sets_its_own_colour(self):
        dna = (0x0000FF00 << 64) | (0x00FF0000 << 32) | 0xFF000000

        self.assertEqual(color_from_word(mind_word(dna)), (0, 255, 255))
        self.assertEqual(color_from_word(body_word(dna)), (255, 0, 255))
        self.assertEqual(color_from_word(social_word(dna)), (255, 255, 0))
        self.assertEqual(social_difference(dna, 0), 0.25)

    def test_random_genome_keeps_requested_species(self):
        rng = random.Random(1)

        self.assertTrue(all(decode_body(make_dna(2, rng))["species"] == "omnivore" for _ in range(20)))

    def test_child_of_identical_parents_differs_by_one_ring_bit_and_mind_steps(self):
        rng = random.Random(5)
        parent = make_dna(1, rng)

        for _ in range(50):
            child = cross(parent, parent, rng)
            self.assertLessEqual((body_word(child) ^ body_word(parent)).bit_count(), 1)
            self.assertLessEqual(((mind_word(child) ^ mind_word(parent)) & SPARE_BITS).bit_count(), 1)
            self.assertEqual(decode_neurodivergence(child), decode_neurodivergence(parent))
            steps = sum(abs(value - decode_temperament(parent)[trait])
                        for trait, value in decode_temperament(child).items())
            self.assertLessEqual(steps, MIND_MUTATION_ROLLS)  # a step a roll, at most, which may fall on one trait

    def test_a_mutation_landing_on_the_species_is_rolled_afresh_half_the_time(self):
        rng = random.Random(9)
        parent = encode_dna("grazer")
        trials = 40000
        changed = sum((body_word(cross(parent, parent, rng, body_chance=1.0)) ^ body_word(parent)) & 0x3 != 0
                      for _ in range(trials))
        expected = 2 / 32 * 0.5 + 2 / 32 * 0.5 * 2 / 32  # kept, or rolled afresh onto the species again
        self.assertAlmostEqual(changed / trials, expected, delta=0.005)

    def test_child_dot_and_social_background_hold_the_average_of_its_parents_traits(self):
        rng = random.Random(2)
        first = encode_dna("grazer", fear=2, curiosity=6, empathy=7, loyalty=1)
        second = encode_dna("grazer", fear=4, curiosity=0, empathy=7, loyalty=5)

        with patch("tools.Genome.MIND_MUTATION_CHANCE", 0):
            child = decode_temperament(cross(first, second, rng))

        self.assertEqual((child["fear"], child["curiosity"], child["empathy"], child["loyalty"]), (3, 3, 7, 3))

    def test_a_child_inherits_its_neurodivergence_whole_from_one_parent(self):
        first, second = encode_dna("grazer", neurodivergence=3), encode_dna("grazer", neurodivergence=12)
        rng = random.Random(8)
        self.assertEqual({decode_neurodivergence(cross(first, second, rng)) for _ in range(60)}, {3, 12})

    def test_a_given_mind_is_written_into_the_dot_rounding_fractions_by_chance(self):
        rng = random.Random(2)
        mind = {trait: 0.0 for trait in TEMPERAMENT} | {"fear": 3.25, "libido": 7.0, "trust": 6.0}

        fears = [decode_mind(cross(0, 0, rng, mind))["fear"] for _ in range(2000)]

        self.assertEqual(set(fears), {3, 4})
        self.assertAlmostEqual(sum(fears) / len(fears), 3.25, delta=0.04)  # no drift
        self.assertEqual(decode_mind(cross(0, 0, rng, mind))["libido"], 7)
        self.assertEqual(decode_social(cross(0, 0, rng, mind))["trust"], 6)

    def test_a_birth_carries_up_to_three_mind_mutations_each_with_the_chance_given(self):
        rng = random.Random(3)
        mind = {trait: 3.0 for trait in TEMPERAMENT}
        mutations = []
        for chance in (0.0, 0.5, 1.0):
            counted = []
            for _ in range(3000):
                child = inherit_mind(mind, mind, rng, chance)
                steps = [abs(child[trait] - 3) for trait in TEMPERAMENT]
                self.assertLessEqual(max(steps), 3)
                counted.append(sum(steps))
            mutations.append(sum(counted) / len(counted))
        self.assertEqual(mutations[0], 0)
        # three rolls: about 1.5 mutations a birth at 50, three at 100 (less a few that cancel out)
        self.assertAlmostEqual(mutations[1], 1.5 * 0.9, delta=0.15)
        self.assertAlmostEqual(mutations[2], 3 * 0.9, delta=0.2)
        self.assertEqual((MIND_MUTATION_CHANCE, MIND_MUTATION_ROLLS), (0.25, 3))

    def test_a_certain_mutation_steps_away_from_the_limits(self):
        rng = random.Random(4)
        for value in (0.0, 7.0):
            mind = {trait: value for trait in TEMPERAMENT}
            for _ in range(50):
                child = inherit_mind(mind, mind, rng, 1.0)
                self.assertTrue(all(0 <= child[trait] <= 7 for trait in TEMPERAMENT))
                self.assertGreater(sum(abs(child[trait] - value) for trait in TEMPERAMENT), 0)

    def test_the_body_mutates_with_the_chance_given(self):
        rng = random.Random(6)
        parent = make_dna(1, rng)
        for chance, expected in ((0.0, 0), (1.0, 1)):
            for _ in range(30):
                child = cross(parent, parent, rng, body_chance=chance, mind_chance=0.0, terrain_mutation=False)
                self.assertEqual((body_word(child) ^ body_word(parent)).bit_count(), expected)
                self.assertEqual((mind_word(child), social_word(child)), (mind_word(parent), social_word(parent)))

    def test_a_childs_mind_starts_at_its_parents_current_average(self):
        with patch("tools.Genome.MIND_MUTATION_CHANCE", 0):
            mind = inherit_mind({trait: 6.5 for trait in TEMPERAMENT}, {trait: 2.0 for trait in TEMPERAMENT},
                                random.Random(1))
        self.assertEqual(set(mind.values()), {4.25})

    def test_child_ring_takes_each_bit_from_a_parent(self):
        rng = random.Random(4)
        first, second = make_dna(1, rng), make_dna(1, rng)
        agreed = ~(body_word(first) ^ body_word(second)) & 0xFFFFFFFF

        with patch("tools.Genome.BODY_MUTATION_CHANCE", 0):
            for _ in range(20):
                child = cross(first, second, rng)
                self.assertEqual(body_word(child) & agreed, body_word(first) & agreed)

    def test_genome_is_recorded_as_its_version_and_three_hex_colour_words(self):
        dna = (0x0A1B2C3D << 64) | (0x7421FF41 << 32) | 0x637E4B01

        self.assertEqual(genome_record(dna), {"version": 4, "ring": "7421FF41", "dot": "637E4B01", "social": "0A1B2C3D"})
        self.assertEqual(genome_from_record(genome_record(dna)), dna)
        # Version 3 had no terrain, its bits spare: it is terrestrial.
        self.assertEqual(genome_from_record({**genome_record(dna), "version": 3}), with_terrain(dna, "terrestrial"))
        self.assertEqual(genome_from_record({"version": 3, "ring": "#00000001", "dot": "abcdef01", "social": "00000002"}),
                         (2 << 64) | (1 << 32) | 0xABCDEF01)
        with self.assertRaisesRegex(ValueError, 'needs "social"'):
            genome_from_record({"version": 3, "ring": "00000001", "dot": "00000000"})

    def test_version_two_genomes_move_their_social_traits_to_the_social_background(self):
        # Version 2's dot: aggression 1, sociability 2, curiosity 6, intellect 5, empathy 4, tolerance 3; cold-hardy.
        old_dot = 1 | (2 << 6) | (6 << 9) | (5 << 21) | (4 << 24) | (3 << 27) | (1 << 31)
        dna = genome_from_record({"version": 2, "ring": "40000002", "dot": f"{old_dot:08X}"})
        self.assertEqual(dna, convert_v2((0x40000002 << 32) | old_dot))
        self.assertEqual(body_word(dna), 0x40000002)  # the ring as it was
        self.assertEqual(decode_climate(dna), (1, False, True))
        self.assertEqual(decode_mind(dna), {"aggression": 1, "libido": 0, "curiosity": 6, "fear": 0, "appetite": 0,
                                            "patience": 0, "intellect": 5, "memory": 3, "resilience": 3, "moodiness": 3})
        self.assertEqual(decode_social(dna), {"sociability": 2, "empathy": 4, "tolerance": 3, "loyalty": 3,
                                              "dominance": 3, "trust": 3, "kinship": 3})
        self.assertEqual(decode_neurodivergence(dna), 0)

    def test_version_one_genomes_are_converted(self):
        old = ((6 | (15 << 4) | (8 << 28)) << 32) | (10 << 16)  # omnivore, size 15, fertility 8; fear 10

        dna = genome_from_record({"ring": f"{body_word(old):08X}", "dot": f"{mind_word(old):08X}"})

        self.assertEqual(dna, convert_v1(old))
        self.assertEqual(decode_body(dna), {"species": "omnivore", "size": 7, "strength": 0, "armour": 0,
                                            "weapons": 0, "speed": 0, "sight": 0, "fertility": 4, "litter": 0,
                                            "carrion": 0})
        self.assertEqual((decode_mind(dna)["fear"], decode_social(dna)["tolerance"], decode_social(dna)["empathy"],
                          decode_mind(dna)["memory"]), (5, 3, 0, 3))

    def test_unknown_genome_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "version 5"):
            genome_from_record({"version": 5, "ring": "00000001", "dot": "00000000", "social": "00000000"})

    def test_bad_hex_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eight hex digits"):
            genome_from_record({"ring": "12", "dot": "00000000"})


if __name__ == "__main__":
    unittest.main()
