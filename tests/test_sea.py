"""The sea, the terrain gene that says who may go into it, stamina, and the biomes' starting groups, as in the C++ version."""

import random
import unittest

from tools.Brain import ACTIONS

from tools.Genome import TERRAINS, cross, decode_terrain, encode_dna, with_terrain
from tools.Simulation import (AMPHIBIOUS_SPEED, BIOME_GROUPS, BIOME_SIZE, DEFAULT_OCEAN, FULL_STAMINA, REST_RECOVERY, Simulation,
                              zone_temperature)
from tools.WorldFile import world_bytes, world_from_bytes


def seaside(seed: int = 3, population: int = 0) -> Simulation:
    simulation = Simulation(40, 40, population, seed, ocean=DEFAULT_OCEAN)
    if not population:
        simulation.grass = {}
    return simulation


def coast(simulation: Simulation) -> tuple[int, int]:
    """A square of land with land just west of it and the sea just east, away from the top and bottom."""
    for y in range(5, simulation.height - 5):
        for x in range(1, simulation.width - 2):
            if not simulation.sea(x - 1, y) and not simulation.sea(x, y) and simulation.sea(x + 1, y):
                return x, y
    raise AssertionError("no coast")


class SeaTests(unittest.TestCase):
    def test_the_sea_covers_its_share_laid_out_by_the_seed_as_the_cpp_version_lays_it(self):
        first, again, other = seaside(3), seaside(3), seaside(4)
        self.assertEqual(sum(first.sea_map), round(40 * 40 * DEFAULT_OCEAN / 100))
        self.assertEqual(first.sea_map, again.sea_map)
        self.assertNotEqual(first.sea_map, other.sea_map)
        self.assertFalse(Simulation(20, 20, 0, 3).has_sea)

    def test_terrain_sits_after_neurodivergence_and_a_mutation_moves_it_along_its_chain(self):
        avian = encode_dna("grazer", terrain="avian", neurodivergence=15)
        self.assertEqual(decode_terrain(avian), "avian")
        self.assertEqual(TERRAINS.index(decode_terrain(with_terrain(avian, "amphibious"))), 3)
        rng = random.Random(12)
        parent = encode_dna("grazer", terrain="amphibious")  # its third species bit clear: setting it, toward the air
        for _ in range(200):
            child = cross(parent, parent, rng, None, 0.0, 1.0)
            flipped = ((child ^ parent) >> 34) & 1
            self.assertEqual(decode_terrain(child), "terrestrial" if flipped else "amphibious")

    def test_the_terrestrial_keep_to_the_land_and_the_aquatic_to_the_sea(self):
        simulation = seaside()
        x, y = coast(simulation)
        walker = simulation._add_agent(encode_dna("grazer"), x, y)
        self.assertFalse(simulation._step(walker, x + 1, y))
        fish = simulation._add_agent(encode_dna("grazer", terrain="aquatic"), x + 1, y)
        self.assertTrue(fish.wet)
        self.assertFalse(simulation._step(fish, x, y))
        gull = simulation._add_agent(encode_dna("grazer", terrain="avian"), x, y)
        self.assertTrue(simulation._step(gull, x + 1, y))
        self.assertTrue(gull.wet)

    def test_a_flyer_is_caught_only_while_it_rests_and_the_amphibious_are_slow(self):
        simulation = seaside()
        x, y = coast(simulation)
        wolf = simulation._add_agent(encode_dna("predator"), x, y)
        gull = simulation._add_agent(encode_dna("grazer", terrain="avian"), x, y)
        self.assertFalse(Simulation._is_prey(wolf, gull))
        gull.last_action = "rested"
        self.assertTrue(Simulation._is_prey(wolf, gull))
        walker = simulation._add_agent(encode_dna("grazer", speed=7), x, y)
        frog = simulation._add_agent(encode_dna("grazer", speed=7, terrain="amphibious"), x, y)
        walker.age = frog.age = int(walker.lifespan / 2)
        self.assertEqual(frog.steps, int(walker.steps * AMPHIBIOUS_SPEED + 0.5))  # on land, slow
        frog.wet = True
        self.assertEqual(frog.steps, walker.steps)  # in the water, as fast as any

    def test_an_exhausted_animal_only_rests_and_a_flyer_exhausted_at_sea_drowns(self):
        simulation = seaside()
        x, y = coast(simulation)
        walker = simulation._add_agent(encode_dna("grazer"), x, y)
        walker.stamina = 0
        simulation._think(walker)
        self.assertEqual(walker.last_action, "rested")
        self.assertEqual(walker.stamina, REST_RECOVERY)
        gull = simulation._add_agent(encode_dna("grazer", terrain="avian"), x + 1, y)
        self.assertNotIn(ACTIONS.index("rest"), simulation._sense(gull)[1])  # no rest at sea
        gull.stamina = 1
        action = next(name for name, (dx, dy) in (("walk_north", (0, -1)), ("walk_south", (0, 1)), ("walk_east", (1, 0)))
                      if simulation.sea(x + 1 + dx, y + dy))
        simulation._act(gull, action, {})
        self.assertFalse(gull.alive)
        self.assertEqual(simulation.death_records[gull.identifier].cause, "drowned")
        self.assertLessEqual(walker.stamina, FULL_STAMINA)

    def test_every_biome_starts_with_groups_at_home_in_it(self):
        for ocean in (DEFAULT_OCEAN, 0.0):
            simulation = Simulation(60, 60, 30, 7, ocean=ocean)
            for biome in simulation.biomes():
                self.assertGreaterEqual(len(biome), BIOME_SIZE)
                home, wet, squares = zone_temperature(biome[0][1], 60), simulation.sea(*biome[0]), set(biome)
                at_home = sum((agent.x, agent.y) in squares and agent.climate[0] == home and agent.can_be(wet)
                              for agent in simulation.agents)
                self.assertGreaterEqual(at_home, BIOME_GROUPS)
        self.assertEqual(Simulation(60, 60, 0, 7).agents, [])  # an empty world stays empty

    def test_the_sea_and_stamina_are_saved_with_the_world(self):
        simulation = seaside(6, 60)
        for _ in range(3):
            simulation.advance()
        loaded = world_from_bytes(world_bytes(simulation))
        self.assertEqual(loaded.sea_map, simulation.sea_map)
        self.assertEqual([(agent.terrain, agent.stamina) for agent in loaded.agents],
                         [(agent.terrain, agent.stamina) for agent in simulation.agents])


if __name__ == "__main__":
    unittest.main()
