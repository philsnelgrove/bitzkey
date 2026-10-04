import unittest

from Main import DOT_RADIUS_RANGE, BitzkeyApp
from tools.Genome import encode_dna
from tools.Simulation import Bitzkey


def bitzkey(size: int) -> Bitzkey:
    return Bitzkey(0, encode_dna("grazer", size=size), 0, 0)


class DotSizeTests(unittest.TestCase):
    def test_dot_radius_spans_the_range_by_body_size(self):
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(0)), DOT_RADIUS_RANGE[0])
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(7)), DOT_RADIUS_RANGE[1])

    def test_bigger_bodies_never_have_smaller_dots(self):
        radii = [BitzkeyApp._dot_radius(bitzkey(size)) for size in range(8)]

        self.assertEqual(radii, sorted(radii))
        self.assertEqual(len(set(radii)), DOT_RADIUS_RANGE[1] - DOT_RADIUS_RANGE[0] + 1)

    def test_bigger_cells_raise_the_largest_dot(self):
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(7), largest=14), 14)
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(0), largest=14), DOT_RADIUS_RANGE[0])


if __name__ == "__main__":
    unittest.main()
