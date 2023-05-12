import json
import unittest

import planetmapper


class TestDataLoader(unittest.TestCase):
    def test_make_data_path(self):
        p = planetmapper.data_loader.make_data_path('text.txt')
        self.assertTrue(p.endswith('planetmapper/data/text.txt'))

    def test_get_ring_radii(self):
        data = planetmapper.data_loader.get_ring_radii()
        self.assertIsInstance(data, dict)

        self.assertIsInstance(data['JUPITER'], dict)
        self.assertTrue(set(data.keys()) >= {'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE'})
        self.assertEqual(data['SATURN']['A'], [122340.0, 136780.0])

        with open(planetmapper.data_loader.make_data_path('rings.json'), 'r') as f:
            json_data = json.load(f)

        self.assertEqual(data, json_data)
