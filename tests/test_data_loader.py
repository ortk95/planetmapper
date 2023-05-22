import json
import os
import unittest

import planetmapper


class TestDataLoader(unittest.TestCase):
    def test_make_data_path(self):
        p = planetmapper.data_loader.make_data_path('text.txt')
        self.assertTrue(p.endswith(os.path.join('planetmapper', 'data', 'text.txt')))

    def test_get_ring_radii(self):
        data = planetmapper.data_loader.get_ring_radii()
        self.assertIsInstance(data, dict)

        self.assertIsInstance(data['JUPITER'], dict)
        self.assertTrue(set(data.keys()) >= {'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE'})
        self.assertEqual(data['SATURN']['A'], [122340.0, 136780.0])
        self.assertEqual(data['SATURN']['B'], [91975.0, 117507.0])
        self.assertEqual(data['SATURN']['C'], [74658.0, 91975.0])

        with open(
            planetmapper.data_loader.make_data_path('rings.json'), encoding='utf-8'
        ) as f:
            json_data = json.load(f)

        self.assertEqual(data, json_data)
