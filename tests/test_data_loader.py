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

        # Check that copy is returned properly
        radii = planetmapper.data_loader.get_ring_radii()
        radii['<< test >>'] = {'test': [1.1, 2.2]}
        radii['SATURN']['A'] = [1.1, 2.2]
        del radii[f'JUPITER']
        self.assertNotEqual(radii, json_data)
        self.assertNotEqual(radii, data)
        self.assertNotEqual(radii, planetmapper.data_loader.get_ring_radii())
        self.assertEqual(data, planetmapper.data_loader.get_ring_radii())
        self.assertEqual(json_data, planetmapper.data_loader.get_ring_radii())

    def test_get_ring_aliases(self):
        data = planetmapper.data_loader.get_ring_aliases()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['liberte'], 'liberté')
        self.assertEqual(data['egalite'], 'egalité')

        with open(
            planetmapper.data_loader.make_data_path('ring_aliases.json'),
            encoding='utf-8',
        ) as f:
            json_data = json.load(f)
        self.assertEqual(data, json_data)

        # Check that copy is returned properly
        aliases = planetmapper.data_loader.get_ring_aliases()
        aliases['<< test >>'] = 'test'
        aliases['liberte'] = 'test'
        del aliases['egalite']
        self.assertNotEqual(aliases, json_data)
        self.assertNotEqual(aliases, data)
        self.assertNotEqual(aliases, planetmapper.data_loader.get_ring_aliases())
        self.assertEqual(data, planetmapper.data_loader.get_ring_aliases())
        self.assertEqual(json_data, planetmapper.data_loader.get_ring_aliases())
