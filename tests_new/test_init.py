import unittest
import planetmapper
import common_testing

class TestInit(unittest.TestCase):
    def test_init(self):
        self.assertEqual(planetmapper.__author__, 'Oliver King')
        self.assertEqual(planetmapper.__url__, 'https://github.com/ortk95/planetmapper')