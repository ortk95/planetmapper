import unittest
import planetmapper
import planetmapper.common
import common_testing


class TestCommon(unittest.TestCase):
    def test_init(self):
        self.assertEqual(planetmapper.common.__author__, 'Oliver King')
        self.assertEqual(
            planetmapper.common.__url__, 'https://github.com/ortk95/planetmapper'
        )
        # See also test_init
