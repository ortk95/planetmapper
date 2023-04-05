import unittest
import planetmapper
import common_testing
from packaging import version


class TestInit(unittest.TestCase):
    def test_init(self):
        self.assertEqual(planetmapper.__author__, 'Oliver King')
        self.assertEqual(planetmapper.__url__, 'https://github.com/ortk95/planetmapper')
        self.assertTrue(
            version.Version(planetmapper.__version__) > version.Version('1.0')
        )
        self.assertEqual(planetmapper.__version__.count('.'), 2)
        self.assertEqual(planetmapper.__version__.strip(), planetmapper.__version__)
