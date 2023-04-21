import unittest
import planetmapper
import common_testing
from packaging import version


class TestInit(unittest.TestCase):
    def test_init(self):
        self.assertEqual(planetmapper.__author__, 'Oliver King')
        self.assertEqual(planetmapper.__url__, 'https://github.com/ortk95/planetmapper')

    def test_version(self):
        self.assertEqual(planetmapper.__version__.count('.'), 2)
        self.assertEqual(planetmapper.__version__.strip(), planetmapper.__version__)

        self.assertGreater(
            version.Version(planetmapper.__version__), version.Version('1.6.2')
        )
        self.assertEqual(
            str(version.Version(planetmapper.__version__)), planetmapper.__version__
        )
