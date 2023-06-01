import unittest
from unittest.mock import MagicMock, patch

import common_testing
from packaging import version

import planetmapper


class TestInit(unittest.TestCase):
    def test_init(self):
        self.assertEqual(planetmapper.__author__, 'Oliver King')
        self.assertEqual(planetmapper.__url__, 'https://github.com/ortk95/planetmapper')
        self.assertEqual(planetmapper.__license__, 'MIT')
        self.assertEqual(
            planetmapper.__description__,
            'PlanetMapper: A Python module for visualising, navigating and mapping Solar System observations',
        )

    def test_version(self):
        self.assertEqual(planetmapper.__version__.strip(), planetmapper.__version__)
        self.assertEqual(planetmapper.__version__.count('.'), 2)
        self.assertEqual(len(planetmapper.__version__.split('.')), 3)
        self.assertTrue(all(x.isdigit() for x in planetmapper.__version__.split('.')))

        self.assertEqual(
            str(version.Version(planetmapper.__version__)), planetmapper.__version__
        )
        self.assertGreater(
            version.Version(planetmapper.__version__), version.Version('1.6.2')
        )
        self.assertLess(
            version.Version(planetmapper.__version__), version.Version('2.0.0')
        )

    @patch('planetmapper.gui._main')
    def test_main(self, mock_gui_main: MagicMock):
        planetmapper.main()
        mock_gui_main.assert_called_once()
