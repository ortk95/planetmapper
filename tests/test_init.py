import unittest
from unittest.mock import MagicMock, patch

import common_testing
from packaging import version

import planetmapper


class TestInit(common_testing.BaseTestCase):
    def test_dunder_info(self):
        self.assertEqual(planetmapper.__author__, 'Oliver King')
        self.assertEqual(planetmapper.__url__, 'https://github.com/ortk95/planetmapper')
        self.assertEqual(planetmapper.__license__, 'MIT')
        self.assertEqual(
            planetmapper.__description__,
            'PlanetMapper: A Python package for visualising, navigating and mapping Solar System observations',
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

    def test_all(self):
        """
        Test that all submodules, classes and functions are imported correctly.

        E.g. this tests that planetmapper.run_gui and planetmapper.gui.run_gui are the
        same thing, then the actual functionality of the run_gui function is tested in
        test_gui.py.
        """
        self.assertEqual(len(planetmapper.__all__), 19)  # ensure tests are up to date

        self.assertIs(planetmapper.run_gui, planetmapper.gui.run_gui)
        self.assertIs(planetmapper.set_kernel_path, planetmapper.base.set_kernel_path)
        self.assertIs(planetmapper.get_kernel_path, planetmapper.base.get_kernel_path)
        self.assertIs(planetmapper.SpiceBase, planetmapper.base.SpiceBase)
        self.assertIs(planetmapper.Body, planetmapper.body.Body)
        self.assertIs(planetmapper.Backplane, planetmapper.body_xy.Backplane)
        self.assertIs(planetmapper.BodyXY, planetmapper.body_xy.BodyXY)
        self.assertIs(planetmapper.Observation, planetmapper.observation.Observation)
        self.assertIs(planetmapper.BasicBody, planetmapper.basic_body.BasicBody)
        self.assertIs(
            planetmapper.AngularCoordinateKwargs,
            planetmapper.body.AngularCoordinateKwargs,
        )
        self.assertIs(planetmapper.WireframeKwargs, planetmapper.body.WireframeKwargs)
        self.assertIs(
            planetmapper.WireframeComponent, planetmapper.body.WireframeComponent
        )
        self.assertIs(
            planetmapper.DEFAULT_WIREFRAME_FORMATTING,
            planetmapper.body.DEFAULT_WIREFRAME_FORMATTING,
        )
        self.assertIs(planetmapper.MapKwargs, planetmapper.body_xy.MapKwargs)

        # test backward compatible aliases
        self.assertIs(
            planetmapper.body._WireframeKwargs, planetmapper.body.WireframeKwargs
        )
        self.assertIs(
            planetmapper.body._WireframeComponent, planetmapper.body.WireframeComponent
        )
        self.assertIs(planetmapper.body_xy._MapKwargs, planetmapper.body_xy.MapKwargs)
