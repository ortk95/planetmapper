import unittest
from unittest.mock import MagicMock, patch

import common_testing

import planetmapper
import planetmapper.exceptions


class TestExceptions(common_testing.BaseTestCase):
    def test_warn(self):
        with self.assertWarns(planetmapper.exceptions.PlanetmapperWarning) as cm:
            planetmapper.exceptions.warn('Test warning')
        self.assertEqual(str(cm.warning), 'Test warning')

        with self.assertWarns(UserWarning) as cm:
            planetmapper.exceptions.warn('Test warning 2', category=UserWarning)
        self.assertEqual(str(cm.warning), 'Test warning 2')
