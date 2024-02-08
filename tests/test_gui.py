import unittest

import common_testing

from planetmapper.gui import GUI


class TestGUI(common_testing.BaseTestCase):
    def test_init(self):
        GUI()
