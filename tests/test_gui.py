import unittest

import common_testing

from planetmapper.gui import GUI


class TestGUI(unittest.TestCase):
    def test_init(self):
        GUI()
