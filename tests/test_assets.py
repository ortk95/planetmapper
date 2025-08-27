import json
import os
import unittest

import common_testing
from PIL import Image

import planetmapper._assets


class TestAssets(common_testing.BaseTestCase):
    def test_make_data_path(self):
        path = planetmapper._assets.make_asset_path('text.txt')
        self.assertTrue(
            path.endswith(os.path.join('planetmapper', 'assets', 'text.txt'))
        )

    def test_gui_icon(self):
        path = planetmapper._assets.get_gui_icon_path()
        self.assertTrue(
            path.endswith(os.path.join('planetmapper', 'assets', 'gui_icon.png'))
        )
        self.assertTrue(os.path.exists(path))

        with Image.open(path) as img:
            self.assertEqual(img.size, (256, 256))
