import unittest

import common_testing

import planetmapper


class TestProgress(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
