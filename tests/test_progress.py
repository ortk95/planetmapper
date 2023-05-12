import unittest

import common_testing

import planetmapper
import planetmapper.progress


class TestProgress(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

    def test_progress(self):
        hooks: list[planetmapper.progress.ProgressHook] = [
            planetmapper.progress.CLIProgressHook(),
            planetmapper.progress.TotalTimingProgressHook(),
        ]

        for hook in hooks:
            with self.subTest(hook=hook):
                obj = planetmapper.BodyXY('jupiter', '2005-01-01', sz=5)
                self.assertIsNone(obj._get_progress_hook())

                obj._set_progress_hook(hook)
                self.assertEqual(obj._get_progress_hook(), hook)

                obj.get_backplane_img('EMISSION')

                obj._remove_progress_hook()
                self.assertIsNone(obj._get_progress_hook())
