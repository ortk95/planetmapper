import unittest

import common_testing

import planetmapper
import planetmapper.progress


class TestError(Exception):
    pass


class ExceptionHook(planetmapper.progress.ProgressHook):
    def __call__(self, progress: float, stack: list[str]) -> None:
        raise TestError


class TestProgress(common_testing.BaseTestCase):
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

    def test_exception(self):
        hook = ExceptionHook()
        obj = planetmapper.BodyXY('jupiter', '2005-01-01', sz=5)
        obj._set_progress_hook(hook)
        with self.assertRaises(TestError):
            obj.get_backplane_img('EMISSION')
        obj._remove_progress_hook()
