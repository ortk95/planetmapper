import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import common_testing

import planetmapper


class TkinterImportRaiser:
    """
    Raise ModuleNotFoundError when trying to import any tkinter module if tkinter is not
    installed.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname.lower().startswith('tkinter'):
            # we get here if the module is not loaded and not in sys.modules
            raise ModuleNotFoundError('<MOCK MODULE NOT FOUND ERROR>', name=fullname)


class TestMockGUI(common_testing.BaseTestCase):
    def setUp(self) -> None:
        sys.meta_path.insert(0, TkinterImportRaiser())
        for k in tuple(sys.modules.keys()):
            if 'tkinter' in k.lower() and '_mock_gui_no_tkinter' not in k:
                try:
                    del sys.modules[k]
                except KeyError:
                    pass
            if 'planetmapper' in k:
                try:
                    del sys.modules[k]
                except KeyError:
                    pass

    def tearDown(self) -> None:
        sys.meta_path = [
            m for m in sys.meta_path if not isinstance(m, TkinterImportRaiser)
        ]
        for k in tuple(sys.modules.keys()):
            if 'tkinter' in k.lower() and '_mock_gui_no_tkinter' not in k:
                try:
                    del sys.modules[k]
                except KeyError:
                    pass
            if 'planetmapper' in k:
                try:
                    del sys.modules[k]
                except KeyError:
                    pass
        importlib.import_module('planetmapper')

    def test_import_planetmapper_without_tkinter(self):
        import planetmapper

        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

        with self.subTest('Raise ImportError'):

            def check_message(cm):
                self.assertIn(
                    'The "tkinter" package is not included in your Python installation',
                    str(cm.exception),
                )

            with self.assertRaises(ImportError) as cm:
                planetmapper.gui.GUI
            check_message(cm)

            with self.assertRaises(ImportError) as cm:
                planetmapper.run_gui()
            check_message(cm)

            with self.assertRaises(ImportError) as cm:
                planetmapper.gui.run_gui()
            check_message(cm)

            observation = planetmapper.Observation(
                os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
            )
            with self.assertRaises(ImportError) as cm:
                observation.run_gui()
            check_message(cm)

        with self.subTest('Should work as normal'):
            planetmapper.gui
            body = planetmapper.Body(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
            )
            self.assertEqual(body.target_body_id, 599)
            self.assertEqual(body.utc, '2005-01-01T00:00:00.000000')
            self.assertArraysClose(
                body.lonlat2radec(0, 90), (196.37390490466322, -5.561534444253404)
            )
