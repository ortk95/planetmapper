import importlib
import os
import sys
import unittest

import common_testing


class TkinterImportRaiser:
    """
    Raise ModuleNotFoundError when trying to import any tkinter module if tkinter is not
    installed.
    """

    def find_spec(self, fullname, path, target=None):
        # _tkinter is the actual import that should fail if tk is not installed...
        if fullname.lower().startswith('_tkinter'):
            # we get here if the module is not loaded and not in sys.modules
            raise ModuleNotFoundError('<MOCK MODULE NOT FOUND ERROR>', name=fullname)


class TestMockGUI(common_testing.BaseTestCase):
    def setUp(self) -> None:
        self._deleted_modules = {}
        sys.meta_path.insert(0, TkinterImportRaiser())
        for k in tuple(sys.modules.keys()):
            if 'planetmapper' in k or 'tkinter' in k.lower():
                try:
                    self._deleted_modules[k] = sys.modules[k]
                    del sys.modules[k]
                except KeyError:
                    pass

    def tearDown(self) -> None:
        sys.meta_path = [
            m for m in sys.meta_path if not isinstance(m, TkinterImportRaiser)
        ]
        for k, v in self._deleted_modules.items():
            sys.modules[k] = v
        import planetmapper

        importlib.reload(planetmapper)

    def test_raise_error(self):
        import planetmapper

        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

        def check_message(cm):
            self.assertIn(
                'The "tkinter" package is not included in your Python installation',
                str(cm.exception),
            )

        with self.subTest('planetmapper.gui.GUI'):
            with self.assertRaises(ImportError) as cm:
                planetmapper.gui.GUI
            check_message(cm)

        with self.subTest('planetmapper.run_gui()'):
            with self.assertRaises(ImportError) as cm:
                planetmapper.run_gui()
            check_message(cm)

        with self.subTest('planetmapper.gui.run_gui()'):
            with self.assertRaises(ImportError) as cm:
                planetmapper.gui.run_gui()
        check_message(cm)

        with self.subTest('observation.run_gui()'):
            observation = planetmapper.Observation(
                os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
            )
            with self.assertRaises(ImportError) as cm:
                observation.run_gui()
            check_message(cm)

    def test_work_as_normal(self):
        import planetmapper

        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        planetmapper.gui
        body = planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        self.assertEqual(body.target_body_id, 599)
        self.assertEqual(body.utc, '2005-01-01T00:00:00.000000')
        self.assertArraysClose(
            body.lonlat2radec(0, 90), (196.37390490466322, -5.561534444253404)
        )
