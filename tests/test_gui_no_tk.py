import importlib
import os
import sys
import unittest

import common_testing
import numpy as np

ERROR_MESSAGE = (
    'The "tkinter" package is not included in your Python installation, so PlanetMapper cannot create a graphical user interface. '
    'See https://docs.python.org/3/library/tkinter.html for more information.'
)


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


class TestMockGUIWithTkinterImportRaiser(common_testing.BaseTestCase):
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
            self.assertEqual(ERROR_MESSAGE, str(cm.exception))

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

        # Copied from test_body
        body = planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        self.assertEqual(body.target_body_id, 599)
        self.assertEqual(body.utc, '2005-01-01T00:00:00.000000')
        self.assertAlmostEqual(body.target_ra, 196.37198562427025, places=5)
        self.assertAlmostEqual(body.target_dec, -5.565793847134351, places=5)
        self.assertAlmostEqual(body.subpoint_lon, 153.12585514751467, places=5)
        self.assertAlmostEqual(body.subpoint_lat, -3.0886644594385193, places=5)
        self.assertArraysClose(
            body.radec2lonlat(0, 0), (np.nan, np.nan), equal_nan=True
        )
        self.assertArraysClose(
            body.radec2lonlat(196.37198562427025, -5.565793847134351),
            (153.1235185909613, -3.0887371238645795),
        )

    def test_init(self):
        import planetmapper
        import planetmapper._mock_gui_no_tk

        self.assertIsInstance(
            planetmapper.gui, planetmapper._mock_gui_no_tk._MockGUIModuleClass
        )
        with self.assertRaises(AttributeError):
            planetmapper._get_mocks


class TestMockGUIElements(common_testing.BaseTestCase):
    EXCEPTIONS_TO_RAISE_FROM = [
        ImportError('No module named tkinter', name='tkinter'),
        ModuleNotFoundError('No module named tkinter', name='tkinter'),
        ImportError('No module named _tkinter', name='_tkinter'),
        ModuleNotFoundError('No module named _tkinter', name='_tkinter'),
        ImportError(
            'No module named tkinter.some.sub.module',
            name='tkinter.some.sub.module',
        ),
        ModuleNotFoundError(
            'No module named tkinter.some.sub.module',
            name='tkinter.some.sub.module',
        ),
    ]

    EXCEPTIONS_TO_NOT_RAISE_FROM = [
        ImportError('No module named something_else', name='something_else'),
        ModuleNotFoundError('No module named something_else', name='something_else'),
        ImportError('No module named tkinter'),
        ModuleNotFoundError('No module named tkinter'),
        ImportError(),
        ModuleNotFoundError(),
        ValueError('Some other error'),
        KeyError(),
        AttributeError('No module named tkinter', name='tkinter'),
    ]

    def test_raise_tkinter_import_error(self):
        from planetmapper._mock_gui_no_tk import raise_tkinter_import_error

        for exc in self.EXCEPTIONS_TO_RAISE_FROM:
            with self.subTest('Should raise from', exc=exc):
                with self.assertRaises(ImportError) as cm:
                    raise_tkinter_import_error(exc)
                self.assertEqual(ERROR_MESSAGE, str(cm.exception))
                self.assertIsNot(cm.exception, exc)
                self.assertIs(cm.exception.__cause__, exc)

        for exc in self.EXCEPTIONS_TO_NOT_RAISE_FROM:
            with self.subTest('Should not raise from', exc=exc):
                with self.assertRaises(type(exc)) as cm:
                    raise_tkinter_import_error(exc)
                self.assertIs(cm.exception, exc)

    def test_mocks(self):
        from planetmapper._mock_gui_no_tk import get_mocks

        for exc in self.EXCEPTIONS_TO_RAISE_FROM:
            gui, run_gui = get_mocks(exc)
            with self.subTest('Should raise from', exc=exc, mock='module'):
                with self.assertRaises(ImportError) as cm:
                    gui.GUI
                self.assertEqual(ERROR_MESSAGE, str(cm.exception))
                self.assertIsNot(cm.exception, exc)
                self.assertIs(cm.exception.__cause__, exc)

            with self.subTest('Should raise from', exc=exc, mock='run_gui'):
                with self.assertRaises(ImportError) as cm:
                    run_gui()
                self.assertEqual(ERROR_MESSAGE, str(cm.exception))
                self.assertIsNot(cm.exception, exc)
                self.assertIs(cm.exception.__cause__, exc)

        for exc in self.EXCEPTIONS_TO_NOT_RAISE_FROM:
            gui, run_gui = get_mocks(exc)
            with self.subTest('Should not raise from', exc=exc, mock='module'):
                with self.assertRaises(type(exc)) as cm:
                    gui.GUI
                self.assertIs(cm.exception, exc)

            with self.subTest('Should not raise from', exc=exc, mock='run_gui'):
                with self.assertRaises(type(exc)) as cm:
                    run_gui()
                self.assertIs(cm.exception, exc)
