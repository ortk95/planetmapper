import os
import unittest
from unittest.mock import MagicMock, patch

import common_testing
import matplotlib.backends.registry

import planetmapper
import planetmapper.gui
from planetmapper.gui import GUI


class TestFunctions(common_testing.BaseTestCase):
    def setUp(self) -> None:
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

    @patch('planetmapper.gui.GUI')
    def test_run_gui(self, mock_GUI: MagicMock):
        mock_gui_instance = MagicMock()
        mock_GUI.return_value = mock_gui_instance

        planetmapper.gui.run_gui()
        mock_GUI.assert_called_once_with()
        mock_gui_instance.run.assert_called_once_with()
        mock_gui_instance.set_observation.assert_not_called()

        mock_GUI.reset_mock()
        path = os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
        planetmapper.gui.run_gui(path)
        mock_GUI.assert_called_once_with()
        mock_gui_instance.run.assert_called_once_with()
        mock_gui_instance.set_observation.assert_called_once_with(
            planetmapper.Observation(path)
        )

    @patch('matplotlib.pyplot.switch_backend')
    @patch('matplotlib.backends.backend_registry.resolve_backend')
    @patch('matplotlib.get_backend')
    def test_maybe_switch_matplotlib_backend_to_tkagg(
        self,
        mock_get_backend: MagicMock,
        mock_resolve_backend: MagicMock,
        mock_switch_backend: MagicMock,
    ):
        registry = matplotlib.backends.registry.BackendRegistry()

        def resolve_backend(backend):
            try:
                return registry.resolve_backend(backend)
            except RuntimeError:
                if backend == 'inline':
                    # Treat inline as a headless backend if it isn't available on the
                    # current system.
                    return backend, None
                raise

        mock_resolve_backend.side_effect = resolve_backend

        gui_backends_to_test = [
            'tkagg',
            'TkAgg',
            'tkcairo',
            'gtk3agg',
            'qtagg',
            'wx',
            'macosx',
        ]
        headless_backends_to_test = [
            'agg',
            'cairo',
            'pdf',
            'svg',
            'inline',
        ]
        for backend in gui_backends_to_test + headless_backends_to_test:
            with self.subTest('Success', backend=backend):
                mock_get_backend.return_value = backend
                mock_switch_backend.reset_mock()
                planetmapper.gui._maybe_switch_matplotlib_backend_to_tkagg()
                if backend.lower() == 'tkagg' or backend in headless_backends_to_test:
                    mock_switch_backend.assert_not_called()
                else:
                    mock_switch_backend.assert_called_once_with('tkagg')

        mock_switch_backend.reset_mock()
        for backend in gui_backends_to_test + headless_backends_to_test:
            with self.subTest('Failure', backend=backend):
                mock_get_backend.return_value = backend
                mock_switch_backend.reset_mock()
                msg_to_raise = (
                    'Cannot load backend {!r} which requires the {!r} interactive '
                    'framework, as {!r} is currently running'.format(
                        'tkagg', '???', backend
                    )
                )
                mock_switch_backend.side_effect = ImportError(msg_to_raise)
                if backend.lower() == 'tkagg' or backend in headless_backends_to_test:
                    planetmapper.gui._maybe_switch_matplotlib_backend_to_tkagg()
                    mock_switch_backend.assert_not_called()
                else:
                    with self.assertRaises(ImportError) as cm:
                        planetmapper.gui._maybe_switch_matplotlib_backend_to_tkagg()
                    msg_caught = str(cm.exception)
                    self.assertIn(msg_caught, msg_caught)
                    self.assertIn(planetmapper.gui._BACKEND_ERROR_HELP_TEXT, msg_caught)
                    mock_switch_backend.assert_called_once_with('tkagg')


class TestGUI(common_testing.BaseTestCase):
    @patch('planetmapper.gui._maybe_switch_matplotlib_backend_to_tkagg')
    def test_init(self, mock_maybe_switch_backend: MagicMock):
        GUI()
        mock_maybe_switch_backend.assert_called_once()

        mock_maybe_switch_backend.reset_mock()
        GUI(check_matplotlib_backend=False)
        mock_maybe_switch_backend.assert_not_called()
