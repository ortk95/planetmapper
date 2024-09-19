import os
import unittest
from unittest.mock import MagicMock, patch

import common_testing

import planetmapper
import planetmapper.gui
from planetmapper.gui import GUI


class TestFunctions(common_testing.BaseTestCase):

    @patch('planetmapper.gui.GUI', autospec=True)
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


class TestGUI(common_testing.BaseTestCase):
    def test_init(self):
        GUI()
