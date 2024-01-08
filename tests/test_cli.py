import unittest
from unittest.mock import MagicMock, patch

import common_testing

import planetmapper
import planetmapper.cli


class TestCommon(unittest.TestCase):
    @patch('planetmapper.cli._run_gui')
    def test_main(self, mock_run_gui: MagicMock):
        planetmapper.cli.main([])
        mock_run_gui.assert_called_once_with(None)
        mock_run_gui.reset_mock()

        planetmapper.cli.main(['test.fits'])
        mock_run_gui.assert_called_once_with('test.fits')
        mock_run_gui.reset_mock()

        flags = [
            '-v',
            '--version',
            '-h',
            '--help',
        ]
        for flag in flags:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as cm:
                    planetmapper.cli.main([flag])
                self.assertEqual(cm.exception.code, 0)
                mock_run_gui.assert_not_called()
                mock_run_gui.reset_mock()

    def test_parser(self):
        parser = planetmapper.cli._get_parser()
        args = parser.parse_args([])
        self.assertEqual(args.file_path, None)

        args = parser.parse_args(['test.fits'])
        self.assertEqual(args.file_path, 'test.fits')

        good_flags = [
            '-v',
            '--version',
            '-h',
            '--help',
        ]
        for flag in good_flags:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args([flag])
                self.assertEqual(cm.exception.code, 0)

        bad_arguments: list[list[str]] = [
            ['test.fits', 'test2.fits'],
            ['-xyz'],
        ]
        for arguments in bad_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args(arguments)
                self.assertEqual(cm.exception.code, 2)

    @patch('planetmapper.gui._run_gui_from_cli')
    def test_run_gui(self, mock_run_gui_from_cli: MagicMock):
        planetmapper.cli._run_gui('test.fits')
        mock_run_gui_from_cli.assert_called_once_with('test.fits')

        mock_run_gui_from_cli.reset_mock()
        planetmapper.cli._run_gui(None)
        mock_run_gui_from_cli.assert_called_once_with(None)

    def test_get_version(self):
        self.assertEqual(planetmapper.cli._get_version(), planetmapper.__version__)
