import unittest
from unittest.mock import MagicMock, patch

import common_testing


class TestMain(common_testing.BaseTestCase):
    @patch('planetmapper.cli.main')
    def test_main(self, mock_cli_main: MagicMock):
        import planetmapper.__main__

        mock_cli_main.assert_called_once()
