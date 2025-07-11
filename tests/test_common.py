import unittest

import common_testing

import planetmapper
import planetmapper.common


class TestCommon(common_testing.BaseTestCase):
    def test_init(self):
        self.assertEqual(planetmapper.common.__author__, 'Oliver King')
        self.assertEqual(
            planetmapper.common.__url__, 'https://github.com/ortk95/planetmapper'
        )
        self.assertEqual(planetmapper.common.__license__, 'MIT')
        self.assertIsInstance(planetmapper.common.__version__, str)
        self.assertIsInstance(planetmapper.common.__description__, str)

        self.assertIsInstance(planetmapper.common.CITATION_STRING, str)
        self.assertIsInstance(planetmapper.common.CITATION_DOI, str)
        self.assertIsInstance(planetmapper.common.CITATION_BIBTEX, str)

        # See also test_init
