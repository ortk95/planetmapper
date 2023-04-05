import datetime
import unittest

import common_testing

import planetmapper
from planetmapper import BasicBody


class TestBasicBody(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = BasicBody('Jupiter', observer='HST', utc='2005-01-01T00:00:00')

    def test_attributes(self):
        self.assertEqual(self.body.target, 'JUPITER')
        self.assertEqual(self.body.utc, '2005-01-01T00:00:00.000000')
        self.assertEqual(self.body.observer, 'HST')
        self.assertAlmostEqual(self.body.et, 157809664.1839331)
        self.assertEqual(
            self.body.dtm,
            datetime.datetime(2005, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        )
        self.assertEqual(self.body.target_body_id, 599)
        self.assertAlmostEqual(self.body.target_light_time, 2734.018326542542)
        self.assertAlmostEqual(self.body.target_distance, 819638074.3312353)
        self.assertAlmostEqual(self.body.target_ra, 196.37198562427025)
        self.assertAlmostEqual(self.body.target_dec, -5.565793847134351)

    def test_repr(self):
        self.assertEqual(
            repr(self.body), "BasicBody('JUPITER', '2005-01-01T00:00:00.000000')"
        )

    def test_eq(self):
        self.assertEqual(self.body, self.body)
        self.assertEqual(
            self.body, BasicBody('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        )
        self.assertNotEqual(
            self.body,
            planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00'),
        )
        self.assertNotEqual(
            self.body, BasicBody('Jupiter', observer='HST', utc='2005-01-01T00:00:01')
        )
        self.assertNotEqual(self.body, BasicBody('Jupiter', utc='2005-01-01T00:00:00'))
        self.assertNotEqual(
            self.body, BasicBody('amalthea', observer='HST', utc='2005-01-01T00:00:00')
        )
        self.assertNotEqual(
            self.body,
            BasicBody(
                'Jupiter',
                observer='HST',
                utc='2005-01-01T00:00:00',
                aberration_correction='CN+S',
            ),
        )
