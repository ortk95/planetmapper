import unittest
import planetmapper
import datetime
import numpy as np
import spiceypy as spice
from typing import Callable, ParamSpec, Any
import common_testing
import planetmapper.progress
import planetmapper.base

P = ParamSpec('P')


class TestBody(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = planetmapper.Body(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
        )

    def test_init(self):
        self.assertEqual(
            planetmapper.Body('Jupiter', utc='2005-01-01').subpoint_lon,
            153.12547767272153,
        )
        self.assertEqual(
            planetmapper.Body(
                'Jupiter', utc='2005-01-01', aberration_correction='CN+S'
            ).subpoint_lon,
            153.12614128206837,
        )

    def test_attributes(self):
        self.assertEqual(self.body.target, 'JUPITER')
        self.assertEqual(self.body.utc, '2005-01-01T00:00:00.000000')
        self.assertEqual(self.body.observer, 'HST')
        self.assertEqual(self.body.et, 157809664.1839331)
        self.assertEqual(
            self.body.dtm,
            datetime.datetime(2005, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        )
        self.assertEqual(self.body.target_body_id, 599)
        self.assertEqual(self.body.r_eq, 71492.0)
        self.assertEqual(self.body.r_polar, 66854.0)
        self.assertEqual(self.body.flattening, 0.0648743915403122)
        self.assertEqual(self.body.prograde, True)
        self.assertEqual(self.body.positive_longitude_direction, 'W')
        self.assertEqual(self.body.target_light_time, 2734.018326542542)
        self.assertEqual(self.body.target_distance, 819638074.3312353)
        self.assertEqual(self.body.target_ra, 196.37198562427025)
        self.assertEqual(self.body.target_dec, -5.565793847134351)
        self.assertEqual(self.body.target_diameter_arcsec, 35.98242703657337)
        self.assertEqual(self.body.subpoint_distance, 819566594.28005)
        self.assertEqual(self.body.subpoint_lon, 153.12585514751467)
        self.assertEqual(self.body.subpoint_lat, -3.0886644594385193)
        self.assertEqual(
            self.body.named_ring_data,
            {
                'Halo': [89400.0, 123000.0],
                'Main Ring': [123000.0, 128940.0],
                'Amalthea Ring': [128940.0, 181350.0],
                'Thebe Ring': [181350.0, 221900.0],
                'Thebe Extension': [221900.0, 280000.0],
            },
        )
        self.assertEqual(self.body.ring_radii, set())
        self.assertEqual(self.body.coordinates_of_interest_lonlat, [])
        self.assertEqual(self.body.coordinates_of_interest_radec, [])
        self.assertEqual(self.body.other_bodies_of_interest, [])

    def test_repr(self):
        self.assertEqual(
            repr(self.body), "Body('JUPITER', '2005-01-01T00:00:00.000000')"
        )

    def test_eq(self):
        self.assertEqual(self.body, self.body)
        self.assertEqual(
            self.body,  planetmapper.Body(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
            )
        )
        self.assertNotEqual(
            self.body , planetmapper.Body(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:01'
            )
        )
        self.assertNotEqual(
            self.body , planetmapper.Body(
                'Jupiter', utc='2005-01-01T00:00:00')
        )
        self.assertNotEqual(
            self.body , planetmapper.Body(
                'amalthea', observer='HST', utc='2005-01-01T00:00:00'
            ))
        self.assertNotEqual(
            self.body, planetmapper.Body(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:00',
                aberration_correction='CN+S')
        )

    def test_create_other_body(self):
        self.assertEqual(
            self.body.create_other_body('amalthea'),
                planetmapper.Body(
                    'AMALTHEA', observer='HST', utc='2005-01-01T00:00:00'
                )
        )

    def test_add_other_bodies_of_interest(self):
        self.body.add_other_bodies_of_interest('amalthea')
        self.assertEqual(
            self.body.other_bodies_of_interest,
            [
                planetmapper.Body(
                    'AMALTHEA', observer='HST', utc='2005-01-01T00:00:00'
                )
            ],
        )
        self.body.add_other_bodies_of_interest('METIS', 'thebe')
        self.assertEqual(
            self.body.other_bodies_of_interest,
            [
                planetmapper.Body(
                    'AMALTHEA', observer='HST', utc='2005-01-01T00:00:00'
                ),
                planetmapper.Body(
                    
                    'METIS', observer='HST', utc='2005-01-01T00:00:00'
                ),
                planetmapper.Body(
                    'THEBE', observer='HST', utc='2005-01-01T00:00:00'
                ),
            ],
        )
        self.body.other_bodies_of_interest.clear()
        self.assertEqual(self.body.other_bodies_of_interest, [])
    