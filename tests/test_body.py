import copy
import datetime
import sys
import unittest
from typing import Callable
from unittest.mock import MagicMock, patch

import common_testing
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from numpy import array, nan
from spiceypy.utils.exceptions import (
    NotFoundError,
    SpiceBODIESNOTDISTINCT,
    SpiceKERNELVARNOTFOUND,
    SpiceSPKINSUFFDATA,
)

import planetmapper
import planetmapper.base
import planetmapper.progress
import planetmapper.utils
from planetmapper import Body


class TestBody(common_testing.BaseTestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')

    def test_get_default_init_kwargs(self):
        self._test_get_default_init_kwargs(
            Body, target='Jupiter', utc='2005-01-01T00:00:00'
        )

    def test_init(self):
        self.assertAlmostEqual(
            Body('Jupiter', utc='2005-01-01').subpoint_lon,
            153.12547767272153,
        )
        self.assertEqual(
            Body(
                'Jupiter', utc='2005-01-01', aberration_correction='CN+S'
            ).subpoint_lon,
            153.12614128206837,
        )

        jupiter_with_frame = Body(
            'Jupiter', utc='2005-01-01', target_frame='iau_jupiter'
        )
        self.assertAlmostEqual(jupiter_with_frame.subpoint_lon, 153.12547767272153)
        self.assertEqual(jupiter_with_frame.target_frame, 'iau_jupiter')

        # Test Saturrn automatically has A, B & C rings added
        saturn = Body('saturn', '2000-01-01')
        self.assertEqual(saturn.target, 'SATURN')
        self.assertEqual(saturn.target_body_id, 699)
        self.assertEqual(
            saturn.ring_radii, {74658.0, 91975.0, 117507.0, 122340.0, 136780.0}
        )

        # Test SpiceBODIESNOTDISTINCT is raised appropriately, without any divide by
        # zero errors occuring first
        with self.assertRaises(SpiceBODIESNOTDISTINCT):
            planetmapper.Body('earth', observer='earth', utc='2005-01-01')

    def test_kernel_errors(self):
        try:
            Body(
                target='mars',
                utc='2000-01-01',
                observer='earth',
                aberration_correction='CN+S',
                observer_frame='J2000',
            )
        except SpiceSPKINSUFFDATA as e:
            self.assertIn(planetmapper.base._SPICE_ERROR_HELP_TEXT, e.message)
            self.assertIn(planetmapper.base.get_kernel_path(), e.message)

            # Ensure help message isn't added twice from nested decorated functions
            self.assertEquals(
                e.message.count(planetmapper.base._SPICE_ERROR_HELP_TEXT), 1
            )

    def test_rotation_sense(self):
        comparisons: list[tuple[str, str, bool]] = [
            ('sun', 'E', True),
            ('moon', 'E', True),
            ('earth', 'E', True),
            ('jupiter', 'W', True),
            ('amalthea', 'W', True),
            ('uranus', 'E', False),
        ]
        for target, positive_dir, prograde in comparisons:
            with self.subTest(target=target):
                body = planetmapper.Body(
                    target,
                    observer='HST',
                    utc='2005-01-01T00:00:00',
                )
                self.assertEqual(body.positive_longitude_direction, positive_dir)
                self.assertEqual(body.prograde, prograde)

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
        self.assertEqual(self.body.r_eq, 71492.0)
        self.assertEqual(self.body.r_polar, 66854.0)
        self.assertAlmostEqual(self.body.flattening, 0.0648743915403122)
        self.assertEqual(self.body.prograde, True)
        self.assertEqual(self.body.positive_longitude_direction, 'W')
        self.assertAlmostEqual(self.body.target_light_time, 2734.018326542542)
        self.assertAlmostEqual(self.body.target_distance, 819638074.3312353)
        self.assertAlmostEqual(self.body.target_ra, 196.37198562427025)
        self.assertAlmostEqual(self.body.target_dec, -5.565793847134351)
        self.assertAlmostEqual(self.body.target_diameter_arcsec, 35.98242689969618)
        self.assertAlmostEqual(self.body.km_per_arcsec, 3973.7175149019004)
        self.assertAlmostEqual(self.body.subpoint_distance, 819566594.28005)
        self.assertAlmostEqual(self.body.subpoint_lon, 153.12585514751467)
        self.assertAlmostEqual(self.body.subpoint_lat, -3.0886644594385193)
        self.assertAlmostEqual(self.body.subsol_lon, 163.44768812575543)
        self.assertAlmostEqual(self.body.subsol_lat, -2.7185371707509427)
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

        self.assertEqual(self.body._alt_adjustment, 0.0)

        moon = Body('moon', '2005-01-01')
        self.assertEqual(moon.positive_longitude_direction, 'E')
        self.assertTrue(moon.prograde)

        sun = Body('sun', '2005-01-01')
        self.assertEqual(sun.positive_longitude_direction, 'E')
        self.assertTrue(sun.prograde)
        self.assertTrue(np.isnan(sun.subsol_lon))
        self.assertTrue(np.isnan(sun.subsol_lat))

        # Check types are actually floats and not e.g. np.float64
        self.assertIs(type(self.body.flattening), float)
        self.assertIs(type(self.body.km_per_arcsec), float)
        self.assertIs(type(self.body.r_eq), float)
        self.assertIs(type(self.body.r_polar), float)

    def test_repr(self):
        self.assertEqual(
            repr(self.body),
            "Body('JUPITER', '2005-01-01T00:00:00.000000', observer='HST')",
        )
        self.assertEqual(
            repr(
                Body(
                    'Jupiter',
                    observer='HST',
                    utc='2005-01-01T00:00:00',
                    show_progress=True,
                    aberration_correction='CN+S',
                    optimize_speed=False,
                    auto_load_kernels=True,
                )
            ),
            "Body('JUPITER', '2005-01-01T00:00:00.000000', observer='HST', aberration_correction='CN+S', show_progress=True, optimize_speed=False)",
        )

    def test_eq(self):
        self.assertEqual(self.body, self.body)
        self.assertEqual(
            self.body, Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        )
        self.assertNotEqual(
            self.body,
            planetmapper.BasicBody(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
            ),
        )
        self.assertNotEqual(
            self.body,
            planetmapper.BodyXY('Jupiter', observer='HST', utc='2005-01-01T00:00:00'),
        )
        self.assertNotEqual(
            self.body, Body('Jupiter', observer='HST', utc='2005-01-01T00:00:01')
        )
        self.assertNotEqual(self.body, Body('Jupiter', utc='2005-01-01T00:00:00'))
        self.assertNotEqual(
            self.body, Body('amalthea', observer='HST', utc='2005-01-01T00:00:00')
        )
        self.assertNotEqual(
            self.body,
            Body(
                'Jupiter',
                observer='HST',
                utc='2005-01-01T00:00:00',
                aberration_correction='CN+S',
            ),
        )

    def test_hash(self):
        self.assertEqual(
            hash(self.body),
            hash(Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')),
        )
        times = [
            '2005-01-01T00:00:00',
            '2005-01-01T00:00:00',
            '2005-01-01T00:00:00',
            '2005-01-01T00:00:01',
            '2005-01-01T00:00:02',
        ]
        d = {}
        for time in times:
            d[Body('Jupiter', observer='HST', utc=time)] = time
        self.assertEqual(len(d), 3)

    def test_get_kwargs(self):
        self.assertEqual(
            self.body._get_kwargs(),
            {
                'optimize_speed': True,
                'show_progress': False,
                'auto_load_kernels': True,
                'kernel_path': None,
                'manual_kernels': None,
                'target': 'JUPITER',
                'target_frame': None,
                'utc': '2005-01-01T00:00:00.000000',
                'observer': 'HST',
                'aberration_correction': 'CN',
                'observer_frame': 'J2000',
                'illumination_source': 'SUN',
                'subpoint_method': 'INTERCEPT/ELLIPSOID',
                'surface_method': 'ELLIPSOID',
            },
        )

    def test_copy(self):
        body = Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        body.add_other_bodies_of_interest('amalthea')
        body.coordinates_of_interest_lonlat.append((0, 0))
        body.coordinates_of_interest_radec.extend([(1, 2), (3, 4)])
        body.add_named_rings()

        new = body.copy()
        self.assertEqual(body, new)
        self.assertIsNot(body, new)
        self.assertEqual(body._get_kwargs(), new._get_kwargs())
        self._test_if_body_has_same_options(body, new)

        new.coordinates_of_interest_lonlat.append((5, 6))
        self.assertNotEqual(
            body.coordinates_of_interest_lonlat, new.coordinates_of_interest_lonlat
        )

        with self.subTest('copy.copy'):
            self.assertEqual(body.copy(), copy.copy(body))

    def test_replace(self):
        body = Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        body.add_other_bodies_of_interest('amalthea')
        body.coordinates_of_interest_lonlat.append((0, 0))
        body.coordinates_of_interest_radec.extend([(1, 2), (3, 4)])
        body.add_named_rings()

        with self.subTest('no changes'):
            new = body.replace()
            self.assertEqual(body, new)
            self.assertIsNot(body, new)
            self.assertEqual(body._get_kwargs(), new._get_kwargs())
            self._test_if_body_has_same_options(body, new)

            new.coordinates_of_interest_lonlat.append((5, 6))
            self.assertNotEqual(
                body.coordinates_of_interest_lonlat, new.coordinates_of_interest_lonlat
            )

        with self.subTest('change utc'):
            utc = '2005-01-01T12:34:56'
            new = body.replace(utc=utc)
            self.assertNotEqual(body, new)
            self.assertEqual(new.utc, '2005-01-01T12:34:56.000000')
            self._test_if_body_has_same_options(body, new)

        with self.subTest('change multiple'):
            utc = '2005-01-01T12:34:56'
            observer = 'earth'
            new = body.replace(utc=utc, observer=observer)
            self.assertNotEqual(body, new)
            self.assertEqual(new.utc, '2005-01-01T12:34:56.000000')
            self.assertEqual(new.observer, 'EARTH')
            self._test_if_body_has_same_options(body, new)

        with self.subTest('round trip'):
            new = body.replace(utc='2005-01-01T00:00:00', observer='HST')
            self.assertEqual(body, new)

            new = body.replace(observer='earth', utc='2005-01-01T12:34:56').replace(
                utc='2005-01-01T00:00:00', observer='HST'
            )
            self.assertEqual(body, new)

        if sys.version_info >= (3, 13):
            with self.subTest('copy.replace'):
                self.assertEqual(
                    body.replace(observer='earth', utc='2005-01-01T12:34:56'),
                    copy.replace(body, observer='earth', utc='2005-01-01T12:34:56'),
                )

    def _test_if_body_has_same_options(self, original: Body, new: Body) -> None:
        with self.subTest(original=original, new=new):
            self.assertEqual(
                original.other_bodies_of_interest, new.other_bodies_of_interest
            )
            self.assertEqual(
                original.coordinates_of_interest_lonlat,
                new.coordinates_of_interest_lonlat,
            )
            self.assertEqual(
                original.coordinates_of_interest_radec,
                new.coordinates_of_interest_radec,
            )
            self.assertEqual(original.ring_radii, new.ring_radii)

    def test_create_other_body(self):
        self.assertEqual(
            self.body.create_other_body('amalthea'),
            Body('AMALTHEA', observer='HST', utc='2005-01-01T00:00:00'),
        )
        self.assertEqual(
            self.body.create_other_body('daphnis'),
            planetmapper.BasicBody(
                'DAPHNIS', observer='HST', utc='2005-01-01T00:00:00'
            ),
        )
        with self.assertRaises(SpiceKERNELVARNOTFOUND):
            self.body.create_other_body('daphnis', fallback_to_basic_body=False)

        target = '<<< test >>>'
        with self.assertRaises(NotFoundError):
            self.body.create_other_body(target)
        try:
            self.body.create_other_body(target)
        except NotFoundError as e:
            self.assertIn(target, e.message)

    def test_add_other_bodies_of_interest(self):
        self.body.add_other_bodies_of_interest('amalthea')
        self.assertEqual(
            self.body.other_bodies_of_interest,
            [Body('AMALTHEA', observer='HST', utc='2005-01-01T00:00:00')],
        )
        self.body.add_other_bodies_of_interest('METIS', 'thebe')
        self.assertEqual(
            self.body.other_bodies_of_interest,
            [
                Body('AMALTHEA', observer='HST', utc='2005-01-01T00:00:00'),
                Body('METIS', observer='HST', utc='2005-01-01T00:00:00'),
                Body('THEBE', observer='HST', utc='2005-01-01T00:00:00'),
            ],
        )
        self.body.other_bodies_of_interest.clear()
        self.assertEqual(self.body.other_bodies_of_interest, [])

        utc = '2005-01-01 04:00:00'
        jupiter = planetmapper.Body('Jupiter', utc)
        jupiter.add_other_bodies_of_interest('THEBE', only_visible=True)
        self.assertEqual(jupiter.other_bodies_of_interest, [])

        jupiter.add_other_bodies_of_interest('AMALTHEA', 'THEBE', only_visible=True)
        self.assertEqual(jupiter.other_bodies_of_interest, [Body('AMALTHEA', utc)])

        jupiter.other_bodies_of_interest.clear()
        self.assertEqual(jupiter.other_bodies_of_interest, [])

        jupiter.add_other_bodies_of_interest('AMALTHEA', 'THEBE')
        self.assertEqual(
            jupiter.other_bodies_of_interest,
            [Body('AMALTHEA', utc), Body('THEBE', utc)],
        )
        # Check duplicate bodies are not added
        jupiter.add_other_bodies_of_interest('AMALTHEA', 'THEBE')
        self.assertEqual(
            jupiter.other_bodies_of_interest,
            [Body('AMALTHEA', utc), Body('THEBE', utc)],
        )

    def test_add_satellites_to_bodies_of_interest(self):
        self.body.other_bodies_of_interest.clear()
        with self.assertRaises(SpiceSPKINSUFFDATA):
            self.body.add_satellites_to_bodies_of_interest()

        expected = [
            Body('AMALTHEA', '2005-01-01T00:00:00.000000', 'HST'),
            Body('THEBE', '2005-01-01T00:00:00.000000', 'HST'),
            Body('ADRASTEA', '2005-01-01T00:00:00.000000', 'HST'),
            Body('METIS', '2005-01-01T00:00:00.000000', 'HST'),
        ]
        self.body.other_bodies_of_interest.clear()
        self.body.add_satellites_to_bodies_of_interest(skip_insufficient_data=True)
        self.assertEqual(self.body.other_bodies_of_interest, expected)

        # Test duplicates aren't added
        self.body.add_satellites_to_bodies_of_interest(skip_insufficient_data=True)
        self.assertEqual(self.body.other_bodies_of_interest, expected)
        self.body.other_bodies_of_interest.clear()
        self.assertEqual(self.body.other_bodies_of_interest, [])
        self.body.add_other_bodies_of_interest('amalthea', 'thebe')
        self.assertEqual(self.body.other_bodies_of_interest, expected[:2])
        self.body.add_satellites_to_bodies_of_interest(skip_insufficient_data=True)
        self.assertEqual(self.body.other_bodies_of_interest, expected)
        self.body.other_bodies_of_interest.clear()
        self.assertEqual(self.body.other_bodies_of_interest, [])

        utc = '2005-01-01 04:00:00'
        jupiter = planetmapper.Body('Jupiter', utc)
        jupiter.add_satellites_to_bodies_of_interest(
            skip_insufficient_data=True, only_visible=True
        )
        self.assertEqual(
            jupiter.other_bodies_of_interest,
            [Body('AMALTHEA', utc), Body('ADRASTEA', utc), Body('METIS', utc)],
        )
        jupiter.other_bodies_of_interest.clear()
        self.assertEqual(jupiter.other_bodies_of_interest, [])

        jupiter.add_satellites_to_bodies_of_interest(skip_insufficient_data=True)
        self.assertEqual(
            jupiter.other_bodies_of_interest,
            [
                Body('AMALTHEA', utc),
                Body('THEBE', utc),
                Body('ADRASTEA', utc),
                Body('METIS', utc),
            ],
        )
        jupiter.other_bodies_of_interest.clear()
        self.assertEqual(jupiter.other_bodies_of_interest, [])

        # This checks the `except NotFoundError` branch works properly
        earth = planetmapper.Body('earth', utc, observer='HST')
        earth.add_satellites_to_bodies_of_interest()
        self.assertEqual(
            earth.other_bodies_of_interest,
            [
                Body('MOON', utc, observer='HST'),
            ],
        )

    def test_standardise_ring_name(self):
        pairs = [
            ('a', 'a'),
            ('A', 'a'),
            ('  a  ', 'a'),
            (' c  RiNg ', 'c'),
            ('liberte', 'liberté'),
            ('égalité', 'egalité'),
            ('égalité', 'egalité'),
            (' FrAternitE ring ', 'fraternité'),
        ]
        for name, expected in pairs:
            with self.subTest(name=name):
                self.assertEqual(self.body._standardise_ring_name(name), expected)

    def test_ring_radii_from_name(self):
        self.assertEqual(self.body.ring_radii_from_name('Halo'), [89400.0, 123000.0])
        self.assertEqual(
            self.body.ring_radii_from_name('   MaIn rinG         '),
            [123000.0, 128940.0],
        )
        self.assertEqual(self.body.ring_radii_from_name('main'), [123000.0, 128940.0])
        with self.assertRaises(ValueError):
            self.body.ring_radii_from_name('spam')

    def test_add_named_rings(self):
        self.body.ring_radii.clear()
        self.assertEqual(self.body.ring_radii, set())

        self.body.add_named_rings('halo', '   MaIn rinG ')
        self.assertEqual(self.body.ring_radii, {89400.0, 123000.0, 128940.0})

        self.body.add_named_rings('thebe extension')
        self.assertEqual(
            self.body.ring_radii, {89400.0, 123000.0, 128940.0, 221900.0, 280000.0}
        )

        with self.assertRaises(ValueError):
            self.body.add_named_rings('<<<< test ring name >>>>')

        self.body.ring_radii.clear()
        self.assertEqual(self.body.ring_radii, set())

        self.body.add_named_rings()
        self.assertEqual(
            self.body.ring_radii,
            {280000.0, 181350.0, 128940.0, 221900.0, 89400.0, 123000.0},
        )

        self.body.ring_radii.clear()
        self.assertEqual(self.body.ring_radii, set())

    def test_lonlat2obsvec(self):
        # (((lon, lat), alt, not_visible_nan), expected)
        to_test: list[tuple[tuple[tuple[float, float], float, bool], np.ndarray]] = [
            (
                ((196, -5), 0.0, False),
                array([-7.82631523e08, -2.29970069e08, -7.95130215e07]),
            ),
            (
                ((0, 90), 0.0, False),
                array([-7.82696778e08, -2.29972887e08, -7.94353094e07]),
            ),
            (
                ((0, -90), 0.0, False),
                array([-7.82694825e08, -2.29915349e08, -7.95559886e07]),
            ),
            (
                ((0, 0), 0.0, False),
                array([-7.82766264e08, -2.29932797e08, -7.94913913e07]),
            ),
            (
                ((196, -5), -123.456, False),
                array([-7.82631634e08, -2.29970025e08, -7.95129903e07]),
            ),
            (
                ((0, 90), -123.456, False),
                array([-7.82696776e08, -2.29972834e08, -7.94354209e07]),
            ),
            (
                ((0, -90), -123.456, False),
                array([-7.82694827e08, -2.29915403e08, -7.95558772e07]),
            ),
            (
                ((0, 0), -123.456, False),
                array([-7.82766143e08, -2.29932816e08, -7.94913987e07]),
            ),
            (
                ((196, -5), 100, False),
                array([-7.82631433e08, -2.29970105e08, -7.95130468e07]),
            ),
            (
                ((0, 90), 100, False),
                array([-7.82696780e08, -2.29972930e08, -7.94352192e07]),
            ),
            (
                ((0, -90), 100, False),
                array([-7.82694824e08, -2.29915306e08, -7.95560789e07]),
            ),
            (
                ((0, 0), 100, False),
                array([-7.82766363e08, -2.29932781e08, -7.94913853e07]),
            ),
            (
                ((196, -5), 42424242, False),
                array([-7.44175722e08, -2.44544072e08, -8.99360751e07]),
            ),
            (
                ((0, 90), 42424242, False),
                array([-7.83316420e08, -2.48228896e08, -4.11449650e07]),
            ),
            (
                ((0, -90), 42424242, False),
                array([-7.82075211e08, -2.11659261e08, -1.17846296e08]),
            ),
            (
                ((0, 0), 42424242, False),
                array([-8.24409499e08, -2.22373962e08, -7.65613347e07]),
            ),
            (
                ((196, -5), 0.0, True),
                array([-7.82631523e08, -2.29970069e08, -7.95130215e07]),
            ),
            (((0, 90), 0.0, True), array([nan, nan, nan])),
            (
                ((0, -90), 0.0, True),
                array([-7.82694825e08, -2.29915349e08, -7.95559886e07]),
            ),
            (((0, 0), 0.0, True), array([nan, nan, nan])),
            (((196, -5), -123.456, True), array([nan, nan, nan])),
            (((0, 90), -123.456, True), array([nan, nan, nan])),
            (((0, -90), -123.456, True), array([nan, nan, nan])),
            (((0, 0), -123.456, True), array([nan, nan, nan])),
            (
                ((196, -5), 100, True),
                array([-7.82631433e08, -2.29970105e08, -7.95130468e07]),
            ),
            (
                ((0, 90), 100, True),
                array([-7.82696780e08, -2.29972930e08, -7.94352192e07]),
            ),
            (
                ((0, -90), 100, True),
                array([-7.82694824e08, -2.29915306e08, -7.95560789e07]),
            ),
            (((0, 0), 100, True), array([nan, nan, nan])),
            (
                ((196, -5), 42424242, True),
                array([-7.44175722e08, -2.44544072e08, -8.99360751e07]),
            ),
            (
                ((0, 90), 42424242, True),
                array([-7.83316420e08, -2.48228896e08, -4.11449650e07]),
            ),
            (
                ((0, -90), 42424242, True),
                array([-7.82075211e08, -2.11659261e08, -1.17846296e08]),
            ),
            (
                ((0, 0), 42424242, True),
                array([-8.24409499e08, -2.22373962e08, -7.65613347e07]),
            ),
        ]
        for ((lon, lat), alt, not_visible_nan), expected in to_test:
            with self.subTest(
                lon=lon, lat=lat, alt=alt, not_visible_nan=not_visible_nan
            ):
                self.assertArraysClose(
                    self.body._lonlat2obsvec(
                        lon, lat, alt=alt, not_visible_nan=not_visible_nan
                    ),
                    expected,
                    equal_nan=True,
                )

    def test_lonlat2radec(self):
        pairs = [
            [(0, 90), (196.37390490466322, -5.561534444253404)],
            [(0, 0), (196.36982789576643, -5.565060944053696)],
            [(123.456, -56.789), (196.3691609381441, -5.5685956879058764)],
            [(np.nan, np.nan), (np.nan, np.nan)],
            [(np.nan, 0), (np.nan, np.nan)],
            [(0, np.nan), (np.nan, np.nan)],
            [(np.inf, np.inf), (np.nan, np.nan)],
        ]
        for lonlat, radec in pairs:
            with self.subTest(lonlat):
                self.assertTrue(
                    np.allclose(self.body.lonlat2radec(*lonlat), radec, equal_nan=True)
                )

        pairs_with_alts: list[
            tuple[tuple[float, float, float], tuple[float, float]]
        ] = [
            ((42, 23.4, 0), (196.36871162182828, -5.5624995718895915)),
            ((42, 23.4, -123.456), (196.36871704240835, -5.562505596011716)),
            ((42, 23.4, 1234.567), (196.3686574157507, -5.562439330354751)),
            ((42, 23.4, nan), (nan, nan)),
        ]
        for (lon, lat, alt), expected in pairs_with_alts:
            with self.subTest((lon, lat, alt)):
                self.assertArraysClose(
                    self.body.lonlat2radec(lon, lat, alt=alt),
                    expected,
                    equal_nan=True,
                )
        # Test array broadcasting implementation
        self.assertArraysClose(
            self.body.lonlat2radec(
                np.array([0, 90, 123]),
                np.array([1, 2, 3]),
                alt=123.456,
                not_visible_nan=True,
            ),
            (
                array([nan, 196.36800057, 196.3698629]),
                array([nan, -5.56373086, -5.56437196]),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.lonlat2radec(
                [[0, 90, 123], [100, 200, 300.123]],
                0,
                alt=123.456,
                not_visible_nan=True,
            ),
            (
                array(
                    [
                        [nan, 196.36793564, 196.36976609],
                        [196.36837244, 196.37540197, nan],
                    ]
                ),
                array(
                    [[nan, -5.56386914, -5.56457942], [-5.56402583, -5.56714199, nan]]
                ),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.lonlat2radec(lon=123, lat=-12.34),
            (196.3694301738864, -5.5654598621335625),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.lonlat2radec([[[[123]]]], -12.34),
            (array([[[[196.36943017]]]]), array([[[[-5.56545986]]]])),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.lonlat2radec([np.nan, np.inf, 0, 1.234, 1234], 0),
            (
                array([nan, nan, 196.3698279, 196.36974134, 196.37215256]),
                array([nan, nan, -5.56506094, -5.56501974, -5.56561021]),
            ),
            equal_nan=True,
        )

    def test_radec2lonlat(self):
        self.assertTrue(
            np.array_equal(
                self.body.radec2lonlat(0, 0), (np.nan, np.nan), equal_nan=True
            )
        )
        with self.assertRaises(NotFoundError):
            self.body.radec2lonlat(0, 0, not_found_nan=False)

        pairs = [
            [
                (196.37198562427025, -5.565793847134351),
                (153.1235185909613, -3.0887371238645795),
            ],
            [(196.372, -5.566), (154.24480750302573, -5.475831082435726)],
            [
                (196.3742715121965, -5.561743939677709),
                (180.00086055026196, 80.00042229835671),
            ],
            [(np.nan, np.nan), (np.nan, np.nan)],
            [(np.nan, 0), (np.nan, np.nan)],
            [(0, np.nan), (np.nan, np.nan)],
            [(np.inf, np.inf), (np.nan, np.nan)],
        ]
        for radec, lonlat in pairs:
            with self.subTest(radec):
                self.assertTrue(
                    np.allclose(
                        self.body.radec2lonlat(*radec),
                        lonlat,
                        equal_nan=True,
                    )
                )
                if all(np.isfinite(x) for x in radec):
                    self.assertTrue(
                        np.allclose(
                            self.body.lonlat2radec(*lonlat), radec, equal_nan=True
                        )
                    )

        pairs_with_alts: list[
            tuple[tuple[float, float, float], tuple[float, float]]
        ] = [
            (
                (196.37198562427025, -5.565793847134351, 0),
                (153.1235185909613, -3.0887371238645795),
            ),
            (
                (196.37198562427025, -5.565793847134351, 123456.789),
                (153.12766781084477, -2.834663828028037),
            ),
            (
                (196.37198562427025, -5.565793847134351, -1000),
                (153.12348498172653, -3.0948138787454225),
            ),
            ((nan, -5, 123), (nan, nan)),
        ]
        for (ra, dec, alt), expected in pairs_with_alts:
            with self.subTest((ra, dec, alt)):
                self.assertArraysClose(
                    self.body.radec2lonlat(ra, dec, alt=alt), expected, equal_nan=True
                )

    def test_lonlat2targvec(self):
        pairs = [
            ((0, 0), np.array([71492.0, -0.0, 0.0])),
            ((360, 0), np.array([[71492.0, -0.0, 0.0]])),
            ((123, 45), np.array([-28439.90450754, -43793.6125254, 45662.45633365])),
            (
                (-80, -12.3456789),
                np.array([[12162.32647743, 68975.98103572, -13405.21131042]]),
            ),
            [(np.nan, np.nan), np.array([np.nan, np.nan, np.nan])],
            [(np.nan, 0), np.array([np.nan, np.nan, np.nan])],
            [(0, np.nan), np.array([np.nan, np.nan, np.nan])],
            [(np.inf, np.inf), np.array([np.nan, np.nan, np.nan])],
        ]
        for (lon, lat), targvec in pairs:
            with self.subTest((lon, lat)):
                self.assertTrue(
                    np.allclose(
                        self.body.lonlat2targvec(lon, lat), targvec, equal_nan=True
                    )
                )

        pairs_with_alts: list[tuple[tuple[float, float, float], np.ndarray]] = [
            ((42, 23.4, 0), array([49249.33355035, -44344.29910771, 25077.9757777])),
            (
                (42, 23.4, -123.456),
                array([49165.13352119, -44268.48506093, 25028.94548771]),
            ),
            (
                (42, 23.4, 1234.567),
                array([50091.3386161, -45102.44387423, 25568.2814576]),
            ),
            ((42, 23.4, nan), array([nan, nan, nan])),
        ]
        for (lon, lat, alt), expected in pairs_with_alts:
            with self.subTest((lon, lat, alt)):
                self.assertArraysClose(
                    self.body.lonlat2targvec(lon, lat, alt=alt),
                    expected,
                    equal_nan=True,
                )

    def test_targvec2lonlat(self):
        pairs = [
            (np.array([0, 0, 0]), (0.0, 90.0)),
            (np.array([1, 2, 3]), (296.565051177078, 89.98665551067639)),
            (np.array([-9876, 543210, 0]), (268.9584308375042, 0.0)),
            (np.array([np.nan, np.nan, np.nan]), (np.nan, np.nan)),
            (np.array([np.nan, 0, 0]), (np.nan, np.nan)),
            (np.array([0, np.nan, 0]), (np.nan, np.nan)),
            (np.array([0, 0, np.nan]), (np.nan, np.nan)),
            (np.array([np.inf, np.inf, np.inf]), (np.nan, np.nan)),
        ]
        for targvec, lonlat in pairs:
            with self.subTest(targvec):
                self.assertTrue(
                    np.allclose(
                        self.body.targvec2lonlat(targvec), lonlat, equal_nan=True
                    )
                )
        pairs_with_alts: list[tuple[tuple[np.ndarray, float], tuple[float, float]]] = [
            ((array([0, 0, 0]), 0), (0.0, 90.0)),
            ((array([1, 2, 3]), 0), (296.565051177078, 89.98665551067639)),
            ((array([-9876, 543210, 0]), 0), (268.9584308375042, 0.0)),
            ((array([0, 0, 0]), -123.45), (0.0, 90.0)),
            ((array([1, 2, 3]), -123.45), (296.565051177078, 89.98665633798927)),
            ((array([-9876, 543210, 0]), -123.45), (268.9584308375042, 0.0)),
            ((array([0, 0, 0]), 987654321), (0.0, 90.0)),
            ((array([1, 2, 3]), 987654321), (296.565051177078, 89.98619280529013)),
            ((array([-9876, 543210, 0]), 987654321), (268.9584308375042, 0.0)),
            ((array([-9876, 543210, nan]), 987654321), (nan, nan)),
        ]
        for (targvec, alt), lonlat in pairs_with_alts:
            with self.subTest((targvec, alt)):
                self.assertArraysClose(
                    self.body.targvec2lonlat(targvec, alt=alt), lonlat, equal_nan=True
                )

    def test_angular_radec(self):
        pairs: list[tuple[tuple[float, float], dict, tuple[float, float]]] = [
            ((0, 0), {}, (196.37198562131056, -5.565793839734843)),
            (
                (0, 0),
                {'coordinate_rotation': 123},
                (196.37198562131056, -5.565793839734843),
            ),
            ((1.234, 5.678), {}, (196.37164122076928, -5.564216617412704)),
            ((-3600.1234, 45678), {}, (197.35518558863563, 7.1233716685998285)),
            (
                (1.234, 5.678),
                {'coordinate_rotation': 123},
                (196.3708441579451, -5.566940333059796),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': 123},
                (122.99965559945868, -5.564216624812211),
            ),
            (
                (1.234, 5.678),
                {'origin_dec': 12.3},
                (196.37163479126497, 12.301577221998656),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': -12.3},
                (236.99964917120613, -12.298422777554215),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': 12.3, 'coordinate_rotation': -123},
                (237.001544919471, 12.299428456509167),
            ),
        ]
        for (x, y), kw, radec in pairs:
            with self.subTest(x=x, y=y, kw=kw):
                self.assertArraysClose(self.body.angular2radec(x, y, **kw), radec)
                self.assertArraysClose(
                    self.body.radec2angular(*radec, **kw), (x, y), atol=1e-4
                )

        # test for lack of changes
        ra, dec = self.body.target_ra, self.body.target_dec
        kwargs: list[dict] = [
            dict(),
            dict(coordinate_rotation=0),
            dict(origin_ra=ra),
            dict(origin_dec=dec),
            dict(origin_ra=ra, origin_dec=dec),
            dict(coordinate_rotation=0, origin_ra=ra, origin_dec=dec),
            dict(coordinate_rotation=0, origin_ra=None, origin_dec=None),
        ]
        x, y = 2.34, -5.67
        radec_expected = self.body.angular2radec(x, y)
        for kw in kwargs:
            with self.subTest(kw=kw):
                radec = self.body.angular2radec(x, y, **kw)
                self.assertArraysClose(radec, radec_expected)

                xy = self.body.radec2angular(*radec_expected, **kw)
                self.assertArraysClose(xy, (x, y))

    def test_angular_lonlat(self):
        pairs = [
            ((0, 0), {}, (153.12351859061235, -3.0887371240013572)),
            (
                (0, 0),
                {'coordinate_rotation': 123},
                (153.12351859061235, -3.0887371240013572),
            ),
            ((1.234, 5.678), {}, (141.76181779277195, 14.187903497915688)),
            ((-3600.1234, 45678), {}, (nan, nan)),
            (
                (1.234, 5.678),
                {'coordinate_rotation': 123},
                (146.10317442767905, -23.08048248991215),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': 196.372, 'origin_dec': -5.566},
                (143.01960641488623, 11.717675615612585),
            ),
            (
                (1.234, 0.678),
                {
                    'origin_ra': 196.372,
                    'origin_dec': -5.566,
                    'coordinate_rotation': -123,
                },
                (156.98171972231182, -1.4107148298315533),
            ),
        ]
        for (x, y), kw, lonlat in pairs:
            with self.subTest(x=x, y=y, kw=kw):
                self.assertArraysClose(
                    self.body.angular2lonlat(x, y, **kw),
                    lonlat,
                    equal_nan=True,
                    atol=1e-3,
                )
                if np.isfinite(lonlat[0]):
                    self.assertArraysClose(
                        self.body.lonlat2angular(*lonlat, **kw), (x, y), atol=1e-4
                    )
                    self.assertArraysClose(
                        self.body.angular2lonlat(x, y, **kw, not_found_nan=False),
                        lonlat,
                        equal_nan=True,
                    )

                else:
                    with self.assertRaises(NotFoundError):
                        self.body.angular2lonlat(x, y, **kw, not_found_nan=False)

        inputs = [
            (np.nan, np.nan),
            (np.nan, 0),
            (0, np.nan),
            (np.inf, np.inf),
        ]
        for a in inputs:
            with self.subTest(a):
                self.assertTrue(
                    all(not np.isfinite(x) for x in self.body.angular2lonlat(*a))
                )
                self.assertTrue(
                    all(not np.isfinite(x) for x in self.body.lonlat2angular(*a))
                )

        pairs_with_alts: list[
            tuple[tuple[float, float, float], tuple[float, float]]
        ] = [
            ((42, 23.4, 0), (11.730907264131929, 11.859358373960486)),
            ((42, 23.4, -123.456), (11.711484946681594, 11.837671641863478)),
            ((42, 23.4, 1234.567), (11.92513145342673, 12.076226814058423)),
            ((42, 23.4, nan), (nan, nan)),
        ]
        for (lon, lat, alt), expected in pairs_with_alts:
            with self.subTest((lon, lat, alt)):
                self.assertArraysClose(
                    self.body.lonlat2angular(lon, lat, alt=alt),
                    expected,
                    equal_nan=True,
                )
        pairs_with_alts = [
            (
                (11.730907264131929, 11.859358373960486, 0),
                (86.30139500952406, 21.109249946237032),
            ),
            (
                (11.730907264131929, 11.859358373960486, 123456.789),
                (134.58218536012419, 4.708273802335033),
            ),
            (
                (11.730907264131929, 11.859358373960486, -1000),
                (83.89699519490205, 21.59807910857171),
            ),
            ((nan, 11, 123), (nan, nan)),
        ]
        for (x, y, alt), expected in pairs_with_alts:
            with self.subTest((x, y, alt)):
                self.assertArraysClose(
                    self.body.angular2lonlat(x, y, alt=alt), expected, equal_nan=True
                )

    def test_km_rotation(self):
        x_target, y_target = self.body.radec2km(
            self.body.target_ra, self.body.target_dec
        )
        self.assertAlmostEqual(x_target, 0)
        self.assertAlmostEqual(y_target, 0)
        for lat in [-90, 90]:
            with self.subTest(lat):
                x, y = self.body.lonlat2km(0, lat)
                self.assertAlmostEqual(x, x_target, delta=1)
                if lat > 0:
                    self.assertGreater(y, y_target)
                else:
                    self.assertLess(y, y_target)

    def test_km_radec(self):
        pairs = [
            ((0, 0), (196.3719856242702, -5.56579384713435)),
            ((99999, 99999), (196.36845127590436, -5.556555100442686)),
            ((1234, -5678), (196.37174335301282, -5.566120708196197)),
            ((-0.1234, 9999.5678), (196.37227302705824, -5.565156047930656)),
        ]
        for km, radec in pairs:
            with self.subTest(km):
                self.assertArraysClose(self.body.km2radec(*km), radec)
                self.assertArraysClose(self.body.radec2km(*radec), km, atol=1e-3)

        inputs = [
            (np.nan, np.nan),
            (np.nan, 0),
            (0, np.nan),
            (np.inf, np.inf),
        ]
        for a in inputs:
            with self.subTest(a):
                self.assertTrue(all(not np.isfinite(x) for x in self.body.km2radec(*a)))
                self.assertTrue(all(not np.isfinite(x) for x in self.body.radec2km(*a)))

    def test_km_lonlat(self):
        pairs = [
            ((0, 0), (153.12351859061235, -3.0887371240013572)),
            ((123, 456.789), (153.02485721448028, -2.6703253305682195)),
            ((-500, -200), (153.52477375354786, -3.2718421646109985)),
            ((5000, 50001), (147.39408652731262, 47.4410279733397)),
        ]
        for km, lonlat in pairs:
            with self.subTest(km):
                self.assertTrue(np.allclose(self.body.km2lonlat(*km), lonlat))
                self.assertTrue(
                    np.allclose(self.body.lonlat2km(*lonlat), km, atol=1e-3)
                )

        self.assertTrue(
            np.array_equal(
                self.body.km2lonlat(100000000, 0), (np.nan, np.nan), equal_nan=True
            )
        )

        inputs = [
            (np.nan, np.nan),
            (np.nan, 0),
            (0, np.nan),
            (np.inf, np.inf),
        ]
        for a in inputs:
            with self.subTest(a):
                self.assertTrue(
                    all(not np.isfinite(x) for x in self.body.km2lonlat(*a))
                )
                self.assertTrue(
                    all(not np.isfinite(x) for x in self.body.lonlat2km(*a))
                )

        pairs_with_alts: list[
            tuple[tuple[float, float, float], tuple[float, float]]
        ] = [
            ((42, 23.4, 0), (61817.98981185463, 23924.02130814744)),
            ((42, 23.4, -123.456), (61712.30433782919, 23876.972243949425)),
            ((42, 23.4, 1234.567), (62874.850051314235, 24394.514357706197)),
            ((42, 23.4, nan), (nan, nan)),
        ]
        for (lon, lat, alt), expected in pairs_with_alts:
            with self.subTest((lon, lat, alt)):
                self.assertArraysClose(
                    self.body.lonlat2km(lon, lat, alt=alt),
                    expected,
                    equal_nan=True,
                )
        pairs_with_alts = [
            (
                (61817.98981185463, 23924.02130814744, 0),
                (86.30139500952406, 21.109249946237032),
            ),
            (
                (61817.98981185463, 23924.02130814744, 123456.789),
                (134.58218536012419, 4.708273802335033),
            ),
            (
                (61817.98981185463, 23924.02130814744, -1000),
                (83.89699519490205, 21.59807910857171),
            ),
            ((nan, 23924, 123), (nan, nan)),
        ]
        for (x, y, alt), expected in pairs_with_alts:
            with self.subTest((x, y, alt)):
                self.assertArraysClose(
                    self.body.km2lonlat(x, y, alt=alt), expected, equal_nan=True
                )

    def test_km_angular(self):
        pairs: list[tuple[tuple[float, float], dict, tuple[float, float]]] = [
            ((0, 0), {}, (0.0, 0.0)),
            ((0, 0), {'coordinate_rotation': 123}, (0.0, 0.0)),
            ((1.234, 5.678), {}, (13707.106875939699, 18580.59989529313)),
            ((-3600.1234, 45678), {}, (61222909.71285939, 171472523.56580824)),
            (
                (1.234, 5.678),
                {'coordinate_rotation': 123},
                (8117.576807789242, -21615.467104869596),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': 123},
                (928803175.7862874, -478472263.2296324),
            ),
            (
                (1.234, 5.678),
                {'origin_dec': 12.3},
                (104598412.22915992, 233217325.082532),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': -12.3},
                (-569001780.3607075, 128938234.54185842),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': 12.3, 'coordinate_rotation': -123},
                (-446038232.73474604, 458652497.8006319),
            ),
        ]
        for (x, y), kw, km in pairs:
            with self.subTest(x=x, y=y, kw=kw):
                self.assertArraysClose(self.body.angular2km(x, y, **kw), km, atol=1e-3)
                self.assertArraysClose(
                    self.body.km2angular(*km, **kw), (x, y), atol=1e-3
                )

    def test_limbradec(self):
        self.assertTrue(
            np.allclose(
                self.body.limb_radec(npts=10),
                (
                    np.array(
                        [
                            196.37390736,
                            196.37615012,
                            196.37694412,
                            196.37568283,
                            196.37297113,
                            196.37006385,
                            196.36782109,
                            196.36702713,
                            196.36828846,
                            196.37100013,
                            196.37390736,
                        ]
                    ),
                    np.array(
                        [
                            -5.56152901,
                            -5.56341574,
                            -5.56632605,
                            -5.56912521,
                            -5.57047072,
                            -5.57005866,
                            -5.56817191,
                            -5.56526158,
                            -5.56246245,
                            -5.56111695,
                            -5.56152901,
                        ]
                    ),
                ),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.limb_radec(npts=3, close_loop=False),
                (
                    array([196.37390736, 196.37487476, 196.36707757]),
                    array([-5.56152901, -5.56977427, -5.56629386]),
                ),
            )
        )

    def test_limb_lonlat(self):
        self.assertArraysClose(
            self.body.limb_lonlat(npts=5),
            (
                array(
                    [
                        153.1234683,
                        242.11517437,
                        247.35606526,
                        58.89081584,
                        64.1317418,
                        153.1234683,
                    ]
                ),
                array(
                    [
                        87.29379713,
                        20.35346551,
                        -57.46299289,
                        -57.46299289,
                        20.35346551,
                        87.29379713,
                    ]
                ),
            ),
        )

    def test_limb_radec_by_illumination(self):
        self.assertTrue(
            np.allclose(
                self.body.limb_radec_by_illumination(npts=5),
                (
                    array(
                        [
                            196.37390736,
                            196.37694412,
                            196.37297113,
                            nan,
                            nan,
                            196.37390736,
                        ]
                    ),
                    array(
                        [-5.56152901, -5.56632605, -5.57047072, nan, nan, -5.56152901]
                    ),
                    array([nan, nan, nan, 196.36782109, 196.36828846, nan]),
                    array([nan, nan, nan, -5.56817191, -5.56246245, nan]),
                ),
                equal_nan=True,
            )
        )

    def test_limb_coordinates_from_radec(self):
        args: list[tuple[tuple[float, float], tuple[float, float, float]]] = [
            ((0, 0), (82.72145635455739, -7.331180721378409, 243226446.365406)),
            (
                (196.3719829300016, -5.565779946690757),
                (67.23274105785333, 58.34599234749429, -68089.8880967631),
            ),
            (
                (196.372, -5.566),
                (248.13985326986065, -64.83923990338549, -64857.80811442864),
            ),
            ((196.3, -5.5), (64.1290135632679, 20.79992677586983, 1320579.9259661217)),
            ((np.nan, np.nan), (np.nan, np.nan, np.nan)),
            ((np.nan, 0), (np.nan, np.nan, np.nan)),
            ((0, np.nan), (np.nan, np.nan, np.nan)),
            ((np.inf, np.inf), (np.nan, np.nan, np.nan)),
        ]
        for (ra, dec), (lon_expected, lat_expected, dist_expected) in args:
            with self.subTest(ra=ra, dec=dec):
                lon, lat, dist = self.body.limb_coordinates_from_radec(ra, dec)
                self.assertTrue(
                    np.allclose(lon, lon_expected, rtol=1e-5, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(lat, lat_expected, rtol=1e-5, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(dist, dist_expected, rtol=1e-5, equal_nan=True)
                )

    def test_if_lonlat_visible(self):
        pairs: list[tuple[tuple[float, float], bool]] = [
            ((0, 0), False),
            ((180, 12), True),
            ((50, -80), True),
            ((np.nan, np.nan), False),
            ((np.nan, 0), False),
            ((0, np.nan), False),
            ((np.inf, np.inf), False),
        ]
        for lonlat, visible in pairs:
            with self.subTest(lonlat=lonlat):
                self.assertEqual(
                    self.body.test_if_lonlat_visible(*lonlat), visible, lonlat
                )

        pairs_with_alts: list[tuple[tuple[float, float, float], bool]] = [
            ((0, 0, 0), False),
            ((0, 0, 1000000.0), True),
            ((0, 0, -1000000.0), True),
            ((153.1, -3.0, 0), True),
            ((153.1, -3.0, -1), False),
            ((153.1, -3.0, 1), True),
            ((153.1, -3.0, nan), False),
            ((153.1, nan, 1), False),
        ]
        for (lon, lat, alt), visible in pairs_with_alts:
            with self.subTest(lon=lon, lat=lat, alt=alt):
                self.assertEqual(
                    self.body.test_if_lonlat_visible(lon, lat, alt=alt), visible
                )

    def test_other_body_los_intercept(self):
        utc = '2005-01-01 04:00:00'
        jupiter = planetmapper.Body('Jupiter', utc)

        intercepts: list[tuple[str, str | None, bool]] = [
            ('thebe', 'hidden', False),
            ('metis', 'transit', True),
            ('amalthea', None, True),
            ('adrastea', None, True),
            ('jupiter', 'same', True),
        ]

        for moon, intercept, visible in intercepts:
            body = jupiter.create_other_body(moon)
            arguments = [
                moon,
                body,
                body.target_body_id,
                planetmapper.BasicBody(moon, utc),
            ]
            for arg in arguments:
                with self.subTest(moon=moon, arg=arg):
                    self.assertEqual(jupiter.other_body_los_intercept(arg), intercept)
                    self.assertEqual(
                        jupiter.test_if_other_body_visible(arg),
                        visible,
                    )

        body = planetmapper.Body('Jupiter', '2005-01-01 00:35:24')
        self.assertEqual(body.other_body_los_intercept('amalthea'), 'part hidden')
        self.assertEqual(body.test_if_other_body_visible('amalthea'), True)

        body = planetmapper.Body('Jupiter', '2005-01-01 06:34:05')
        self.assertEqual(body.other_body_los_intercept('amalthea'), 'part transit')
        self.assertEqual(body.test_if_other_body_visible('amalthea'), True)

    def test_illimination_angles_from_lonlat(self):
        args: list[tuple[tuple[float, float], tuple[float, float, float]]] = [
            ((0, 0), (10.31594976458697, 163.2795134457034, 152.99822832991876)),
            (
                (123.456, -78.9),
                (10.316968817304499, 79.16351827229181, 77.68583738495468),
            ),
            ((np.nan, np.nan), (np.nan, np.nan, np.nan)),
            ((np.nan, 0), (np.nan, np.nan, np.nan)),
            ((0, np.nan), (np.nan, np.nan, np.nan)),
            ((np.inf, np.inf), (np.nan, np.nan, np.nan)),
        ]

        for lonlat, angles in args:
            with self.subTest(lonlat=lonlat):
                self.assertTrue(
                    np.allclose(
                        self.body.illumination_angles_from_lonlat(*lonlat),
                        angles,
                        equal_nan=True,
                    )
                )

    def test_azimuth_angle_from_lonlat(self):
        args: list[tuple[tuple[float, float], float]] = [
            ((0, 0), 177.66817822757469),
            ((123.456, -78.9), 169.57651996164563),
            ((np.nan, np.nan), np.nan),
            ((np.nan, 0), np.nan),
            ((0, np.nan), np.nan),
            ((np.inf, np.inf), np.nan),
        ]
        for lonlat, angle in args:
            with self.subTest(lonlat=lonlat):
                self.assertTrue(
                    np.allclose(
                        self.body.azimuth_angle_from_lonlat(*lonlat),
                        angle,
                        equal_nan=True,
                    )
                )

    def test_local_solar_time(self):
        args: list[tuple[float, float, str]] = [
            (0, 22.89638888888889, '22:53:47'),
            (-90, 4.896388888888889, '04:53:47'),
            (123.456, 14.666111111111112, '14:39:58'),
            (999.999, 4.229722222222223, '04:13:47'),
            (np.nan, np.nan, ''),
            (np.inf, np.nan, ''),
        ]
        for lon, lst_expected, s_expected in args:
            with self.subTest(lon=lon):
                lst = self.body.local_solar_time_from_lon(lon)
                s = self.body.local_solar_time_string_from_lon(lon)
                self.assertTrue(np.isclose(lst, lst_expected, equal_nan=True))
                self.assertEqual(s, s_expected)

    def test_terminator_radec(self):
        self.assertTrue(
            np.allclose(
                self.body.terminator_radec(npts=5),
                (
                    array([nan, nan, nan, 196.36784184, 196.36838618, nan]),
                    array([nan, nan, nan, -5.56815505, -5.56246241, nan]),
                ),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.terminator_radec(npts=3, close_loop=False),
                (array([nan, nan, 196.36713568]), array([nan, nan, -5.56628042])),
                equal_nan=True,
            )
        )

    def test_terminator_lonlat(self):
        self.assertArraysClose(
            self.body.terminator_lonlat(npts=5),
            (
                array(
                    [
                        163.44532164,
                        252.60875833,
                        257.26193719,
                        69.62871003,
                        74.2818866,
                        163.44532164,
                    ]
                ),
                array(
                    [
                        87.66650962,
                        20.36259847,
                        -57.48337047,
                        -57.48337047,
                        20.36259847,
                        87.66650962,
                    ]
                ),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.terminator_lonlat(npts=5, only_visible=True),
            (
                array([nan, nan, nan, 69.62871003, 74.2818866, nan]),
                array([nan, nan, nan, -57.48337047, 20.36259847, nan]),
            ),
            equal_nan=True,
        )

    def test_if_lonlat_illuminated(self):
        pairs: list[tuple[tuple[float, float], bool]] = [
            ((0, 0), False),
            ((180, 12), True),
            ((50, -80), False),
            ((np.nan, np.nan), False),
            ((np.nan, 0), False),
            ((0, np.nan), False),
            ((np.inf, np.inf), False),
        ]
        for (lon, lat), illuminated in pairs:
            with self.subTest(lon=lon, lat=lat):
                self.assertEqual(
                    self.body.test_if_lonlat_illuminated(lon, lat), illuminated
                )

    def test_ring_plane_coordinates(self):
        args: list[tuple[tuple[float, float, bool], tuple[float, float, float]]] = [
            ((0, 0, True), (nan, nan, nan)),
            ((196.37198562427025, -5.565793847134351, True), (nan, nan, nan)),
            (
                (196.37347182693253, -5.561472466522512, True),
                (1377914.753652832, 152.91772706249577, 818261707.8278764),
            ),
            ((196.3696997398314, -5.569843641306982, True), (nan, nan, nan)),
            (
                (196.37198562427025, -5.565793847134351, False),
                (4638.105239104683, 156.0690984698183, 819638074.3312378),
            ),
            (
                (196.3, -5.5, True),
                (9305877.091704229, 145.3644753085151, 810435703.2382222),
            ),
            ((np.nan, np.nan, True), (np.nan, np.nan, np.nan)),
            ((np.nan, 0, True), (np.nan, np.nan, np.nan)),
            ((0, np.nan, True), (np.nan, np.nan, np.nan)),
            ((np.inf, np.inf, True), (np.nan, np.nan, np.nan)),
        ]
        for (lon, lat, only_visible), coords in args:
            with self.subTest(lon=lon, lat=lat, only_visible=only_visible):
                self.assertTrue(
                    np.allclose(
                        self.body.ring_plane_coordinates(
                            lon, lat, only_visible=only_visible
                        ),
                        coords,
                        equal_nan=True,
                    ),
                )

        # test default args
        self.assertTrue(
            np.allclose(
                self.body.ring_plane_coordinates(196.3, -5.5),
                (9305877.091704229, 145.3644753085151, 810435703.2382222),
                equal_nan=True,
            ),
        )

    def test_ring_radec(self):
        # inside jupiter
        self.assertArraysClose(
            self.body.ring_radec(10000, npts=5),
            (
                array([nan, nan, nan, nan, nan]),
                array([nan, nan, nan, nan, nan]),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.ring_radec(100000, npts=5),
            (
                array([nan, 196.36633034, 196.37500382, 196.37764017, nan]),
                array([nan, -5.56310623, -5.56681892, -5.56848105, nan]),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.ring_radec(123456.789, npts=3, only_visible=False),
            (
                array([196.36825958, 196.37571178, 196.36825958]),
                array([-5.56452821, -5.56705935, -5.56452821]),
            ),
            equal_nan=True,
        )
        self.assertArraysClose(
            self.body.ring_radec(np.nan, npts=2, only_visible=False),
            (array([nan, nan]), array([nan, nan])),
            equal_nan=True,
        )

        with planetmapper.body._AdjustedSurfaceAltitude(self.body, 20000):
            # check that the only effect from surface alt is on visibility
            self.assertArraysClose(
                self.body.ring_radec(123456.789, npts=3, only_visible=False),
                (
                    array([196.36825958, 196.37571178, 196.36825958]),
                    array([-5.56452821, -5.56705935, -5.56452821]),
                ),
                equal_nan=True,
            )
            self.assertArraysClose(
                self.body.ring_radec(100000, npts=5),
                (
                    array([nan, 196.36633034, 196.37500382, nan, nan]),
                    array([nan, -5.56310623, -5.56681892, nan, nan]),
                ),
                equal_nan=True,
            )
            self.assertArraysClose(
                self.body.ring_radec(80000, npts=2),
                (array([nan, nan]), array([nan, nan])),
                equal_nan=True,
            )

    def test_visible_lonlat_grid_radec(self):
        self.assertTrue(
            np.allclose(
                self.body.visible_lonlat_grid_radec(interval=45, npts=5),
                [
                    (
                        array([196.3700663, nan, nan, nan, nan]),
                        array([-5.57005326, nan, nan, nan, nan]),
                    ),
                    (
                        array([196.3700663, nan, nan, nan, nan]),
                        array([-5.57005326, nan, nan, nan, nan]),
                    ),
                    (
                        array(
                            [196.3700663, 196.36772166, 196.36794262, 196.37034361, nan]
                        ),
                        array(
                            [-5.57005326, -5.56729981, -5.56387245, -5.56148116, nan]
                        ),
                    ),
                    (
                        array(
                            [196.3700663, 196.36970087, 196.37065239, 196.37232288, nan]
                        ),
                        array(
                            [-5.57005326, -5.56808941, -5.56495336, -5.56227057, nan]
                        ),
                    ),
                    (
                        array(
                            [196.3700663, 196.37225066, 196.37414339, 196.37487263, nan]
                        ),
                        array([-5.57005326, -5.56923855, -5.5665267, -5.56341971, nan]),
                    ),
                    (
                        array(
                            [196.3700663, 196.37387716, 196.37637019, 196.37649901, nan]
                        ),
                        array(
                            [-5.57005326, -5.57007398, -5.56767064, -5.56425534, nan]
                        ),
                    ),
                    (
                        array([196.3700663, nan, nan, nan, nan]),
                        array([-5.57005326, nan, nan, nan, nan]),
                    ),
                    (
                        array([196.3700663, nan, nan, nan, nan]),
                        array([-5.57005326, nan, nan, nan, nan]),
                    ),
                    (
                        array(
                            [
                                196.3700663,
                                196.3700663,
                                196.3700663,
                                196.3700663,
                                196.3700663,
                            ]
                        ),
                        array(
                            [
                                -5.57005326,
                                -5.57005326,
                                -5.57005326,
                                -5.57005326,
                                -5.57005326,
                            ]
                        ),
                    ),
                    (
                        array([nan, 196.36772166, 196.37225066, nan, nan]),
                        array([nan, -5.56729981, -5.56923855, nan, nan]),
                    ),
                    (
                        array([nan, 196.36794262, 196.37414339, nan, nan]),
                        array([nan, -5.56387245, -5.5665267, nan, nan]),
                    ),
                    (
                        array([nan, 196.37034361, 196.37487263, nan, nan]),
                        array([nan, -5.56148116, -5.56341971, nan, nan]),
                    ),
                ],
                equal_nan=True,
            )
        )

        self.assertTrue(
            np.allclose(
                self.body.visible_lonlat_grid_radec(lat_limit=30, npts=5),
                [
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array(
                            [
                                196.36751616,
                                196.36759734,
                                196.36794262,
                                196.36853038,
                                196.36933989,
                            ]
                        ),
                        array(
                            [
                                -5.56612675,
                                -5.56496537,
                                -5.56387245,
                                -5.56289478,
                                -5.56207953,
                            ]
                        ),
                    ),
                    (
                        array(
                            [
                                196.368942,
                                196.36916913,
                                196.36956301,
                                196.37010219,
                                196.37076577,
                            ]
                        ),
                        array(
                            [
                                -5.56667572,
                                -5.56557049,
                                -5.56449624,
                                -5.56349983,
                                -5.56262837,
                            ]
                        ),
                    ),
                    (
                        array(
                            [
                                196.37093916,
                                196.37137073,
                                196.37183267,
                                196.3723038,
                                196.37276294,
                            ]
                        ),
                        array(
                            [
                                -5.56753064,
                                -5.5665129,
                                -5.56546778,
                                -5.56444221,
                                -5.56348323,
                            ]
                        ),
                    ),
                    (
                        array(
                            [
                                196.37297245,
                                196.37361216,
                                196.37414339,
                                196.3745452,
                                196.3747962,
                            ]
                        ),
                        array(
                            [
                                -5.56846241,
                                -5.56754006,
                                -5.5665267,
                                -5.56546938,
                                -5.56441503,
                            ]
                        ),
                    ),
                    (
                        array(
                            [
                                196.37449692,
                                196.37529266,
                                196.37587583,
                                196.37622567,
                                196.37632061,
                            ]
                        ),
                        array(
                            [
                                -5.56922131,
                                -5.56837668,
                                -5.5673892,
                                -5.56630605,
                                -5.56517403,
                            ]
                        ),
                    ),
                    (
                        array(
                            [
                                196.37510403,
                                196.37596188,
                                196.37656571,
                                196.37689485,
                                196.37692764,
                            ]
                        ),
                        array(
                            [
                                -5.56960397,
                                -5.56879854,
                                -5.56782415,
                                -5.566728,
                                -5.56555684,
                            ]
                        ),
                    ),
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array([nan, nan, nan, nan, nan]),
                        array([nan, nan, nan, nan, nan]),
                    ),
                    (
                        array([nan, 196.36751616, 196.37297245, nan, nan]),
                        array([nan, -5.56612675, -5.56846241, nan, nan]),
                    ),
                    (
                        array([nan, 196.36794262, 196.37414339, nan, nan]),
                        array([nan, -5.56387245, -5.5665267, nan, nan]),
                    ),
                    (
                        array([nan, 196.36933989, 196.3747962, nan, nan]),
                        array([nan, -5.56207953, -5.56441503, nan, nan]),
                    ),
                ],
                equal_nan=True,
            )
        )

        self.assertArraysClose(
            self.body.visible_lonlat_grid_radec(
                lat_limit=60, npts=5, planetocentric=True
            ),
            [
                (array([nan, nan, nan, nan, nan]), array([nan, nan, nan, nan, nan])),
                (array([nan, nan, nan, nan, nan]), array([nan, nan, nan, nan, nan])),
                (array([nan, nan, nan, nan, nan]), array([nan, nan, nan, nan, nan])),
                (array([nan, nan, nan, nan, nan]), array([nan, nan, nan, nan, nan])),
                (
                    array(
                        [196.37247263, 196.37487449, 196.37656571, 196.37689104, nan]
                    ),
                    array([-5.57050647, -5.56975909, -5.56782415, -5.56528375, nan]),
                ),
                (
                    array(
                        [
                            196.37214481,
                            196.37428742,
                            196.37587583,
                            196.37630406,
                            196.37552262,
                        ]
                    ),
                    array(
                        [-5.5702999, -5.56938907, -5.5673892, -5.56491357, -5.56280363]
                    ),
                ),
                (
                    array(
                        [
                            196.37132177,
                            196.37281328,
                            196.37414339,
                            196.37482999,
                            196.37469964,
                        ]
                    ),
                    array(
                        [-5.5698902, -5.56865523, -5.5665267, -5.56417963, -5.56239384]
                    ),
                ),
                (
                    array(
                        [
                            196.37022404,
                            196.37084713,
                            196.37183267,
                            196.37286387,
                            196.37360195,
                        ]
                    ),
                    array(
                        [-5.56938717, -5.56775423, -5.56546778, -5.5632786, -5.56189078]
                    ),
                ),
                (
                    array(
                        [
                            196.3691458,
                            196.36891591,
                            196.36956301,
                            196.37093265,
                            196.3725237,
                        ]
                    ),
                    array(
                        [
                            -5.56892559,
                            -5.56692753,
                            -5.56449624,
                            -5.56245197,
                            -5.56142927,
                        ]
                    ),
                ),
                (
                    array(
                        [
                            196.36837598,
                            196.36753714,
                            196.36794262,
                            196.36955384,
                            196.37175384,
                        ]
                    ),
                    array(
                        [
                            -5.56862917,
                            -5.56639668,
                            -5.56387245,
                            -5.56192125,
                            -5.56113297,
                        ]
                    ),
                ),
                (
                    array([196.36812084, nan, nan, nan, nan]),
                    array([-5.56857731, nan, nan, nan, nan]),
                ),
                (array([nan, nan, nan, nan, nan]), array([nan, nan, nan, nan, nan])),
                (
                    array([nan, nan, 196.37132177, 196.36837598, nan]),
                    array([nan, nan, -5.5698902, -5.56862917, nan]),
                ),
                (
                    array([nan, nan, 196.37281328, 196.36753714, nan]),
                    array([nan, nan, -5.56865523, -5.56639668, nan]),
                ),
                (
                    array([nan, nan, 196.37414339, 196.36794262, nan]),
                    array([nan, nan, -5.5665267, -5.56387245, nan]),
                ),
                (
                    array([nan, nan, 196.37482999, 196.36955384, nan]),
                    array([nan, nan, -5.56417963, -5.56192125, nan]),
                ),
                (
                    array([nan, nan, 196.37469964, 196.37175384, nan]),
                    array([nan, nan, -5.56239384, -5.56113297, nan]),
                ),
            ],
            equal_nan=True,
        )

    def test_radial_velocity_from_lonlat(self):
        args: list[tuple[tuple[float, float], float]] = [
            ((0, 0), -20.796924908179438),
            ((np.nan, np.nan), np.nan),
            ((np.nan, 0), np.nan),
            ((0, np.nan), np.nan),
            ((np.inf, np.inf), np.nan),
        ]
        for lonlat, x in args:
            with self.subTest(lonlat=lonlat):
                self.assertTrue(
                    np.allclose(
                        self.body.radial_velocity_from_lonlat(*lonlat),
                        x,
                        equal_nan=True,
                    )
                )

    def test_distance_from_lonalt(self):
        args: list[tuple[tuple[float, float], float]] = [
            ((0, 0), 819701772.0279644),
            ((np.nan, np.nan), np.nan),
            ((np.nan, 0), np.nan),
            ((0, np.nan), np.nan),
            ((np.inf, np.inf), np.nan),
        ]
        for lonlat, x in args:
            with self.subTest(lonlat=lonlat):
                self.assertTrue(
                    np.allclose(
                        self.body.distance_from_lonlat(*lonlat), x, equal_nan=True
                    )
                )

    def test_graphic_centric_lonlat(self):
        pairs = [
            [(0, 0), (0, 0)],
            [(0, 90), (0, 90)],
            [(0, -90), (0, -90)],
            [(90, 0), (-90, 0)],
            [(123.4, 56.789), (-123.4, 53.17999536010973)],
            [
                (array([1.0, 2.0, 3.0, nan]), array([40.0, 50.0, 60.0, nan])),
                (
                    array([-1.0, -2.0, -3.0, nan]),
                    array([36.26969371, 46.18216311, 56.56575448, nan]),
                ),
            ],
        ]
        for graphic, centric in pairs:
            with self.subTest(graphic):
                self.assertArraysClose(
                    self.body.graphic2centric_lonlat(*graphic),
                    centric,
                    equal_nan=True,
                )
                self.assertArraysClose(
                    self.body.centric2graphic_lonlat(*centric),
                    graphic,
                    equal_nan=True,
                )

        pairs = [
            ((np.nan, np.nan), (np.nan, np.nan)),
            ((np.nan, 0), (np.nan, np.nan)),
            ((0, np.nan), (np.nan, np.nan)),
            ((np.inf, np.inf), (np.nan, np.nan)),
        ]
        for a, b in pairs:
            with self.subTest(a):
                self.assertTrue(
                    np.allclose(self.body.graphic2centric_lonlat(*a), b, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(self.body.centric2graphic_lonlat(*a), b, equal_nan=True)
                )

    def test_north_pole_angle(self):
        # test angle < 0 branch
        self.assertAlmostEqual(self.body.north_pole_angle(), -24.15516987997688)

        # test angle > 0 branch
        body2 = planetmapper.Body('Jupiter', observer='HST', utc='2009-01-01T00:00:00')
        self.assertAlmostEqual(body2.north_pole_angle(), 13.550583134129457)

    def test_get_description(self):
        self.assertEqual(
            self.body.get_description(),
            'JUPITER (599)\nfrom HST\nat 2005-01-01 00:00 UTC',
        )
        self.assertEqual(
            self.body.get_description(multiline=False),
            'JUPITER (599) from HST at 2005-01-01 00:00 UTC',
        )

        with planetmapper.body._AdjustedSurfaceAltitude(self.body, 123.0):
            self.assertEqual(
                self.body.get_description(),
                'JUPITER (599), alt = 123 km\nfrom HST\nat 2005-01-01 00:00 UTC',
            )

    def test_get_poles_to_plot(self):
        self.assertEqual(self.body.get_poles_to_plot(), [(0, -90, 'S')])

        moon = Body('moon', utc='2000-01-08 03:00:00')
        self.assertEqual(moon.get_poles_to_plot(), [(0, 90, '(N)'), (0, -90, '(S)')])

    def test_add_nans_for_radec_array_wraparounds(self):
        pairs: list[tuple[tuple[list, list], tuple[np.ndarray, np.ndarray]]] = [
            (([], []), (array([]), array([]))),
            (([1], [2]), (array([1]), array([2]))),
            (([1, 2], [3, 4]), (array([1, 2]), array([3, 4]))),
            (([-1, 1], [2, 3]), (array([-1, 1]), array([2, 3]))),
            (([360, 359], [1, 2]), (array([360, 359]), array([1, 2]))),
            (([175, 185], [1, 2]), (array([175, 185]), array([1, 2]))),
            (
                ([0, 360], [-1, -2]),
                (array([0.0, nan, 360.0]), array([-1.0, nan, -2.0])),
            ),
            (
                ([360, 0], [-1, -2]),
                (array([360.0, nan, 0.0]), array([-1.0, nan, -2.0])),
            ),
            (
                ([-175, 175], [-1, -2]),
                (array([-175.0, nan, 175.0]), array([-1.0, nan, -2.0])),
            ),
            (
                ([175, -175], [-1, -2]),
                (array([175.0, nan, -175.0]), array([-1.0, nan, -2.0])),
            ),
            (
                ([1, 2, 359, 350, 340.123, 360, 0, 0.1234], [1, 2, 3, 4, 5, 6, 7, 8]),
                (
                    array(
                        [
                            1.00000e00,
                            2.00000e00,
                            nan,
                            3.59000e02,
                            3.50000e02,
                            3.40123e02,
                            3.60000e02,
                            nan,
                            0.00000e00,
                            1.23400e-01,
                        ]
                    ),
                    array([1.0, 2.0, nan, 3.0, 4.0, 5.0, 6.0, nan, 7.0, 8.0]),
                ),
            ),
            (([0, 269.9, 0], [1, 2]), (array([0.0, 269.9]), array([1, 2]))),
            (([0, 270, 0], [1, 2]), (array([0, 270]), array([1, 2]))),
            (
                ([0, 270.1, 0], [1, 2]),
                (array([0.0, nan, 270.1]), array([1.0, nan, 2.0])),
            ),
        ]
        for (ra_in, dec_in), (ra_expected, dec_expected) in pairs:
            with self.subTest(ra=ra_in, dec=dec_in):
                self.assertArraysEqual(
                    self.body._add_nans_for_radec_array_wraparounds(ra_in, dec_in),
                    (ra_expected, dec_expected),
                    equal_nan=True,
                )

        ras = [1, 1, 0, 270, 0, 270.1, 0, 44.9, 0, 45, 0, 45.1]
        decs = np.arange(len(ras))
        expected: list[tuple[float, np.ndarray, np.ndarray]] = [
            (
                270,
                array(
                    [
                        1.0,
                        1.0,
                        0.0,
                        270.0,
                        0.0,
                        nan,
                        270.1,
                        nan,
                        0.0,
                        44.9,
                        0.0,
                        45.0,
                        0.0,
                        45.1,
                    ]
                ),
                array(
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        nan,
                        5.0,
                        nan,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                    ]
                ),
            ),
            (
                45,
                array(
                    [
                        1.0,
                        1.0,
                        0.0,
                        nan,
                        270.0,
                        nan,
                        0.0,
                        nan,
                        270.1,
                        nan,
                        0.0,
                        44.9,
                        0.0,
                        45.0,
                        0.0,
                        nan,
                        45.1,
                    ]
                ),
                array(
                    [
                        0.0,
                        1.0,
                        2.0,
                        nan,
                        3.0,
                        nan,
                        4.0,
                        nan,
                        5.0,
                        nan,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        nan,
                        11.0,
                    ]
                ),
            ),
            (
                45.1,
                array(
                    [
                        1.0,
                        1.0,
                        0.0,
                        nan,
                        270.0,
                        nan,
                        0.0,
                        nan,
                        270.1,
                        nan,
                        0.0,
                        44.9,
                        0.0,
                        45.0,
                        0.0,
                        45.1,
                    ]
                ),
                array(
                    [
                        0.0,
                        1.0,
                        2.0,
                        nan,
                        3.0,
                        nan,
                        4.0,
                        nan,
                        5.0,
                        nan,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                    ]
                ),
            ),
            (
                1,
                array(
                    [
                        1.0,
                        1.0,
                        0.0,
                        nan,
                        270.0,
                        nan,
                        0.0,
                        nan,
                        270.1,
                        nan,
                        0.0,
                        nan,
                        44.9,
                        nan,
                        0.0,
                        nan,
                        45.0,
                        nan,
                        0.0,
                        nan,
                        45.1,
                    ]
                ),
                array(
                    [
                        0.0,
                        1.0,
                        2.0,
                        nan,
                        3.0,
                        nan,
                        4.0,
                        nan,
                        5.0,
                        nan,
                        6.0,
                        nan,
                        7.0,
                        nan,
                        8.0,
                        nan,
                        9.0,
                        nan,
                        10.0,
                        nan,
                        11.0,
                    ]
                ),
            ),
        ]
        for threshold, ra_expected, dec_expected in expected:
            with self.subTest(threshold=threshold):
                self.assertArraysEqual(
                    self.body._add_nans_for_radec_array_wraparounds(
                        ras, decs, threshold=threshold
                    ),
                    (ra_expected, dec_expected),
                    equal_nan=True,
                )

    @patch('matplotlib.pyplot.show')
    def test_plot_wireframe(self, mock_show: MagicMock):
        # TODO improve these tests by mocking the various matplotlib functions and
        # checking that they are called with the correct arguments (also applies to
        # other plotting tests elsewhere)

        fig, ax = plt.subplots()
        self.body.plot_wireframe_radec(ax, color='red')
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 32)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_wireframe_radec(ax, label_poles=False)
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 31)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_wireframe_radec(ax, grid_lat_limit=30)
        self.assertEqual(len(ax.get_lines()), 18)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 29)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_wireframe_angular(ax, grid_lat_limit=30)
        self.assertEqual(len(ax.get_lines()), 18)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 29)
        plt.close(fig)

        ax = self.body.plot_wireframe_km()
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 32)
        plt.close(ax.figure)

        self.body.add_named_rings()
        self.body.coordinates_of_interest_lonlat.extend(
            [(0, 0), (90, 0), (180, 0), (270, 0)]
        )
        self.body.coordinates_of_interest_radec.append(
            (self.body.target_ra, self.body.target_dec)
        )
        self.body.add_other_bodies_of_interest('amalthea')
        fig, ax = plt.subplots()
        self.body.plot_wireframe_km(
            ax,
            label_poles=False,
            add_axis_labels=False,
            aspect_adjustable='box',
            add_title=False,
            grid_interval=45,
            indicate_equator=True,
            indicate_prime_meridian=True,
        )
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 36)
        plt.close(fig)
        self.body.ring_radii.clear()
        self.body.coordinates_of_interest_lonlat.clear()
        self.body.coordinates_of_interest_radec.clear()
        self.body.other_bodies_of_interest.clear()

        # Test hidden other bodies
        jupiter = planetmapper.Body('jupiter', utc='2005-01-01T04:00')
        jupiter.add_other_bodies_of_interest('thebe', 'metis', 'amalthea', 'adrastea')
        fig, ax = plt.subplots()
        jupiter.plot_wireframe_radec(ax)
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 40)
        plt.close(fig)

        # Test show
        fig, ax = plt.subplots()
        jupiter.plot_wireframe_radec(ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        fig, ax = plt.subplots()
        jupiter.plot_wireframe_km(ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        fig, ax = plt.subplots()
        jupiter.plot_wireframe_angular(ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test radec wraparound
        jupiter_from_amalthea = planetmapper.Body(
            'jupiter', '2005-01-01', observer='amalthea'
        )

        fig, ax = plt.subplots()
        jupiter_from_amalthea.plot_wireframe_radec(ax)
        xlim = ax.get_xlim()
        self.assertTrue(400 > xlim[0] > 350)
        self.assertTrue(10 > xlim[1] > -60)
        plt.close(fig)

        fig, ax = plt.subplots()
        jupiter_from_amalthea.plot_wireframe_radec(ax, use_shifted_meridian=True)
        xlim = ax.get_xlim()
        self.assertTrue(60 > xlim[0] > 10)
        self.assertTrue(-10 > xlim[1] > -60)
        plt.close(fig)

        # Test scalings
        self._test_wireframe_scaling(
            self.body.plot_wireframe_radec,
            'Right Ascension',
            'Declination',
            [
                (
                    None,
                    (196.3774505836621, 196.36652066566225),
                    (-5.570996600931527, -5.560591073745357),
                ),
                (
                    50,
                    (9818.872529183103, 9818.326033283114),
                    (-278.54983004657635, -278.0295536872678),
                ),
                (
                    123456.786,
                    (24244128.89193274, 24242779.519385245),
                    (-687777.3351679309, -686492.7022248907),
                ),
                (
                    1e-06,
                    (0.00019637745058366206, 0.00019636652066566226),
                    (-5.570996600931527e-06, -5.560591073745357e-06),
                ),
            ],
        )
        self._test_wireframe_scaling(
            self.body.plot_wireframe_km,
            'Projected distance (km)',
            'Projected distance (km)',
            [
                (
                    None,
                    (-78640.99608058519, 78641.15962987275),
                    (-73550.89564237543, 73551.12774884349),
                ),
                (
                    50,
                    (-3932049.804029259, 3932057.981493638),
                    (-3677544.782118771, 3677556.387442174),
                ),
                (
                    123456.786,
                    (-9708764623.947643, 9708784815.21704),
                    (-9080357183.429073, 9080385838.54763),
                ),
                (
                    1e-06,
                    (-0.07864099608058517, 0.07864115962987274),
                    (-0.07355089564237542, 0.07355112774884348),
                ),
            ],
        )

        self._test_wireframe_scaling(
            self.body.plot_wireframe_angular,
            'Angular distance (arcsec)',
            'Angular distance (arcsec)',
            [
                (
                    None,
                    (-19.581092792776644, 19.58110648969864),
                    (-18.729913838924922, 18.729984031278835),
                ),
                (
                    50,
                    (-979.0546396388322, 979.055324484932),
                    (-936.4956919462461, 936.4992015639417),
                ),
                (
                    123456.786,
                    (-2417418.7825639686, 2417420.473541936),
                    (-2312334.9646105925, 2312343.6303330082),
                ),
                (
                    1e-06,
                    (-1.9581092792776644e-05, 1.9581106489698642e-05),
                    (-1.872991383892492e-05, 1.8729984031278834e-05),
                ),
            ],
        )

        with self.subTest('radec dms ticks'):
            fig, ax = plt.subplots()
            self.body.plot_wireframe_radec(ax)
            for axis in (ax.xaxis, ax.yaxis):
                self.assertIsInstance(
                    axis.get_major_formatter(), planetmapper.utils.DMSFormatter
                )
                self.assertIsInstance(
                    axis.get_major_locator(), planetmapper.utils.DMSLocator
                )
            plt.close(fig)

            fig, ax = plt.subplots()
            self.body.plot_wireframe_radec(ax, scale_factor=10)
            for axis in (ax.xaxis, ax.yaxis):
                self.assertNotIsInstance(
                    axis.get_major_formatter(), planetmapper.utils.DMSFormatter
                )
                self.assertNotIsInstance(
                    axis.get_major_locator(), planetmapper.utils.DMSLocator
                )
            plt.close(fig)

            for dms_ticks in (True, False):
                for scale_factor in (None, 10):
                    with self.subTest(dms_ticks=dms_ticks, scale_factor=scale_factor):
                        fig, ax = plt.subplots()
                        self.body.plot_wireframe_radec(
                            ax, dms_ticks=dms_ticks, scale_factor=scale_factor
                        )
                        for axis in (ax.xaxis, ax.yaxis):
                            self.assertEqual(
                                isinstance(
                                    axis.get_major_formatter(),
                                    planetmapper.utils.DMSFormatter,
                                ),
                                dms_ticks,
                            )
                            self.assertEqual(
                                isinstance(
                                    axis.get_major_locator(),
                                    planetmapper.utils.DMSLocator,
                                ),
                                dms_ticks,
                            )
                        plt.close(fig)

        with self.subTest('altitude offset'):
            # change in limits ensures that all elements are scaling properly

            # alt, title, xlim ((x0_min, x0_max), (x1_min, x1_max)), ylim (...)
            to_test: list[
                tuple[
                    float,
                    str,
                    tuple[tuple[float, float], tuple[float, float]],
                    tuple[tuple[float, float], tuple[float, float]],
                ]
            ] = [
                (
                    0,
                    'JUPITER (599)\nfrom HST\nat 2005-01-01 00:00 UTC',
                    ((-39320, -117961), (39320, 117961)),
                    ((-36775, -110326), (36775, 110326)),
                ),
                (
                    1000000.0,
                    'JUPITER (599), alt = 1e+06 km\nfrom HST\nat 2005-01-01 00:00 UTC',
                    ((-589320, -1767962), (589320, 1767962)),
                    ((-586775, -1760325), (586775, 1760327)),
                ),
                (
                    -50000.0,
                    'JUPITER (599), alt = -50000 km\nfrom HST\nat 2005-01-01 00:00 UTC',
                    ((-11820, -35461), (11820, 35461)),
                    ((-9276, -27828), (9276, 27828)),
                ),
            ]
            for alt, title, xlims, ylims in to_test:
                with self.subTest(alt=alt):
                    fig, ax = plt.subplots()
                    self.body.plot_wireframe_km(ax, alt=alt)
                    self.assertEqual(ax.get_title(), title)
                    for i in (0, 1):
                        self.assertTrue(
                            min(xlims[i]) < ax.get_xlim()[i] < max(xlims[i])
                        )
                        self.assertTrue(
                            min(ylims[i]) < ax.get_ylim()[i] < max(ylims[i])
                        )
                    plt.close(fig)

    def test_get_local_affine_transform_matrix(self):
        tests: list[
            tuple[
                Callable[[float, float], tuple[float, float]],
                tuple[float, float],
                np.ndarray,
            ]
        ] = [
            (lambda a, b: (a, b), (0, 0), np.eye(3)),
            (lambda a, b: (a, b), (1.234, -56.789), np.eye(3)),
            (
                lambda a, b: (b, a),
                (1.234, -56.789),
                array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            ),
            (
                lambda a, b: (2.3 * a, -5.67 * b),
                (1.234, -56.789),
                array([[2.3, 0.0, 0.0], [0.0, -5.67, 0.0], [0.0, 0.0, 1.0]]),
            ),
            (
                lambda a, b: (2.3 * a**2, -5.67 * b**3 - a),
                (1.234, -56.789),
                array(
                    [
                        [7.97640000e00, 0.00000000e00, -6.34053880e00],
                        [-1.00000000e00, -5.38967779e04, -2.02231771e06],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ),
            (
                lambda a, b: (2.3 * a**2, -5.67 * b**3 - a),
                (100, 300),
                array(
                    [
                        [4.62300000e02, 0.00000000e00, -2.32300000e04],
                        [-1.00000000e00, -1.53600867e06, 3.07712601e08],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ),
        ]

        for func, location, expected in tests:
            with self.subTest(func=func, location=location):
                self.assertArraysClose(
                    self.body._get_local_affine_transform_matrix(func, location),
                    expected,
                )

    def test_get_matplotlib_transform(self):
        self.assertArraysClose(
            self.body._get_matplotlib_transform(
                lambda a, b: (a, b), (1.234, -56.78), None
            ).get_matrix(),
            np.eye(3),
        )

    def test_matplotlib_transforms(self):
        with self.subTest('km2radec'):
            self.assertArraysClose(
                self.body.matplotlib_km2radec_transform().get_matrix(),
                array(
                    [
                        [-6.40343479e-08, 2.88537788e-08, 1.96371986e02],
                        [2.87177471e-08, 6.37324567e-08, -5.56579385e00],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
        with self.subTest('radec2km'):
            self.assertArraysClose(
                self.body.matplotlib_radec2km_transform().get_matrix(),
                array(
                    [
                        [-1.29961991e07, 5.85389484e06, 2.58467100e09],
                        [5.81529840e06, 1.30528119e07, -1.06931243e09],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
        with self.subTest('angular2radec'):
            self.assertArraysClose(
                self.body.matplotlib_angular2radec_transform().get_matrix(),
                array(
                    [
                        [-2.79093570e-04, 0.00000000e00, 1.96371986e02],
                        [6.56168453e-11, 2.77777778e-04, -5.56579385e00],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
            self.assertArraysClose(
                self.body.matplotlib_angular2radec_transform(
                    coordinate_rotation=45
                ).get_matrix(),
                array(
                    [
                        [-1.97349022e-04, -1.97348890e-04, 1.96371986e02],
                        [-1.96418518e-04, 1.96418583e-04, -5.56579385e00],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
        with self.subTest('radec2angular'):
            self.assertArraysClose(
                self.body.matplotlib_radec2angular_transform().get_matrix(),
                array(
                    [
                        [-3.58302602e03, 0.00000000e00, 7.03605934e05],
                        [-3.03254848e00, 3.60000000e03, 2.06323654e04],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
            self.assertArraysClose(
                self.body.matplotlib_radec2angular_transform(
                    coordinate_rotation=45
                ).get_matrix(),
                array(
                    [
                        [-2.53156508e03, -2.54571365e03, 4.82959545e05],
                        [-2.53566278e03, 2.54551979e03, 5.12100973e05],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )

        with self.subTest('inverse'):
            self.assertArraysClose(
                (
                    self.body.matplotlib_km2radec_transform()
                    + self.body.matplotlib_radec2km_transform()
                ).get_matrix(),
                np.eye(3),
                atol=1e-2,
            )
            self.assertArraysClose(
                (
                    self.body.matplotlib_angular2radec_transform()
                    + self.body.matplotlib_radec2angular_transform()
                ).get_matrix(),
                np.eye(3),
                atol=1e-2,
            )

        fig, axis = plt.subplots()
        for ax in [None, axis]:
            with self.subTest(ax=ax):
                # Test consistent results
                self.body.matplotlib_radec2km_transform(ax)
                t1 = self.body.matplotlib_radec2km_transform(ax)
                t2 = self.body.matplotlib_radec2km_transform(ax)
                self.assertEqual(t1, t2)

                self.body.matplotlib_km2radec_transform(ax)
                t1 = self.body.matplotlib_km2radec_transform(ax)
                t2 = self.body.matplotlib_km2radec_transform(ax)
                self.assertEqual(t1, t2)

        plt.close(fig)

    def test_plot_wireframe_custom(self):
        arguments_and_limits: list[
            tuple[
                dict,
                tuple[float, float],
                tuple[float, float],
            ]
        ] = [
            (
                {},
                (196.36652066566225, 196.3774505836621),
                (-5.570996600931527, -5.560591073745357),
            ),
            (
                dict(coordinate_func=None, transform=None),
                (196.36652066566225, 196.3774505836621),
                (-5.570996600931527, -5.560591073745357),
            ),
            (
                dict(coordinate_func=self.body.radec2km),
                (-78640.99608058519, 78641.15962987275),
                (-73550.89564237543, 73551.12774884349),
            ),
            (
                dict(transform=self.body.matplotlib_radec2km_transform()),
                (-78666.01732656956, 78665.97486374379),
                (-73527.70551617145, 73527.85605175495),
            ),
            (
                dict(
                    coordinate_func=self.body.radec2angular,
                    transform=self.body.matplotlib_angular2radec_transform(),
                ),
                (196.36652066335904, 196.37745058135863),
                (-5.570996601039565, -5.560591073731259),
            ),
        ]
        atol = 1e-5
        rtol = 1e-2
        for kwargs, xlim, ylim in arguments_and_limits:
            with self.subTest(kwargs):
                fig, ax = plt.subplots()
                self.body.plot_wireframe_custom(ax, **kwargs)
                self.assertArraysClose(ax.get_xlim(), xlim, atol=atol, rtol=rtol)
                self.assertArraysClose(ax.get_ylim(), ylim, atol=atol, rtol=rtol)
                plt.close(fig)


class TestAdjustedSurfaceAltitude(common_testing.BaseTestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        self.original_radii = self.body.radii
        self.original_r_eq = self.body.r_eq
        self.original_r_polar = self.body.r_polar
        self.original_flattening = self.body.flattening

        self.adjustments_to_check = [0, 0.0, 0.12345, 1, 1.0, 9e9, -42.12345]

    def check_adjustment(self, alt: float):
        self.assertAlmostEqual(self.body._alt_adjustment, alt)
        self.assertArraysClose(self.body.radii, self.original_radii + alt)
        self.assertAlmostEqual(self.body.r_eq, self.original_r_eq + alt)
        self.assertAlmostEqual(self.body.r_polar, self.original_r_polar + alt)
        self.assertAlmostEqual(
            self.body.flattening,
            (self.original_r_eq - self.original_r_polar) / (self.original_r_eq + alt),
        )
        self.assertArraysClose(
            spice.bodvar(self.body.target_body_id, 'RADII', 3),
            self.original_radii + alt,
        )

    def check_reset(self):
        self.assertEqual(self.body._alt_adjustment, 0)
        self.assertArraysClose(self.body.radii, self.original_radii)
        self.assertAlmostEqual(self.body.r_eq, self.original_r_eq)
        self.assertAlmostEqual(self.body.r_polar, self.original_r_polar)
        self.assertAlmostEqual(self.body.flattening, self.original_flattening)
        self.assertArraysClose(
            spice.bodvar(self.body.target_body_id, 'RADII', 3), self.original_radii
        )
        self.check_adjustment(0)

    def test_context_manager(self):
        for alt in self.adjustments_to_check:
            with self.subTest('normal', alt=alt):
                self.check_reset()
                with planetmapper.body._AdjustedSurfaceAltitude(self.body, alt):
                    self.check_adjustment(alt)
                self.check_reset()

            with self.subTest('error', alt=alt):
                self.check_reset()
                with self.assertRaises(CustomTestException):
                    with planetmapper.body._AdjustedSurfaceAltitude(self.body, alt):
                        self.check_adjustment(alt)
                        raise CustomTestException
                self.check_reset()

    def test_decorator(self):
        @planetmapper.body._adjust_surface_altitude_decorator
        def func(body: Body, *, alt: float = 0.0) -> None:
            self.check_adjustment(alt)

        @planetmapper.body._adjust_surface_altitude_decorator
        def func_with_error(body: Body, *, alt: float = 0.0) -> None:
            self.check_adjustment(alt)
            raise CustomTestException

        for alt in self.adjustments_to_check:
            with self.subTest('normal', alt=alt):
                self.check_reset()
                func(self.body, alt=alt)
                self.check_reset()

            with self.subTest('error', alt=alt):
                self.check_reset()
                with self.assertRaises(CustomTestException):
                    func_with_error(self.body, alt=alt)
                self.check_reset()

    def test_error_when_nested(self):
        self.check_reset()
        with planetmapper.body._AdjustedSurfaceAltitude(self.body, 123):
            self.check_adjustment(123)
            with planetmapper.body._AdjustedSurfaceAltitude(self.body, 123):
                self.check_adjustment(123)

            with self.assertRaises(ValueError):
                with planetmapper.body._AdjustedSurfaceAltitude(self.body, 456):
                    self.fail('Context manager should not enter')

            self.check_adjustment(123)
        self.check_reset()
        with planetmapper.body._AdjustedSurfaceAltitude(self.body, -42.34):
            # check everything still works properly after an error
            self.check_adjustment(-42.34)
        self.check_reset()

    def test_error_non_finite(self):
        for v in [np.nan, np.inf, -np.inf]:
            with self.subTest(v):
                self.check_reset()
                with self.assertRaises(ValueError):
                    with planetmapper.body._AdjustedSurfaceAltitude(self.body, v):
                        self.fail('Context manager should not enter')
                self.check_reset()

        self.check_reset()
        with planetmapper.body._AdjustedSurfaceAltitude(self.body, -42.34):
            # check everything still works properly after an error
            self.check_adjustment(-42.34)
        self.check_reset()

    def test_cache(self):
        functions_called = []

        @planetmapper.base._cache_clearable_result
        def f_clearable(body, a, b=1):
            functions_called.append('f_clearable')
            return ('f_clearable', a * b)

        @planetmapper.body._cache_clearable_alt_dependent_result
        def f_clearable_alt_dependent(body, a, b=1):
            functions_called.append('f_clearable_alt_dependent')
            return ('f_clearable_alt_dependent', a * b + body._alt_adjustment)

        @planetmapper.base._cache_stable_result
        def f_stable(body, a, b=1):
            functions_called.append('f_stable')
            return ('f_stable', a * b)

        self.body._clear_cache()

        # Initial runs to populate cache, then retrieve cached values
        for run in (1, 2):
            with self.subTest(run):
                self.assertEqual(f_clearable(self.body, 1), ('f_clearable', 1))
                self.assertEqual(
                    f_clearable_alt_dependent(self.body, 1),
                    ('f_clearable_alt_dependent', 1),
                )
                self.assertEqual(f_stable(self.body, 1), ('f_stable', 1))
                self.assertEqual(
                    functions_called,
                    ['f_clearable', 'f_clearable_alt_dependent', 'f_stable'],
                )
                self.assertEqual(len(self.body._cache), 2)
                self.assertEqual(len(self.body._stable_cache), 1)

        # Populate cache with new args
        functions_called.clear()
        for run in (1, 2):
            with self.subTest(run):
                self.assertEqual(f_clearable(self.body, 2, b=3), ('f_clearable', 6))
                self.assertEqual(
                    f_clearable_alt_dependent(self.body, 2, b=3),
                    ('f_clearable_alt_dependent', 6),
                )
                self.assertEqual(f_stable(self.body, 2, b=3), ('f_stable', 6))
                self.assertEqual(
                    functions_called,
                    [
                        'f_clearable',
                        'f_clearable_alt_dependent',
                        'f_stable',
                    ],
                )
                self.assertEqual(len(self.body._cache), 4)
                self.assertEqual(len(self.body._stable_cache), 2)

        # Test alt adjustment
        functions_called.clear()
        with planetmapper.body._AdjustedSurfaceAltitude(self.body, 100):
            for run in (1, 2):
                with self.subTest(run):
                    self.assertEqual(f_clearable(self.body, 2, b=3), ('f_clearable', 6))
                    self.assertEqual(
                        f_clearable_alt_dependent(self.body, 2, b=3),
                        ('f_clearable_alt_dependent', 106),
                    )
                    self.assertEqual(f_stable(self.body, 2, b=3), ('f_stable', 6))
                    self.assertEqual(
                        functions_called,
                        [
                            'f_clearable_alt_dependent',
                        ],
                    )
                    self.assertEqual(len(self.body._cache), 5)
                    self.assertEqual(len(self.body._stable_cache), 2)

        # Test that we re-hit cache on exit
        functions_called.clear()
        self.assertEqual(f_clearable(self.body, 2, b=3), ('f_clearable', 6))
        self.assertEqual(
            f_clearable_alt_dependent(self.body, 2, b=3),
            ('f_clearable_alt_dependent', 6),
        )
        self.assertEqual(f_stable(self.body, 2, b=3), ('f_stable', 6))
        self.assertEqual(
            functions_called,
            [],
        )
        self.assertEqual(len(self.body._cache), 5)
        self.assertEqual(len(self.body._stable_cache), 2)

        # Test that cache clears properly
        self.body._clear_cache()
        self.assertEqual(len(self.body._cache), 0)
        self.assertEqual(len(self.body._stable_cache), 2)

        functions_called.clear()
        self.assertEqual(f_clearable(self.body, 2, b=3), ('f_clearable', 6))
        self.assertEqual(
            f_clearable_alt_dependent(self.body, 2, b=3),
            ('f_clearable_alt_dependent', 6),
        )
        self.assertEqual(f_stable(self.body, 2, b=3), ('f_stable', 6))
        self.assertEqual(
            functions_called,
            [
                'f_clearable',
                'f_clearable_alt_dependent',
            ],
        )
        self.assertEqual(len(self.body._cache), 2)
        self.assertEqual(len(self.body._stable_cache), 2)


class CustomTestException(Exception):
    pass
