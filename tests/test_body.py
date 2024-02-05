import datetime
import unittest
from unittest.mock import MagicMock, patch

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, nan
from spiceypy.utils.exceptions import (
    NotFoundError,
    SpiceKERNELVARNOTFOUND,
    SpiceSPKINSUFFDATA,
)

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import Body


class TestBody(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')

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

        # Test Saturrn automatically has A, B & C rings added
        saturn = Body('saturn', '2000-01-01')
        self.assertEqual(saturn.target, 'SATURN')
        self.assertEqual(saturn.target_body_id, 699)
        self.assertEqual(
            saturn.ring_radii, {74658.0, 91975.0, 117507.0, 122340.0, 136780.0}
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
        self.assertAlmostEqual(self.body.target_diameter_arcsec, 35.98242703657337)
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

        moon = Body('moon', '2005-01-01')
        self.assertEqual(moon.positive_longitude_direction, 'E')
        self.assertTrue(moon.prograde)

        sun = Body('sun', '2005-01-01')
        self.assertEqual(sun.positive_longitude_direction, 'E')
        self.assertTrue(sun.prograde)
        self.assertTrue(np.isnan(sun.subsol_lon))
        self.assertTrue(np.isnan(sun.subsol_lat))

    def test_repr(self):
        self.assertEqual(
            repr(self.body),
            "Body('JUPITER', '2005-01-01T00:00:00.000000', observer='HST')",
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
                'target': 'JUPITER',
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

        copy = body.copy()
        self.assertEqual(body, copy)
        self.assertIsNot(body, copy)
        self.assertEqual(body._get_kwargs(), copy._get_kwargs())
        self.assertEqual(body.other_bodies_of_interest, copy.other_bodies_of_interest)
        self.assertEqual(
            body.coordinates_of_interest_lonlat, copy.coordinates_of_interest_lonlat
        )
        self.assertEqual(
            body.coordinates_of_interest_radec, copy.coordinates_of_interest_radec
        )
        self.assertEqual(body.ring_radii, copy.ring_radii)

        body.coordinates_of_interest_lonlat.append((5, 6))
        self.assertNotEqual(
            body.coordinates_of_interest_lonlat, copy.coordinates_of_interest_lonlat
        )

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
                self.assertTrue(
                    np.allclose(
                        self.body.angular2radec(x, y, **kw), radec  # type: ignore
                    )
                )
                self.assertTrue(
                    np.allclose(
                        self.body.radec2angular(*radec, **kw), (x, y)  # type: ignore
                    )
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
                self.assertTrue(np.allclose(radec, radec_expected))

                xy = self.body.radec2angular(*radec_expected, **kw)
                self.assertTrue(np.allclose(xy, (x, y)))

    def test_angular_lonlat(self):
        pairs = [
            ((0, 0), {}, (153.1234529836525, -3.088664454046201)),
            (
                (0, 0),
                {'coordinate_rotation': 123},
                (153.1234529836525, -3.088664454046201),
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
                self.assertTrue(
                    np.allclose(
                        self.body.angular2lonlat(x, y, **kw), lonlat, equal_nan=True
                    )
                )
                if np.isfinite(lonlat[0]):
                    self.assertTrue(
                        np.allclose(self.body.lonlat2angular(*lonlat, **kw), (x, y))
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

    def test_km_radec(self):
        pairs = [
            [(0, 0), (196.37198562427025, -5.565793847134351)],
            [(99999, 99999), (196.36846760253624, -5.556548919202668)],
        ]
        for km, radec in pairs:
            with self.subTest(km):
                self.assertTrue(np.allclose(self.body.km2radec(*km), radec))
                self.assertTrue(np.allclose(self.body.radec2km(*radec), km, atol=1e-3))

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
            [(0, 0), (153.1235185909613, -3.0887371238645795)],
            [(123, 456.789), (153.02550380815194, -2.6701272595645387)],
            [(-500, -200), (153.52449023101565, -3.2726499274177465)],
            [(5000, 50001), (147.49441295554598, 47.45174759079364)],
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

    def test_km_angular(self):
        pairs: list[tuple[tuple[float, float], dict, tuple[float, float]]] = [
            ((0, 0), {}, (4.6729617106227635e-09, 1.0370567346858554e-08)),
            (
                (0, 0),
                {'coordinate_rotation': 123},
                (4.6729617106227635e-09, 1.0370567346858554e-08),
            ),
            ((1.234, 5.678), {}, (13739.866378614151, 18556.388206846823)),
            ((-3600.1234, 45678), {}, (61525334.93172047, 171364244.1505089)),
            (
                (1.234, 5.678),
                {'coordinate_rotation': 123},
                (8079.429074795995, -21629.754904840156),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': 123},
                (927957585.3290204, -480110160.1311036),
            ),
            (
                (1.234, 5.678),
                {'origin_dec': 12.3},
                (105009703.24513194, 233032424.31876734),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': -12.3},
                (-568773415.4728397, 129941895.59871267),
            ),
            (
                (1.234, 5.678),
                {'origin_ra': -123, 'origin_dec': 12.3, 'coordinate_rotation': -123},
                (-445228360.6330424, 459438707.21556187),
            ),
        ]
        for (x, y), kw, km in pairs:
            with self.subTest(x=x, y=y, kw=kw):
                self.assertTrue(
                    np.allclose(self.body.angular2km(x, y, **kw), km, atol=1e-3)  # type: ignore
                )
                self.assertTrue(
                    np.allclose(self.body.km2angular(*km, **kw), (x, y))  # type: ignore
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
        self.assertTrue(
            np.allclose(
                self.body.ring_radec(10000, npts=5),
                (
                    array([nan, 196.37142013, 196.37228744, nan, nan]),
                    array([nan, -5.5655251, -5.56589635, nan, nan]),
                ),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.ring_radec(123456.789, npts=3, only_visible=False),
                (
                    array([196.36825958, 196.37571178, 196.36825958]),
                    array([-5.56452821, -5.56705935, -5.56452821]),
                ),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.ring_radec(np.nan, npts=2, only_visible=False),
                (array([nan, nan]), array([nan, nan])),
                equal_nan=True,
            )
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
        ]
        for graphic, centric in pairs:
            with self.subTest(graphic):
                self.assertTrue(
                    np.allclose(self.body.graphic2centric_lonlat(*graphic), centric)
                )
                self.assertTrue(
                    np.allclose(self.body.centric2graphic_lonlat(*centric), graphic)
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
        self.assertAlmostEqual(self.body.north_pole_angle(), -24.256254044782136)

    def test_get_description(self):
        self.assertEqual(
            self.body.get_description(),
            'JUPITER (599)\nfrom HST\nat 2005-01-01 00:00 UTC',
        )

    def test_get_poles_to_plot(self):
        self.assertEqual(self.body.get_poles_to_plot(), [(0, -90, 'S')])

        moon = Body('moon', utc='2000-01-08 03:00:00')
        self.assertEqual(moon.get_poles_to_plot(), [(0, 90, '(N)'), (0, -90, '(S)')])

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

    def test_matplotlib_transforms(self):
        self.assertTrue(
            np.allclose(
                self.body.matplotlib_km2radec_transform().get_matrix(),
                array(
                    [
                        [-6.40343479e-08, 2.88537788e-08, 1.96371986e02],
                        [2.87177471e-08, 6.37324567e-08, -5.56579385e00],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
        )

        self.assertTrue(
            np.allclose(
                self.body.matplotlib_radec2km_transform().get_matrix(),
                array(
                    [
                        [-1.29809749e07, 5.87691418e06, 2.58180951e09],
                        [5.84920736e06, 1.30424639e07, -1.07602880e09],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            )
        )

        fig, axis = plt.subplots()
        for ax in [None, axis]:
            with self.subTest(ax=ax):
                # Test caching works
                self.body.matplotlib_radec2km_transform(ax)
                t1 = self.body.matplotlib_radec2km_transform(ax)
                self.body._mpl_transform_radec2km = None
                t2 = self.body.matplotlib_radec2km_transform(ax)
                self.assertEqual(t1, t2)

                self.body.matplotlib_km2radec_transform(ax)
                t1 = self.body.matplotlib_km2radec_transform(ax)
                self.body._mpl_transform_km2radec = None
                t2 = self.body.matplotlib_km2radec_transform(ax)
                self.assertEqual(t1, t2)

        plt.close(fig)
