import datetime
import unittest

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, nan
from spiceypy.utils.exceptions import NotFoundError, SpiceSPKINSUFFDATA

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

    def test_add_satellites_to_bodies_of_interest(self):
        self.body.other_bodies_of_interest.clear()
        with self.assertRaises(SpiceSPKINSUFFDATA):
            self.body.add_satellites_to_bodies_of_interest()

        self.body.other_bodies_of_interest.clear()
        self.body.add_satellites_to_bodies_of_interest(skip_insufficient_data=True)
        self.assertEqual(
            self.body.other_bodies_of_interest,
            [
                Body('AMALTHEA', '2005-01-01T00:00:00.000000', 'HST'),
                Body('THEBE', '2005-01-01T00:00:00.000000', 'HST'),
                Body('ADRASTEA', '2005-01-01T00:00:00.000000', 'HST'),
                Body('METIS', '2005-01-01T00:00:00.000000', 'HST'),
            ],
        )
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
        ]
        for lonlat, radec in pairs:
            with self.subTest(lonlat):
                self.assertTrue(np.allclose(self.body.lonlat2radec(*lonlat), radec))

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
        ]
        for radec, lonlat in pairs:
            with self.subTest(radec):
                self.assertTrue(
                    np.allclose(
                        self.body.radec2lonlat(*radec),
                        lonlat,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        self.body.lonlat2radec(*lonlat),
                        radec,
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
        ]
        for (lon, lat), targvec in pairs:
            with self.subTest((lon, lat)):
                self.assertTrue(
                    np.allclose(self.body.lonlat2targvec(lon, lat), targvec)
                )

    def test_targvec2lonlat(self):
        pairs = [
            (np.array([0, 0, 0]), (0.0, 90.0)),
            (np.array([1, 2, 3]), (296.565051177078, 89.98665551067639)),
            (np.array([-9876, 543210, 0]), (268.9584308375042, 0.0)),
        ]
        for targvec, lonlat in pairs:
            with self.subTest(targvec):
                self.assertTrue(np.allclose(self.body.targvec2lonlat(targvec), lonlat))

    def test_km_radec(self):
        pairs = [
            [(0, 0), (196.37198562427025, -5.565793847134351)],
            [(99999, 99999), (196.36846760253624, -5.556548919202668)],
        ]
        for km, radec in pairs:
            with self.subTest(km):
                self.assertTrue(np.allclose(self.body.km2radec(*km), radec))
                self.assertTrue(np.allclose(self.body.radec2km(*radec), km, atol=1e-3))

    def test_km_lonlat(self):
        pairs = [
            [(0, 0), (153.1235185909613, -3.0887371238645795)],
            [(123, 456.789), (153.02550380815194, -2.6701272595645387)],
            [(-500, -200), (153.52449023101565, -3.2726499274177465)],
            [(5000, 50001), (147.49451214685632, 47.45177666020315)],
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
        self.assertTrue(
            np.allclose(
                self.body.illumination_angles_from_lonlat(0, 0),
                (10.31594976458697, 163.2795134457034, 152.99822832991876),
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.illumination_angles_from_lonlat(123.456, -78.9),
                (10.316968817304499, 79.16351827229181, 77.68583738495468),
            )
        )

    def test_azimuth_angle_from_lonlat(self):
        self.assertAlmostEqual(
            self.body.azimuth_angle_from_lonlat(0, 0), 177.66817822757469
        )
        self.assertAlmostEqual(
            self.body.azimuth_angle_from_lonlat(123.456, -78.9), 169.57651996164563
        )

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

    def test_if_lonlat_illuminated(self):
        pairs: list[tuple[tuple[float, float], bool]] = [
            ((0, 0), False),
            ((180, 12), True),
        ]
        for (lon, lat), illuminated in pairs:
            with self.subTest(lon=lon, lat=lat):
                self.assertEqual(
                    self.body.test_if_lonlat_illuminated(lon, lat), illuminated
                )

    def test_ring_plane_coordinates(self):
        self.assertTrue(
            np.allclose(
                self.body.ring_plane_coordinates(
                    196.37198562427025, -5.565793847134351
                ),
                (nan, nan, nan),
                equal_nan=True,
            ),
        )
        self.assertTrue(
            np.allclose(
                self.body.ring_plane_coordinates(
                    196.37198562427025, -5.565793847134351, only_visible=False
                ),
                (4638.105239104683, 156.0690984698183, 819638074.3312378),
                equal_nan=True,
            ),
        )
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

    def test_radial_velocity_from_lonlat(self):
        self.assertAlmostEqual(
            self.body.radial_velocity_from_lonlat(0, 0), -20.796924908179438
        )

    def test_distance_from_lonalt(self):
        self.assertAlmostEqual(self.body.distance_from_lonlat(0, 0), 819701772.0279644)

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

    def test_north_pole_angle(self):
        self.assertAlmostEqual(self.body.north_pole_angle(), -24.256254044782136)

    def test_get_description(self):
        self.assertEqual(
            self.body.get_description(),
            'JUPITER (599)\nfrom HST\nat 2005-01-01 00:00 UTC',
        )

    def test_get_poles_to_plot(self):
        self.assertEqual(self.body.get_poles_to_plot(), [(0, -90, 'S')])

    def test_plot_wireframe(self):
        fig, ax = plt.subplots()
        self.body.plot_wireframe_radec(ax, color='red')
        plt.close(fig)

        ax = self.body.plot_wireframe_km()
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
            grid_interval=43,
            indicate_equator=True,
            indicate_prime_meridian=True,
        )
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
        plt.close(fig)
