import unittest

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, nan

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import BodyXY
from planetmapper.body_xy import Backplane, BackplaneNotFoundError


class TestFunctions(unittest.TestCase):
    def test_make_backplane_documentation_str(self):
        self.assertIsInstance(
            planetmapper.body_xy._make_backplane_documentation_str(), str
        )


class TestBodyXY(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
        )
        self.body_zero_size = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
        )

    def test_init(self):
        self.assertEqual(
            BodyXY('jupiter', utc='2005-01-01T00:00:00', sz=50),
            BodyXY('jupiter', utc='2005-01-01T00:00:00', nx=50, ny=50),
        )
        with self.assertRaises(ValueError):
            BodyXY('jupiter', utc='2005-01-01T00:00:00', nx=1, ny=2, sz=50)

    def test_attributes(self):
        self.assertEqual(self.body._nx, 15)
        self.assertEqual(self.body._ny, 10)
        self.assertEqual(self.body_zero_size._nx, 0)
        self.assertEqual(self.body_zero_size._ny, 0)

    def test_from_body(self):
        body = planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
        body.add_other_bodies_of_interest('amalthea')
        body.coordinates_of_interest_lonlat.append((0, 0))
        body.coordinates_of_interest_radec.extend([(0, 0), (1, 1)])
        body.add_named_rings()

        body_xy = BodyXY.from_body(body, nx=15, ny=10)
        self.assertEqual(
            body_xy,
            BodyXY('Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10),
        )

        self.assertEqual(body.target, body_xy.target)
        self.assertEqual(body.utc, body_xy.utc)
        self.assertEqual(body.observer, body_xy.observer)
        self.assertEqual(
            body.coordinates_of_interest_lonlat, body_xy.coordinates_of_interest_lonlat
        )
        self.assertEqual(
            body.coordinates_of_interest_radec, body_xy.coordinates_of_interest_radec
        )
        self.assertEqual(body.ring_radii, body_xy.ring_radii)

        body.coordinates_of_interest_radec.clear()
        self.assertNotEqual(
            body.coordinates_of_interest_radec, body_xy.coordinates_of_interest_radec
        )

    def test_to_body(self):
        body_xy = BodyXY('Jupiter', observer='HST', utc='2005-01-01T00:00:00', sz=10)
        body_xy.add_other_bodies_of_interest('amalthea')
        body_xy.coordinates_of_interest_lonlat.append((0, 0))
        body_xy.coordinates_of_interest_radec.extend([(0, 0), (1, 1)])

        body = body_xy.to_body()
        self.assertEqual(
            body,
            planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00'),
        )

        self.assertEqual(body.target, body_xy.target)
        self.assertEqual(body.utc, body_xy.utc)
        self.assertEqual(body.observer, body_xy.observer)
        self.assertEqual(
            body.coordinates_of_interest_lonlat, body_xy.coordinates_of_interest_lonlat
        )
        self.assertEqual(
            body.coordinates_of_interest_radec, body_xy.coordinates_of_interest_radec
        )
        self.assertEqual(body.ring_radii, body_xy.ring_radii)

        body.coordinates_of_interest_radec.clear()
        self.assertNotEqual(
            body.coordinates_of_interest_radec, body_xy.coordinates_of_interest_radec
        )

        self.assertEqual(BodyXY.from_body(body, sz=10), body_xy)

    def test_repr(self):
        self.assertEqual(
            repr(self.body),
            "BodyXY('JUPITER', '2005-01-01T00:00:00.000000', 15, 10, observer='HST')",
        )
        self.assertEqual(
            repr(self.body_zero_size),
            "BodyXY('JUPITER', '2005-01-01T00:00:00.000000', 0, 0, observer='HST')",
        )

    def test_eq(self):
        self.assertEqual(self.body, self.body)
        self.assertEqual(self.body_zero_size, self.body_zero_size)
        self.assertEqual(
            self.body,
            BodyXY('Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10),
        )
        self.assertEqual(
            self.body_zero_size,
            BodyXY('Jupiter', observer='HST', utc='2005-01-01T00:00:00'),
        )

        self.assertNotEqual(self.body, self.body_zero_size)
        self.assertNotEqual(
            self.body, BodyXY('Jupiter', utc='2005-01-01T00:00:00', nx=14, ny=10)
        )
        self.assertNotEqual(
            self.body, BodyXY('Jupiter', utc='2005-01-01T00:00:00', nx=15, ny=11)
        )

    def test_hash(self):
        with self.assertRaises(TypeError):
            hash(self.body)
        with self.assertRaises(TypeError):
            hash(self.body_zero_size)
        with self.assertRaises(TypeError):
            d = {self.body: 1}

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
                'nx': 15,
                'ny': 10,
            },
        )

    def test_copy(self):
        body = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
        )
        body.add_other_bodies_of_interest('amalthea')
        body.coordinates_of_interest_lonlat.append((0, 0))
        body.coordinates_of_interest_radec.extend([(1, 2), (3, 4)])
        body.add_named_rings()
        body.set_disc_params(1, 2, 3, 4)

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
        self.assertEqual(body.get_img_size(), copy.get_img_size())
        self.assertEqual(body.get_disc_params(), copy.get_disc_params())
        self.assertEqual(body.get_disc_method(), copy.get_disc_method())

        body.set_x0(-99)
        self.assertNotEqual(body, copy)
        self.assertNotEqual(body.get_x0(), copy.get_x0())

    def test_cache(self):
        self.body._cache[' test '] = None
        self.body._clear_cache()
        self.assertEqual(len(self.body._cache), 0)

        for fn in (
            self.body.set_x0,
            self.body.set_r0,
            self.body.set_y0,
            self.body.set_rotation,
        ):
            with self.subTest(fn.__name__):
                self.body._cache[' test '] = None
                fn(np.random.rand())
                self.assertEqual(len(self.body._cache), 0)

        self.body._stable_cache.clear()
        self.body.get_emission_angle_map(degree_interval=90)
        self.assertGreater(len(self.body._stable_cache), 0)

    def test_xy_conversions(self):
        # xy, radec, lonlat, km
        coordinates = [
            [
                (0, 0),
                (196.3684350770821, -5.581107015413806),
                (nan, nan),
                (-43904.61179685593, -220489.3308737278),
            ],
            [
                (5, 8),
                (196.37198562427025, -5.565793847134351),
                (153.1235185909613, -3.0887371238645795),
                (0.0, 0.0),
            ],
            [
                (4.1, 7.1),
                (196.37198562427025, -5.567914131973045),
                (164.3872136538264, -28.87847195832716),
                (-12460.732038021088, -27653.738419771194),
            ],
            [
                (1.234, 5.678),
                (196.37369462098349, -5.572965121633222),
                (nan, nan),
                (-64329.40829181671, -83534.81246519089),
            ],
            [
                (-3, 25),
                (196.40157351750477, -5.555192422940882),
                (nan, nan),
                (-321776.04008579254, 311334.414850235),
            ],
            [
                (7.9, 5.1),
                (196.36512123303984, -5.565793847134351),
                (nan, nan),
                (89106.49046421051, -40151.24767804146),
            ],
        ]

        for xy, radec, lonlat, km in coordinates:
            for body in (self.body, self.body_zero_size):
                body.set_disc_params(5, 8, 3, 45)
                with self.subTest(xy=xy, body=body):
                    self.assertTrue(
                        np.allclose(body.xy2radec(*xy), radec, equal_nan=True)
                    )
                    self.assertTrue(
                        np.allclose(body.xy2lonlat(*xy), lonlat, equal_nan=True)
                    )
                    self.assertTrue(
                        np.allclose(body.xy2km(*xy), km, equal_nan=True, atol=1e-3)
                    )

                    self.assertTrue(
                        np.allclose(body.radec2xy(*radec), xy, equal_nan=True)
                    )
                    if not any(np.isnan(lonlat)):
                        self.assertTrue(
                            np.allclose(body.lonlat2xy(*lonlat), xy, equal_nan=True)
                        )
                    self.assertTrue(np.allclose(body.km2xy(*km), xy, equal_nan=True))

    def test_set_disc_params(self):
        x0, y0, r0, rotation = [1.1, 2.2, 3.3, 4.4]
        self.body.set_disc_params(x0, y0, r0, rotation)
        self.assertEqual(self.body.get_x0(), x0)
        self.assertEqual(self.body.get_y0(), y0)
        self.assertEqual(self.body.get_r0(), r0)
        self.assertAlmostEqual(self.body.get_rotation(), rotation)

        self.body.set_disc_params()
        self.assertEqual(self.body.get_x0(), x0)
        self.assertEqual(self.body.get_y0(), y0)
        self.assertEqual(self.body.get_r0(), r0)
        self.assertAlmostEqual(self.body.get_rotation(), rotation)

        x0, y0, r0, rotation = [1.11, 2.22, 3.33, 4.44]
        self.body.set_disc_params(x0=x0, y0=y0, r0=r0, rotation=rotation)
        self.assertEqual(self.body.get_x0(), x0)
        self.assertEqual(self.body.get_y0(), y0)
        self.assertEqual(self.body.get_r0(), r0)
        self.assertAlmostEqual(self.body.get_rotation(), rotation)

    def test_disc_params(self):
        with self.subTest('args'):
            self.body.set_disc_params(0, 0, 1, 0)
            x0, y0, r0, rotation = [11.1, 12.2, 13.3, 14.4]
            self.body.adjust_disc_params(x0, y0, r0, rotation)
            self.assertEqual(self.body.get_x0(), x0)
            self.assertEqual(self.body.get_y0(), y0)
            self.assertEqual(self.body.get_r0(), r0 + 1)
            self.assertAlmostEqual(self.body.get_rotation(), rotation)
        with self.subTest('kwargs'):
            self.body.set_disc_params(0, 0, 1, 0)
            x0, y0, r0, rotation = [21.1, 22.2, 23.3, 24.4]
            self.body.adjust_disc_params(dx=x0, dy=y0, dr=r0, drotation=rotation)
            self.assertEqual(self.body.get_x0(), x0)
            self.assertEqual(self.body.get_y0(), y0)
            self.assertEqual(self.body.get_r0(), r0 + 1)
            self.assertAlmostEqual(self.body.get_rotation(), rotation)

        functions = [
            [self.body.set_x0, self.body.get_x0],
            [self.body.set_y0, self.body.get_y0],
            [self.body.set_r0, self.body.get_r0],
            [self.body.set_rotation, self.body.get_rotation],
        ]
        for setter, getter in functions:
            for v in [1e-3, 10, 123.4567]:
                with self.subTest(setter=setter, v=v):
                    setter(v)
                    self.assertAlmostEqual(getter(), v)
        for setter, getter in functions:
            with self.subTest(setter=setter):
                with self.assertRaises(ValueError):
                    setter(np.nan)
                with self.assertRaises(TypeError):
                    setter('a string')
                with self.assertRaises(TypeError):
                    setter(np.array([1, 2, 3]))

        self.body.set_plate_scale_arcsec(1)
        self.assertAlmostEqual(self.body.get_plate_scale_arcsec(), 1)
        self.assertAlmostEqual(self.body.get_r0(), 17.991213518286685)

        self.body.set_plate_scale_km(1)
        self.assertAlmostEqual(self.body.get_plate_scale_km(), 1)
        self.assertAlmostEqual(self.body.get_r0(), 71492.0)

        params = (98.76, -5.4, 3.2, 1.0)
        self.body.set_disc_params(*params)
        self.assertTrue(np.allclose(self.body.get_disc_params(), params))

    def test_centre_disc(self):
        self.body.set_disc_params(0, 0, 1, 0)
        self.body.centre_disc()
        self.assertEqual(self.body.get_disc_params(), (7.5, 5.0, 4.5, 0.0))
        self.assertEqual(self.body.get_disc_method(), 'centre_disc')

    def test_img_size(self):
        for body in (self.body, self.body_zero_size):
            body.set_disc_params(0, 0, 1, 0)

        self.assertEqual(self.body.get_img_size(), (15, 10))
        self.assertEqual(self.body_zero_size.get_img_size(), (0, 0))

        self.body_zero_size.set_img_size(3, 4)
        self.assertEqual(self.body_zero_size.get_img_size(), (3, 4))
        self.body_zero_size.set_img_size()
        self.assertEqual(self.body_zero_size.get_img_size(), (3, 4))
        self.body_zero_size.set_img_size(nx=5)
        self.assertEqual(self.body_zero_size.get_img_size(), (5, 4))
        self.body_zero_size.set_img_size(ny=5)
        self.assertEqual(self.body_zero_size.get_img_size(), (5, 5))

        self.body_zero_size.set_img_size(15, 10)
        self.assertEqual(self.body, self.body_zero_size)
        self.assertTrue(self.body_zero_size._test_if_img_size_valid())

        self.body_zero_size.set_img_size(0, 0)
        self.assertEqual(self.body_zero_size.get_img_size(), (0, 0))
        self.assertNotEqual(self.body, self.body_zero_size)
        self.assertFalse(self.body_zero_size._test_if_img_size_valid())

    def test_test_if_img_size_valid(self):
        self.assertTrue(self.body._test_if_img_size_valid())
        self.assertFalse(self.body_zero_size._test_if_img_size_valid())
        with self.assertRaises(ValueError):
            self.body_zero_size.get_lon_img()

    def test_disc_method(self):
        method = ' test method '
        self.body.set_disc_method(method)
        self.assertEqual(self.body.get_disc_method(), method)

        self.body._clear_cache()
        self.assertEqual(self.body.get_disc_method(), self.body._default_disc_method)

        self.body.set_disc_method(method)
        self.assertEqual(self.body.get_disc_method(), method)

        self.body.set_x0(123)
        self.assertEqual(self.body.get_disc_method(), self.body._default_disc_method)

    def test_add_arcsec_offset(self):
        self.body.set_disc_params(0, 0, 1, 0)
        self.body.add_arcsec_offset(0, 0)
        self.assertEqual(self.body.get_disc_params(), (0, 0, 1, 0))
        self.body.add_arcsec_offset(1, 2)
        self.assertTrue(
            np.allclose(
                self.body.get_disc_params(),
                (-0.05532064212457044, 0.11116537556358708, 1.0, 0.0),
            )
        )

    def test_img_limits(self):
        self.assertEqual(self.body.get_img_limits_xy(), ((-0.5, 14.5), (-0.5, 9.5)))
        self.assertTrue(
            np.allclose(
                self.body.get_img_limits_radec(),
                (
                    (196.38091225891438, 196.36417481895663),
                    (-5.571901975157448, -5.560796287842726),
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                self.body.get_img_limits_km(),
                (
                    (-151773.3647184372, 130762.09502601624),
                    (-125352.05899906158, 117394.22356271744),
                ),
            )
        )

    def test_limb_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertTrue(
            np.allclose(
                self.body.limb_xy(npts=5),
                (
                    array(
                        [
                            8.3280756,
                            -2.73574834,
                            -3.00515718,
                            7.49990606,
                            14.92008563,
                            8.3280756,
                        ]
                    ),
                    array(
                        [
                            16.74059437,
                            14.22970414,
                            2.77048972,
                            -1.2293739,
                            7.50713047,
                            16.74059437,
                        ]
                    ),
                ),
                equal_nan=True,
            )
        )

    def test_limb_xy_by_illumination(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertTrue(
            np.allclose(
                self.body.limb_xy_by_illumination(npts=5),
                (
                    array([8.3280756, -2.73574834, -3.00515718, nan, nan, 8.3280756]),
                    array(
                        [16.74059437, 14.22970414, 2.77048972, nan, nan, 16.74059437]
                    ),
                    array([nan, nan, nan, 7.49990606, 14.92008563, nan]),
                    array([nan, nan, nan, -1.2293739, 7.50713047, nan]),
                ),
                equal_nan=True,
            )
        )

    def test_termrinator_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertTrue(
            np.allclose(
                self.body.terminator_xy(npts=3),
                (
                    array([nan, nan, 11.14140527, nan]),
                    array([nan, nan, 0.48169876, nan]),
                ),
                equal_nan=True,
            )
        )

    def test_visible_lonlat_grid_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertTrue(
            np.allclose(
                self.body.visible_lonlat_grid_xy(interval=90, npts=3),
                [
                    (array([1.67619973, nan, nan]), array([-0.72952731, nan, nan])),
                    (
                        array([1.67619973, 13.41207875, nan]),
                        array([-0.72952731, 5.02509592, nan]),
                    ),
                    (
                        array([1.67619973, 0.92445441, nan]),
                        array([-0.72952731, 10.00171828, nan]),
                    ),
                    (array([1.67619973, nan, nan]), array([-0.72952731, nan, nan])),
                    (
                        array([1.67619973, 1.67619973, 1.67619973]),
                        array([-0.72952731, -0.72952731, -0.72952731]),
                    ),
                    (array([nan, 0.92445441, nan]), array([nan, 10.00171828, nan])),
                ],
                equal_nan=True,
            )
        )

    def test_ring_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertTrue(
            np.allclose(
                self.body.ring_xy(1234.5678, npts=4),
                (
                    array([nan, 5.09062199, 4.8390282, nan]),
                    array([nan, 7.97280096, 8.06177746, nan]),
                ),
                equal_nan=True,
            )
        )

    def test_plot_wireframe(self):
        fig, ax = plt.subplots()
        self.body.plot_wireframe_km(ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax, projection='orthographic', lat=56)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax, projection='azimuthal', lat=-90)
        plt.close(fig)

    def test_get_wireframe_overlay(self):
        img = self.body.get_wireframe_overlay_img(output_size=100)
        self.assertEqual(max(img.shape), 100)
        self.assertEqual(len(img.shape), 2)

        img = self.body.get_wireframe_overlay_map(output_size=100)
        self.assertEqual(max(img.shape), 100)
        self.assertEqual(len(img.shape), 2)

    def test_map_img(self):
        self.body.set_img_size(4, 3)
        self.body.set_disc_params(2, 1, 1.5, 45.678)

        image = np.array(
            [
                [-1.0, 2.2, 3.3, 4.4],
                [999.0, nan, 1.0, 1.0],
                [0.0, 3.0, 0.0, nan],
            ]
        )
        expected = {
            'nearest': array(
                [
                    [nan, nan, nan, 2.31866428, 2.74706312, 3.19651992, nan, nan],
                    [nan, nan, nan, nan, nan, 3.31150404, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.28286675, nan, nan, nan, nan, nan],
                ]
            ),
            'linear': array(
                [
                    [nan, nan, nan, 2.31866428, 2.74706312, 3.19651992, nan, nan],
                    [nan, nan, nan, nan, nan, 3.31150404, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.28286675, nan, nan, nan, nan, nan],
                ]
            ),
            'quadratic': array(
                [
                    [nan, nan, nan, 2.31866428, 2.74706312, 3.19651992, nan, nan],
                    [nan, nan, nan, nan, nan, 3.31150404, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.28286675, nan, nan, nan, nan, nan],
                ]
            ),
            'cubic': array(
                [
                    [nan, nan, nan, 2.31866428, 2.74706312, 3.19651992, nan, nan],
                    [nan, nan, nan, nan, nan, 3.31150404, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.28286675, nan, nan, nan, nan, nan],
                ]
            ),
        }
        for interpolation, expected_img in expected.items():
            with self.subTest(interpolation=interpolation):
                self.assertTrue(
                    np.allclose(
                        self.body.map_img(image, degree_interval=45),
                        expected_img,
                        equal_nan=True,
                    )
                )

        with self.assertRaises(ValueError):
            self.body.map_img(image, projection='manual')

        lons = np.linspace(-180, 180, 5)
        lats = np.linspace(0, 90, 3)

        for attempt in range(2):
            with self.subTest(attempt=attempt):
                # Test twice to check cache behaviour
                self.assertTrue(
                    np.allclose(
                        self.body.map_img(
                            image, projection='manual', lon_coords=lons, lat_coords=lats
                        ),
                        array(
                            [
                                [nan, nan, nan, 2.56786056, nan],
                                [0.27832292, nan, nan, nan, 0.27832292],
                                [nan, nan, nan, nan, nan],
                            ]
                        ),
                        equal_nan=True,
                    )
                )

        lons, lats = np.meshgrid(np.linspace(100, 250, 3), np.linspace(10, 80, 4))
        self.assertTrue(
            np.allclose(
                self.body.map_img(
                    image, projection='manual', lon_coords=lons, lat_coords=lats
                ),
                array(
                    [
                        [1.62335601, nan, nan],
                        [nan, nan, 2.74010963],
                        [nan, nan, nan],
                        [nan, nan, nan],
                    ]
                ),
                equal_nan=True,
            )
        )

        self.body.set_img_size(15, 10)

    def test_standardise_backplane_name(self):
        self.assertEqual(self.body.standardise_backplane_name('EMISSION'), 'EMISSION')
        self.assertEqual(self.body.standardise_backplane_name(' EMISSION '), 'EMISSION')
        self.assertEqual(self.body.standardise_backplane_name('emission'), 'EMISSION')
        self.assertEqual(self.body.standardise_backplane_name('EmIsSiOn'), 'EMISSION')

    def test_register_backplane(self):
        name = '<<<TEST>>>'
        description = 'A test backplane'
        get_img = lambda: None
        get_map = lambda: None

        self.body.register_backplane(
            name,
            description,
            get_img,  #  type: ignore
            get_map,  #  type: ignore
        )

        backplane = self.body.get_backplane(name)
        self.assertEqual(backplane.name, name)
        self.assertEqual(backplane.description, description)
        self.assertEqual(backplane.get_img, get_img)
        self.assertEqual(backplane.get_map, get_map)

        with self.assertRaises(ValueError):
            self.body.register_backplane(
                name,
                description,
                get_img=get_img,  #  type: ignore
                get_map=get_map,  #  type: ignore
            )

        del self.body.backplanes[name]

        with self.assertRaises(planetmapper.body_xy.BackplaneNotFoundError):
            self.body.get_backplane(name)

    def test_backplane_summary_string(self):
        lines = [
            'LON-GRAPHIC: Planetographic longitude, positive W [deg]',
            'LAT-GRAPHIC: Planetographic latitude [deg]',
            'LON-CENTRIC: Planetocentric longitude [deg]',
            'LAT-CENTRIC: Planetocentric latitude [deg]',
            'RA: Right ascension [deg]',
            'DEC: Declination [deg]',
            'PIXEL-X: Observation x pixel coordinate [pixels]',
            'PIXEL-Y: Observation y pixel coordinate [pixels]',
            'KM-X: East-West distance in target plane [km]',
            'KM-Y: North-South distance in target plane [km]',
            'PHASE: Phase angle [deg]',
            'INCIDENCE: Incidence angle [deg]',
            'EMISSION: Emission angle [deg]',
            'AZIMUTH: Azimuth angle [deg]',
            'DISTANCE: Distance to observer [km]',
            'RADIAL-VELOCITY: Radial velocity away from observer [km/s]',
            'DOPPLER: Doppler factor, sqrt((1 + v/c)/(1 - v/c)) where v is radial velocity',
            'RING-RADIUS: Equatorial (ring) plane radius [km]',
            'RING-LON-GRAPHIC: Equatorial (ring) plane planetographic longitude [deg]',
            'RING-DISTANCE: Equatorial (ring) plane distance to observer [km]',
        ]
        self.assertEqual(
            self.body.backplane_summary_string(),
            '\n'.join(lines),
        )

    def test_get_backplane(self):
        self.assertEqual(
            self.body.get_backplane(' emission '),
            Backplane(
                'EMISSION',
                'Emission angle [deg]',
                self.body.get_emission_angle_img,
                self.body.get_emission_angle_map,
            ),
        )
        with self.assertRaises(BackplaneNotFoundError):
            self.body.get_backplane('<test not a backplane>')

    def test_get_backplane_img(self):
        # Actual backplane contents tested against FITS outputs in test_observation
        self.body.set_img_size(4, 3)
        self.body.set_disc_params(2, 1, 1.5, 45.678)
        self.assertTrue(
            np.allclose(
                self.body.get_backplane_img(' emission '),
                array(
                    [
                        [nan, 86.56708848, 46.84006258, 72.67205499],
                        [nan, 42.68886971, 0.38721538, 42.52071712],
                        [nan, 72.63701695, 46.49373305, 86.56516607],
                    ]
                ),
                equal_nan=True,
            )
        )
        self.body.set_img_size(15, 10)

    def test_get_backplane_map(self):
        self.body.set_img_size(4, 3)
        self.body.set_disc_params(2, 1, 1.5, 45.678)
        self.assertTrue(
            np.allclose(
                self.body.get_backplane_map(' emission ', degree_interval=90),
                array(
                    [
                        [129.64320026, 75.34674827, 45.20593116, 100.74624309],
                        [134.80160102, 79.26258633, 50.36478231, 104.66172453],
                    ]
                ),
                equal_nan=True,
            )
        )
        self.body.set_img_size(15, 10)

    def test_plot_backplane(self):
        fig, ax = plt.subplots()
        self.body.plot_backplane_img(' emission ', ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_backplane_map(' emission ', degree_interval=90, ax=ax)
        plt.close(fig)

    def test_plot_map(self):
        fig, ax = plt.subplots()
        self.body.plot_map(np.ones((180, 360)), ax=ax)
        plt.close(fig)

    # Backplane contents tested against FITS reference in test_observation
