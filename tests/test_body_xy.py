from unittest.mock import MagicMock, patch

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from numpy import array, inf, nan

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import BodyXY
from planetmapper.body_xy import Backplane, BackplaneNotFoundError
from planetmapper.body_xy import _MapKwargs as MapKwargs


class TestFunctions(common_testing.BaseTestCase):
    def test_make_backplane_documentation_str(self):
        self.assertIsInstance(
            planetmapper.body_xy._make_backplane_documentation_str(), str
        )

    def test_extract_map_kwargs_from_dict(self):
        pairs: list[tuple[dict, tuple[dict, dict]]] = [
            (
                {},
                ({}, {}),
            ),
            (
                {'a': 1},
                ({}, {'a': 1}),
            ),
            ({'projection': 'orthographic'}, ({'projection': 'orthographic'}, {})),
            (
                {'projection': 'orthographic', 'a': 1},
                ({'projection': 'orthographic'}, {'a': 1}),
            ),
            (
                {'projection': 'orthographic', 'a': 1, 'b': 2},
                ({'projection': 'orthographic'}, {'a': 1, 'b': 2}),
            ),
            (
                {'projection': 'orthographic', 'a': 1, 'b': 2, 'xlim': (0, 1)},
                ({'projection': 'orthographic', 'xlim': (0, 1)}, {'a': 1, 'b': 2}),
            ),
            (
                {
                    'projection': 'orthographic',
                    'color': 'r',
                    'alpha': 0.5,
                    'xlim': (0, 1),
                },
                (
                    {'projection': 'orthographic', 'xlim': (0, 1)},
                    {
                        'color': 'r',
                        'alpha': 0.5,
                    },
                ),
            ),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    planetmapper.body_xy._extract_map_kwargs_from_dict(a), b
                )


class TestBodyXY(common_testing.BaseTestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.body = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
        )
        self.body_zero_size = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
        )

    def test_get_default_init_kwargs(self):
        self._test_get_default_init_kwargs(
            BodyXY, target='jupiter', utc='2005-01-01T00:00:00'
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
            "BodyXY('JUPITER', '2005-01-01T00:00:00.000000', observer='HST', nx=15, ny=10)",
        )
        self.assertEqual(
            repr(self.body_zero_size),
            "BodyXY('JUPITER', '2005-01-01T00:00:00.000000', observer='HST', nx=0, ny=0)",
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
        # xy, radec, lonlat, km, angular
        coordinates = [
            [
                (0, 0),
                (196.3684350770821, -5.581107015413806),
                (nan, nan),
                (-43515.54503863168, -220566.4464649765),
                (12.721709080506116, -55.12740601573759),
            ],
            [
                (5, 8),
                (196.37198562427025, -5.565793847134351),
                (153.1235185909613, -3.0887371238645795),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
            [
                (4.1, 7.1),
                (196.37198562427025, -5.567914131973045),
                (164.3872136538264, -28.87847195832716),
                (-12411.924521414994, -27675.679236383432),
                (0.0, -7.633025448335383),
            ],
            [
                (1.234, 5.678),
                (196.37369462098349, -5.572965121633222),
                (nan, nan),
                (-64181.931835415264, -83648.1756567178),
                (-6.1233826374518685, -25.81658829413859),
            ],
            [
                (-3, 25),
                (196.40157351750477, -5.555192422940882),
                (nan, nan),
                (-322324.8112312332, 310766.23675694194),
                (-106.01424233789203, 38.16512724167089),
            ],
            [
                (7.9, 5.1),
                (196.36512123303984, -5.565793847134351),
                (nan, nan),
                (89177.18865054459, -39993.979013437434),
                (24.59530422240732, 0.0),
            ],
        ]

        for body in (self.body_zero_size, self.body):
            body.set_disc_params(5, 8, 3, 45)
            for xy, radec, lonlat, km, angular in coordinates:
                with self.subTest(xy=xy, body=body, func='xy2radec'):
                    self.assertArraysClose(body.xy2radec(*xy), radec, equal_nan=True)
                with self.subTest(xy=xy, body=body, func='xy2lonlat'):
                    self.assertArraysClose(body.xy2lonlat(*xy), lonlat, equal_nan=True)
                with self.subTest(xy=xy, body=body, func='xy2km'):
                    self.assertArraysClose(
                        body.xy2km(*xy), km, equal_nan=True, atol=1e-3
                    )
                with self.subTest(xy=xy, body=body, func='xy2angular'):
                    self.assertArraysClose(
                        body.xy2angular(*xy), angular, equal_nan=True, atol=1e-5
                    )
                with self.subTest(xy=xy, body=body, func='radec2xy'):
                    self.assertArraysClose(
                        body.radec2xy(*radec), xy, equal_nan=True, atol=1e-3
                    )
                with self.subTest(xy=xy, body=body, func='lonlat2xy'):
                    if not any(np.isnan(lonlat)):
                        self.assertArraysClose(
                            body.lonlat2xy(*lonlat), xy, equal_nan=True, atol=1e-3
                        )
                with self.subTest(xy=xy, body=body, func='km2xy'):
                    self.assertArraysClose(
                        body.km2xy(*km), xy, equal_nan=True, atol=1e-3
                    )
                with self.subTest(xy=xy, body=body, func='angular2xy'):
                    self.assertArraysClose(
                        body.angular2xy(*angular), xy, equal_nan=True, atol=1e-3
                    )

        args = [
            (np.nan, np.nan),
            (np.nan, 0),
            (0, np.nan),
            (np.inf, np.inf),
        ]
        for a in args:
            with self.subTest(a):
                self.assertTrue(not all(np.isfinite(self.body.xy2radec(*a))))
                self.assertTrue(not all(np.isfinite(self.body.xy2lonlat(*a))))
                self.assertTrue(not all(np.isfinite(self.body.xy2km(*a))))
                self.assertTrue(not all(np.isfinite(self.body.radec2xy(*a))))
                self.assertTrue(not all(np.isfinite(self.body.lonlat2xy(*a))))
                self.assertTrue(not all(np.isfinite(self.body.km2xy(*a))))

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

        with self.assertRaises(ValueError):
            self.body.set_r0(-1.23)

        self.body.set_plate_scale_arcsec(1)
        self.assertAlmostEqual(self.body.get_plate_scale_arcsec(), 1)
        self.assertAlmostEqual(self.body.get_r0(), 17.99121344984809)

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
        self.assertArraysClose(self.body.get_disc_params(), (0, 0, 1, 0))
        self.body.add_arcsec_offset(1, 2)
        self.assertTrue(
            np.allclose(
                self.body.get_disc_params(),
                (-0.05532064212457044, 0.11116537556358708, 1.0, 0.0),
            )
        )

    def test_img_limits(self):
        self.assertEqual(
            self.body.get_img_limits_xy(),
            (
                (-0.5, 14.5),
                (-0.5, 9.5),
            ),
        )
        self.assertArraysClose(
            self.body.get_img_limits_radec(),
            (
                (196.38091225891438, 196.36417481895663),
                (-5.571901975157448, -5.560796287842726),
            ),
        )
        self.assertArraysClose(
            self.body.get_img_limits_km(),
            (
                (-151724.69753899056, 130727.50016257458),
                (-125236.31445765976, 117241.42226096484),
            ),
        )
        self.assertArraysClose(
            self.body.get_img_limits_angular(),
            (
                (-31.984379466325663, 27.98633203326517),
                (-21.98926088314898, 17.99121344984992),
            ),
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

    def test_terminator_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertArraysClose(
            self.body.terminator_xy(npts=3),
            (
                array([nan, nan, 11.14140527, nan]),
                array([nan, nan, 0.48169876, nan]),
            ),
            equal_nan=True,
            atol=1e-3,
        )

    def test_visible_lonlat_grid_xy(self):
        self.body.set_disc_params(5, 8, 10, 45)
        self.assertArraysClose(
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
            atol=1e-3,
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

    @patch('matplotlib.pyplot.show')
    def test_plot_wireframe(self, mock_show: MagicMock):
        fig, ax = plt.subplots()
        self.body.plot_wireframe_xy(ax=ax)
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 32)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_wireframe_xy(ax=ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        fig, ax = plt.subplots()
        self.body_zero_size.plot_wireframe_xy(ax=ax)
        plt.close(fig)

        ax = self.body.plot_map_wireframe()
        self.assertEqual(ax.get_xlim(), (360, 0))
        self.assertEqual(len(ax.get_lines()), 16)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 26)
        plt.close('all')

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax, color='r', zorder=2, alpha=0.5)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax)
        self.assertEqual(ax.get_aspect(), 1)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax, aspect_adjustable=None)
        self.assertEqual(ax.get_aspect(), 'auto')
        plt.close(fig)

        uranus = BodyXY('uranus', utc='2000-01-01', sz=5)  # Uranus is +ve E
        ax = uranus.plot_map_wireframe(ax=ax)
        self.assertEqual(ax.get_xlim(), (0, 360))
        plt.close('all')

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(ax=ax, projection='orthographic', lat=56)
        self.assertEqual(len(ax.get_lines()), 18)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 29)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(
            ax=ax, projection='azimuthal', lat=-90, label_poles=False, grid_interval=45
        )
        self.assertEqual(len(ax.get_lines()), 20)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 30)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(
            ax=ax,
            projection='azimuthal equal area',
            lat=-90,
            label_poles=False,
            grid_interval=45,
        )
        self.assertEqual(len(ax.get_lines()), 20)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 30)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(
            ax=ax,
            projection='azimuthal equal area',
            lat=-90,
            label_poles=False,
            grid_interval=45,
            color='r',
        )
        self.assertEqual(len(ax.get_lines()), 20)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 30)
        plt.close(fig)

        # backwards compatibility
        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(
            ax=ax,
            projection='azimuthal equal area',
            lat=-90,
            label_poles=False,
            grid_interval=45,
            common_formatting=dict(color='r'),
        )
        self.assertEqual(len(ax.get_lines()), 20)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 30)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map_wireframe(
            ax=ax,
            projection='manual',
            lon_coords=np.linspace(-180, 180, 5),
            lat_coords=np.linspace(0, 90, 3),
        )
        self.assertEqual(len(ax.get_lines()), 17)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 29)
        plt.close(fig)

        self._test_wireframe_scaling(
            self.body.plot_wireframe_xy,
            'x (pixels)',
            'y (pixels)',
            [
                (None, (-0.5, 14.5), (-0.5, 9.5)),
                (
                    1,
                    (2.6023360665508823, 12.397667359351928),
                    (0.3152347289140154, 9.684782827744533),
                ),
                (
                    1.0,
                    (2.6023360665508823, 12.397667359351928),
                    (0.3152347289140154, 9.684782827744533),
                ),
                (
                    50,
                    (130.11680332754412, 619.8833679675964),
                    (15.761736445700745, 484.2391413872267),
                ),
                (
                    123456.786,
                    (321276.04686825396, 1530576.166082696),
                    (38917.86646730556, 1195652.1610213316),
                ),
                (
                    1e-06,
                    (2.6023360665508817e-06, 1.239766735935193e-05),
                    (3.152347289140152e-07, 9.684782827744531e-06),
                ),
            ],
        )

    def test_get_wireframe_overlay(self):
        img = self.body.get_wireframe_overlay_img(output_size=100)
        self.assertEqual(max(img.shape), 100)
        self.assertEqual(len(img.shape), 2)

        img = self.body.get_wireframe_overlay_map(output_size=100)
        self.assertEqual(max(img.shape), 100)
        self.assertEqual(len(img.shape), 2)

        img = self.body.get_wireframe_overlay_map(output_size=100, rgba=True)
        self.assertEqual(max(img.shape), 100)
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 4)

    @patch('builtins.print')
    def test_map_img(self, mock_print: MagicMock):
        self.body.set_img_size(4, 3)
        self.body.set_disc_params(2, 1, 1.5, 45.678)

        image = np.array(
            [
                [-1.0, 2.2, 3.3, 4.4],
                [999.0, nan, 1.0, 123.456789],
                [0.0, 3.0, 0.0, nan],
                [0.0, 3.0, 0.1, nan],
            ]
        )
        expected_interpolations: dict[str | int | tuple[int, int], np.ndarray] = {
            'nearest': array(
                [
                    [nan, nan, 2.2, 2.2, 2.2, 3.3, nan, nan],
                    [nan, nan, nan, nan, 1.0, 4.4, nan, nan],
                    [nan, nan, 3.0, 3.0, 0.0, 123.456789, nan, nan],
                    [nan, nan, 0.0, 0.0, 0.0, nan, nan, nan],
                ]
            ),
            'linear': array(
                [
                    [nan, nan, nan, 2.31866371, 2.74706025, 3.19651445, nan, nan],
                    [nan, nan, nan, nan, nan, 23.42819481, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.3201943, nan, nan, nan, nan, nan],
                ]
            ),
            'quadratic': array(
                [
                    [nan, nan, nan, 2.39879966, 2.87923688, 3.21452692, nan, nan],
                    [nan, nan, nan, nan, nan, 43.60950368, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 2.51093265, nan, nan, nan, nan, nan],
                ]
            ),
            'cubic': array(
                [
                    [nan, nan, nan, 2.38239724, 2.87854034, 3.22915024, nan, nan],
                    [nan, nan, nan, nan, nan, 75.57633703, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 5.81872817, nan, nan, nan, nan, nan],
                ]
            ),
            (1, 2): array(
                [
                    [nan, nan, nan, 2.39879966, 2.87923688, 3.21452692, nan, nan],
                    [nan, nan, nan, nan, nan, 24.05570999, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.27193437, nan, nan, nan, nan, nan],
                ]
            ),
        }
        # Check aliases
        expected_interpolations[1] = expected_interpolations['linear']
        expected_interpolations[2] = expected_interpolations['quadratic']
        expected_interpolations[3] = expected_interpolations['cubic']
        expected_interpolations[(1, 1)] = expected_interpolations['linear']
        expected_interpolations[(2, 2)] = expected_interpolations['quadratic']
        expected_interpolations[(3, 3)] = expected_interpolations['cubic']
        for interpolation, expected_img in expected_interpolations.items():
            with self.subTest(interpolation=interpolation):
                self.assertArraysClose(
                    self.body.map_img(
                        image,
                        degree_interval=45,
                        interpolation=interpolation,  # type: ignore
                    ),
                    expected_img,
                    equal_nan=True,
                )

        # Check smoothing
        expected_smoothings: dict[float, np.ndarray] = {
            0: array(
                [
                    [nan, nan, nan, 2.31866371, 2.74706025, 3.19651445, nan, nan],
                    [nan, nan, nan, nan, nan, 23.42819481, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.3201943, nan, nan, nan, nan, nan],
                ]
            ),
            1: array(
                [
                    [nan, nan, nan, 2.31766083, 2.74652865, 3.19647729, nan, nan],
                    [nan, nan, nan, nan, nan, 23.42144429, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.32130192, nan, nan, nan, nan, nan],
                ]
            ),
            2.345: array(
                [
                    [nan, nan, nan, 2.31716984, 2.74626094, 3.19644382, nan, nan],
                    [nan, nan, nan, nan, nan, 23.41785299, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.32187724, nan, nan, nan, nan, nan],
                ]
            ),
            67.89: array(
                [
                    [nan, nan, nan, 2.31348483, 2.74377072, 3.19520713, nan, nan],
                    [nan, nan, nan, nan, nan, 23.37223821, nan, nan],
                    [nan, nan, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 0.3283601, nan, nan, nan, nan, nan],
                ]
            ),
        }
        for smoothing, expected_img in expected_smoothings.items():
            with self.subTest(smoothing=smoothing):
                self.assertArraysClose(
                    self.body.map_img(
                        image,
                        degree_interval=45,
                        spline_smoothing=smoothing,
                    ),
                    expected_img,
                    equal_nan=True,
                )

        self.body.map_img(
            image, interpolation='linear', degree_interval=45, warn_nan=True
        )
        mock_print.assert_called_once()
        mock_print.reset_mock()

        with self.assertRaises(ValueError):
            self.body.map_img(image, interpolation='<<<test>>>')  # type: ignore

        with self.assertRaises(ValueError):
            self.body.map_img(image, projection='manual')

        lons = np.linspace(-180, 180, 5)
        lats = np.linspace(0, 90, 3)
        image = np.array(
            [
                [-1.0, 2.2, 3.3, 4.4],
                [999.0, nan, 1.0, 123.456789],
                [0.0, 3.0, 0.0, nan],
            ]
        )
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                # Test twice to check cache behaviour
                self.assertArraysClose(
                    self.body.map_img(
                        image, projection='manual', lon_coords=lons, lat_coords=lats
                    ),
                    array(
                        [
                            [nan, nan, nan, 68.55480206, nan],
                            [0.27833474, nan, nan, nan, 0.27833474],
                            [nan, nan, nan, nan, nan],
                        ]
                    ),
                    equal_nan=True,
                )

        lons, lats = np.meshgrid(np.linspace(100, 250, 3), np.linspace(10, 80, 4))
        self.assertArraysClose(
            self.body.map_img(
                image, projection='manual', lon_coords=lons, lat_coords=lats
            ),
            array(
                [
                    [101.62835326, nan, nan],
                    [nan, nan, 2.74012498],
                    [nan, nan, nan],
                    [nan, nan, nan],
                ]
            ),
            equal_nan=True,
        )

        self.body.set_img_size(15, 10)

    def test_generate_map_coordinates(self):
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(projection='manual')
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'manual',
                lon_coords=np.array([1, 2, 3]),
                lat_coords=np.array([[1, 2, 3], [4, 5, 6]]),
            )
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'manual',
                lon_coords=np.array([[[1, 2, 3]]]),
                lat_coords=np.array([[[1, 2, 3]]]),
            )
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'manual',
                lon_coords=np.array([[1, 2, 3]]),
                lat_coords=np.array([[1, 2, 3], [4, 5, 6]]),
            )
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates('proj=ortho +R=1 +type=crs')
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'proj=ortho +R=1 +type=crs',
                projection_x_coords=np.array([1, 2, 3]),
                projection_y_coords=np.array([[1, 2, 3], [4, 5, 6]]),
            )
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'proj=ortho +R=1 +type=crs',
                projection_x_coords=np.array([[[1, 2, 3]]]),
            )
        with self.assertRaises(ValueError):
            self.body.generate_map_coordinates(
                'proj=ortho +R=1 +type=crs',
                projection_x_coords=np.array([[1, 2, 3]]),
                projection_y_coords=np.array([[1, 2, 3], [4, 5, 6]]),
            )
        output_a = self.body.generate_map_coordinates(
            '+proj=ortho +R=1 +type=crs', projection_x_coords=np.array([0, 0.25, 0.5])
        )
        output_b = self.body.generate_map_coordinates(
            '+proj=ortho +R=1 +type=crs',
            projection_x_coords=np.array([0, 0.25, 0.5]),
            projection_y_coords=np.array([0, 0.25, 0.5]),
        )
        for idx, (a, b) in enumerate(zip(output_a, output_b)):
            if idx == 5:
                # info dict with projection_y_coords = None
                self.assertEqual(a['projection_y_coords'], None)  # type: ignore
                a['projection_y_coords'] = b['projection_y_coords']  # type: ignore
            with self.subTest(idx=idx):
                self.assertEqual(type(a), type(b))
                if isinstance(a, np.ndarray):
                    self.assertTrue(np.array_equal(a, b))  # type: ignore
                else:
                    self.assertEqual(a, b)

        # Test limits
        output_a = self.body.generate_map_coordinates(degree_interval=30)
        output_b = self.body.generate_map_coordinates(
            degree_interval=30, xlim=None, ylim=None
        )
        for idx, (a, b) in enumerate(zip(output_a, output_b)):
            with self.subTest(idx=idx):
                self.assertEqual(type(a), type(b))
                if isinstance(a, np.ndarray):
                    self.assertTrue(np.array_equal(a, b))
                else:
                    self.assertEqual(a, b)

        args_limit: list[
            tuple[
                tuple[float, float] | None,
                tuple[float, float] | None,
                np.ndarray,
                np.ndarray,
            ]
        ] = [
            (
                None,
                None,
                array([[315.0, 225.0, 135.0, 45.0], [315.0, 225.0, 135.0, 45.0]]),
                array([[-45.0, -45.0, -45.0, -45.0], [45.0, 45.0, 45.0, 45.0]]),
            ),
            (
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                array([[315.0, 225.0, 135.0, 45.0], [315.0, 225.0, 135.0, 45.0]]),
                array([[-45.0, -45.0, -45.0, -45.0], [45.0, 45.0, 45.0, 45.0]]),
            ),
            (
                (135, -np.inf),
                (45, np.inf),
                array([[135.0, 45.0]]),
                array([[45.0, 45.0]]),
            ),
            (
                (100, 300),
                (-50, 50),
                array([[225.0, 135.0], [225.0, 135.0]]),
                array([[-45.0, -45.0], [45.0, 45.0]]),
            ),
            (
                (300, 100),
                (50, -50),
                array([[225.0, 135.0], [225.0, 135.0]]),
                array([[-45.0, -45.0], [45.0, 45.0]]),
            ),
        ]
        for xlim, ylim, lons_expected, lats_expected in args_limit:
            with self.subTest(xlim=xlim, ylim=ylim):
                (
                    lons,
                    lats,
                    xx,
                    yy,
                    transformer,
                    info,
                ) = self.body.generate_map_coordinates(
                    degree_interval=90, xlim=xlim, ylim=ylim
                )
                self.assertTrue(
                    np.array_equal(lons, lons_expected),
                    msg=f'{lons} <> {lons_expected}',
                )
                self.assertTrue(np.array_equal(lats, lats_expected), msg=repr(lats))
                self.assertTrue(np.array_equal(xx, lons_expected), msg=repr(xx))
                self.assertTrue(np.array_equal(yy, lats_expected), msg=repr(yy))
                self.assertEqual(info['xlim'], xlim, msg=repr(info))
                self.assertEqual(info['ylim'], ylim, msg=repr(info))

        # Test reference
        args: list[tuple[MapKwargs, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
            (
                MapKwargs(degree_interval=123),
                array([[307.5, 184.5, 61.5]]),
                array([[-28.5, -28.5, -28.5]]),
                array([[307.5, 184.5, 61.5]]),
                array([[-28.5, -28.5, -28.5]]),
            ),
            (
                MapKwargs(projection='orthographic', size=5),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, 36.87110893, 0.0, -36.87110893, nan],
                        [nan, 30.33135236, 0.0, -30.33135236, nan],
                        [nan, 36.87110893, 0.0, -36.87110893, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, -34.45624462, -34.45624462, -34.45624462, nan],
                        [nan, 0.0, 0.0, 0.0, nan],
                        [nan, 34.45624462, 34.45624462, 34.45624462, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01, -1.01],
                        [-0.505, -0.505, -0.505, -0.505, -0.505],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.505, 0.505, 0.505, 0.505, 0.505],
                        [1.01, 1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
            (
                MapKwargs(projection='orthographic', size=5, lon=123.456, lat=-2),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, 161.19011383, 123.456, 85.72188617, nan],
                        [nan, 153.80492624, 123.456, 93.10707376, nan],
                        [nan, 159.53178271, 123.456, 87.38021729, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, -36.20674821, -36.65376937, -36.20674821, nan],
                        [nan, -1.98332476, -2.29643357, -1.98332476, nan],
                        [nan, 32.67332417, 32.24176455, 32.67332417, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01, -1.01],
                        [-0.505, -0.505, -0.505, -0.505, -0.505],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.505, 0.505, 0.505, 0.505, 0.505],
                        [1.01, 1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
            (
                MapKwargs(projection='azimuthal', size=4),
                array(
                    [
                        [nan, nan, nan, nan],
                        [nan, 83.93213465, -83.93213465, nan],
                        [nan, 83.93213465, -83.93213465, nan],
                        [nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan],
                        [nan, -44.83904649, -44.83904649, nan],
                        [nan, 44.83904649, 44.83904649, nan],
                        [nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01],
                        [-0.33666667, -0.33666667, -0.33666667, -0.33666667],
                        [0.33666667, 0.33666667, 0.33666667, 0.33666667],
                        [1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
            (
                MapKwargs(projection='azimuthal', size=4, lat=90, lon=123.456),
                array(
                    [
                        [nan, nan, nan, nan],
                        [nan, 168.456, 78.456, nan],
                        [nan, -101.544, -11.544, nan],
                        [nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan],
                        [nan, 4.29865812, 4.29865812, nan],
                        [nan, 4.29865812, 4.29865812, nan],
                        [nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                        [-1.01, -0.33666667, 0.33666667, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01],
                        [-0.33666667, -0.33666667, -0.33666667, -0.33666667],
                        [0.33666667, 0.33666667, 0.33666667, 0.33666667],
                        [1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
            (
                MapKwargs(projection='azimuthal equal area', size=5),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, 91.6285626, 0.0, -91.6285626, nan],
                        [nan, 60.66270473, 0.0, -60.66270473, nan],
                        [nan, 91.6285626, 0.0, -91.6285626, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, -44.98842597, -60.66270473, -44.98842597, nan],
                        [nan, 0.0, 0.0, 0.0, nan],
                        [nan, 44.98842597, 60.66270473, 44.98842597, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01, -1.01],
                        [-0.505, -0.505, -0.505, -0.505, -0.505],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.505, 0.505, 0.505, 0.505, 0.505],
                        [1.01, 1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
            (
                MapKwargs(projection='azimuthal equal area', size=5, lat=-12, lon=34),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, 137.26373836, 34.0, -69.26373836, nan],
                        [nan, 95.20027738, 34.0, -27.20027738, nan],
                        [nan, 113.79039062, 34.0, -45.79039062, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [nan, nan, nan, nan, nan],
                        [nan, -43.4196019, -72.66270473, -43.4196019, nan],
                        [nan, -5.84665238, -12.0, -5.84665238, nan],
                        [nan, 44.08255341, 48.66270473, 44.08255341, nan],
                        [nan, nan, nan, nan, nan],
                    ]
                ),
                array(
                    [
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                        [-1.01, -0.505, 0.0, 0.505, 1.01],
                    ]
                ),
                array(
                    [
                        [-1.01, -1.01, -1.01, -1.01, -1.01],
                        [-0.505, -0.505, -0.505, -0.505, -0.505],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.505, 0.505, 0.505, 0.505, 0.505],
                        [1.01, 1.01, 1.01, 1.01, 1.01],
                    ]
                ),
            ),
        ]
        for kwargs, lons_expected, lats_expected, xx_expected, yy_expected in args:
            with self.subTest(kwargs=kwargs):
                (
                    lons,
                    lats,
                    xx,
                    yy,
                    transformer,
                    info,
                ) = self.body.generate_map_coordinates(**kwargs)
                self.assertTrue(
                    np.allclose(lons, lons_expected, equal_nan=True), msg=repr(lons)
                )
                self.assertTrue(
                    np.allclose(lats, lats_expected, equal_nan=True), msg=repr(lats)
                )
                self.assertTrue(np.allclose(xx, xx_expected), msg=repr(xx))
                self.assertTrue(np.allclose(yy, yy_expected), msg=repr(yy))

    def test_create_proj_string(self):
        jupiter = BodyXY(
            'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
        )
        earth = BodyXY('Earth', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10)
        self.assertEqual(
            jupiter.create_proj_string('ortho'), '+proj=ortho +axis=wnu +type=crs'
        )
        self.assertEqual(
            earth.create_proj_string('ortho'), '+proj=ortho +axis=enu +type=crs'
        )
        self.assertEqual(
            jupiter.create_proj_string('ortho', axis=None), '+proj=ortho +type=crs'
        )
        self.assertEqual(
            earth.create_proj_string('ortho', axis=None), '+proj=ortho +type=crs'
        )
        self.assertEqual(
            jupiter.create_proj_string('ortho', axis='123'),
            '+proj=ortho +axis=123 +type=crs',
        )
        self.assertEqual(
            earth.create_proj_string('ortho', axis='123'),
            '+proj=ortho +axis=123 +type=crs',
        )
        self.assertEqual(
            jupiter.create_proj_string(
                'eqc', string='a_string', number=123, lat_0=-1.234
            ),
            '+proj=eqc +string=a_string +number=123 +lat_0=-1.234 +axis=wnu +type=crs',
        )

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
            'LOCAL-SOLAR-TIME: Local solar time [local hours]',
            'DISTANCE: Distance to observer [km]',
            'RADIAL-VELOCITY: Radial velocity away from observer [km/s]',
            'DOPPLER: Doppler factor, sqrt((1 + v/c)/(1 - v/c)) where v is radial velocity',
            'LIMB-DISTANCE: Distance above limb [km]',
            'LIMB-LON-GRAPHIC: Planetographic longitude of closest point on the limb [deg]',
            'LIMB-LAT-GRAPHIC: Planetographic latitude of closest point on the limb [deg]',
            'RING-RADIUS: Equatorial (ring) plane radius [km]',
            'RING-LON-GRAPHIC: Equatorial (ring) plane planetographic longitude [deg]',
            'RING-DISTANCE: Equatorial (ring) plane distance to observer [km]',
        ]
        self.assertEqual(
            self.body.backplane_summary_string(),
            '\n'.join(lines),
        )

    @patch('builtins.print')
    def test_print_backplanes(self, mock_print: MagicMock):
        self.body.print_backplanes()
        mock_print.assert_called_once_with(self.body.backplane_summary_string())

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

    @patch('matplotlib.pyplot.show')
    def test_plot_backplane(self, mock_show: MagicMock):
        fig, ax = plt.subplots()
        self.body.plot_backplane_img(' emission ', ax=ax)
        self.assertEqual(len(ax.get_lines()), 21)
        self.assertEqual(len(ax.get_images()), 1)
        self.assertEqual(len(ax.get_children()), 33)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_backplane_img(' EmissioN ', ax=ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        fig, ax = plt.subplots()
        self.body.plot_backplane_map(' emission ', ax=ax, show=True)
        plt.close(fig)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        ax = self.body.plot_backplane_map(' emission ', degree_interval=90)
        self.assertEqual(len(ax.get_lines()), 16)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 27)
        plt.close('all')

        fig, ax = plt.subplots()
        self.body.plot_backplane_map(
            ' emission ',
            degree_interval=90,
            ax=ax,
            cmap='Blues',
            wireframe_kwargs=dict(color='r', zorder=2, alpha=0.5),
        )
        self.assertEqual(len(ax.get_lines()), 16)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 27)
        plt.close(fig)

        # back compatibility
        fig, ax = plt.subplots()
        self.body.plot_backplane_map(
            ' emission ',
            degree_interval=90,
            ax=ax,
            plot_kwargs=dict(
                cmap='Blues',
                wireframe_kwargs=dict(color='r', zorder=2, alpha=0.5),
            ),
        )
        self.assertEqual(len(ax.get_lines()), 16)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 27)
        plt.close(fig)

        plt.close('all')

    def test_plot_map(self):
        fig, ax = plt.subplots()
        h = self.body.plot_map(np.ones((180, 360)), ax=ax)
        self.assertIsInstance(h, QuadMesh)
        children = ax.get_children()
        self.assertIn(h, ax.get_children())
        self.assertEqual(len(ax.get_lines()), 16)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 27)
        plt.close(fig)

        fig, ax = plt.subplots()
        h = self.body.plot_map(
            np.ones((180, 360)), ax=ax, wireframe_kwargs=dict(grid_lat_limit=30)
        )
        self.assertIsInstance(h, QuadMesh)
        children = ax.get_children()
        self.assertIn(h, ax.get_children())
        self.assertEqual(len(ax.get_lines()), 14)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 25)
        plt.close(fig)

        fig, ax = plt.subplots()
        h = self.body.plot_map(np.ones((180, 360)), ax=ax, add_wireframe=False)
        children = ax.get_children()
        self.assertIn(h, ax.get_children())
        self.assertEqual(len(ax.get_lines()), 0)
        self.assertEqual(len(ax.get_images()), 0)
        self.assertEqual(len(ax.get_children()), 11)
        plt.close(fig)

        fig, ax = plt.subplots()
        self.body.plot_map(
            np.ones((180, 360)),
            ax=ax,
            wireframe_kwargs=dict(color='r', zorder=2, alpha=0.5),
        )
        plt.close(fig)

        self.body.imshow_map(np.ones((180, 360)))
        plt.close('all')

    def test_matplotlib_transforms(self):
        self.body.set_disc_params(2, 1, 3.5, 45.678)
        self.body.set_img_size(15, 10)

        # Test outputs
        self.assertArraysClose(
            self.body.matplotlib_radec2xy_transform().get_matrix(),
            array(
                [
                    [-4.87436799e02, 5.01041734e02, 9.85096272e04],
                    [4.98267132e02, 4.89321885e02, -9.51212414e04],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        )

        self.assertArraysClose(
            self.body.matplotlib_xy2radec_transform().get_matrix(),
            array(
                [
                    [-1.00236708e-03, 1.02637498e-03, 1.96372964e02],
                    [1.02153611e-03, 9.97641401e-04, -5.56883456e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        )
        self.assertArraysClose(
            self.body.matplotlib_km2xy_transform().get_matrix(),
            array(
                [
                    [4.55428642e-05, 1.79607788e-05, 2.00000000e00],
                    [-1.79607814e-05, 4.55428570e-05, 1.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        )
        self.assertArraysClose(
            self.body.matplotlib_xy2km_transform().get_matrix(),
            array(
                [
                    [1.90019906e04, -7.49383091e03, -3.05101503e04],
                    [7.49383091e03, 1.90019906e04, -3.39896524e04],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        )
        self.assertArraysClose(
            self.body.matplotlib_xy2angular_transform().get_matrix(),
            array(
                [
                    [3.59150906, -3.67753003, -3.50548809],
                    [3.67753003, 3.59150906, -10.94656911],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertArraysClose(
            self.body.matplotlib_angular2xy_transform().get_matrix(),
            array(
                [
                    [0.13592275, 0.13917826, 2.0],
                    [-0.13917826, 0.13592275, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

        # Test caching works
        for transform_method, attr in [
            (
                self.body._get_matplotlib_angular_fixed2xy_transform,
                '_mpl_transform_angular_fixed2xy',
            ),
            (
                self.body._get_matplotlib_xy2angular_fixed_transform,
                '_mpl_transform_xy2angular_fixed',
            ),
        ]:
            with self.subTest(transform=transform_method, attr=attr):
                t0 = transform_method()
                t1 = transform_method()
                setattr(self.body, attr, None)
                t2 = transform_method()
                self.assertIs(t0, t1)
                self.assertIsNot(t1, t2)
                self.assertEqual(t1, t2)

        # Test inverse
        pairs = [
            (
                self.body.matplotlib_radec2xy_transform(),
                self.body.matplotlib_xy2radec_transform(),
            ),
            (
                self.body.matplotlib_km2xy_transform(),
                self.body.matplotlib_xy2km_transform(),
            ),
            (
                self.body.matplotlib_angular2xy_transform(),
                self.body.matplotlib_xy2angular_transform(),
            ),
        ]
        for t1, t2 in pairs:
            with self.subTest(t1=t1, t2=t2):
                self.assertArraysClose(
                    t1.inverted().get_matrix(), t2.get_matrix(), atol=1e-6, rtol=1e-3
                )
                self.assertArraysClose(
                    t2.inverted().get_matrix(), t1.get_matrix(), atol=1e-6, rtol=1e-3
                )

        # Test update
        for transform_method in [
            self.body.matplotlib_radec2xy_transform,
            self.body.matplotlib_xy2radec_transform,
            self.body.matplotlib_km2xy_transform,
            self.body.matplotlib_xy2km_transform,
            self.body.matplotlib_angular2xy_transform,
            self.body.matplotlib_xy2angular_transform,
        ]:
            with self.subTest(transform_method=transform_method):
                transform = transform_method()
                self.body.set_disc_params(10, 9, 8, 7)
                self.body.update_transform()
                m1 = transform.get_matrix()
                self.body.set_disc_params(1.2, 3.4, 5.6, 178.9)
                self.assertTrue(np.array_equal(m1, transform.get_matrix()))
                self.body.update_transform()
                self.assertFalse(np.array_equal(m1, transform.get_matrix()))

        # Test passive update when getting 'new' transform
        # https://github.com/ortk95/planetmapper/issues/310
        for transform_method in [
            self.body.matplotlib_radec2xy_transform,
            self.body.matplotlib_xy2radec_transform,
            self.body.matplotlib_km2xy_transform,
            self.body.matplotlib_xy2km_transform,
            self.body.matplotlib_angular2xy_transform,
            self.body.matplotlib_xy2angular_transform,
        ]:
            self.body.set_disc_params(10, 9, 8, 7)
            m1 = transform_method().get_matrix()
            self.body.set_disc_params(1.2, 3.4, 5.6, 178.9)
            m2 = transform_method().get_matrix()
            self.body.set_disc_params(10, 9, 8, 7)
            m3 = transform_method().get_matrix()

            self.assertFalse(np.array_equal(m1, m2))
            self.assertTrue(np.array_equal(m1, m3))

    # Backplane contents tested against FITS reference in test_observation
