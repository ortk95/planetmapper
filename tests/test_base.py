import datetime
import unittest
from typing import Any, Callable, ParamSpec

import common_testing
import numpy as np
import spiceypy as spice

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper.base import (
    BodyBase,
    _cache_clearable_result,
    _cache_stable_result,
    _to_tuple,
)

P = ParamSpec('P')


class TestSpiceBase(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.obj = planetmapper.SpiceBase()

    def test_init(self):
        self.assertTrue(self.obj._optimize_speed)

    def test_repr(self):
        self.assertEqual(str(self.obj), 'SpiceBase()')

    def test_eq(self):
        self.assertEqual(self.obj, planetmapper.SpiceBase())
        self.assertNotEqual(self.obj, planetmapper.SpiceBase(optimize_speed=False))

    def test_hash(self):
        self.assertEqual(hash(self.obj), hash(planetmapper.SpiceBase()))
        # Hashes for unequal object can be the same, so don't do assertNotEqual here

    def test_get_kwargs(self):
        self.assertEqual(self.obj._get_kwargs(), {'optimize_speed': True})

    def test_standardise_body_name(self):
        self.assertEqual(self.obj.standardise_body_name('JUPITER'), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name(' JuPiTeR   '), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name('599'), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name(599), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name('HST'), 'HST')
        self.assertEqual(
            self.obj.standardise_body_name('Hubble Space Telescope'), 'HST'
        )

    def test_et2dtm(self):
        pairs = (
            (
                -999999999,
                datetime.datetime(
                    1968, 4, 24, 10, 12, 39, 814453, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                0,
                datetime.datetime(
                    2000, 1, 1, 11, 58, 55, 816073, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                42,
                datetime.datetime(
                    2000, 1, 1, 11, 59, 37, 816073, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                123456789,
                datetime.datetime(
                    2003, 11, 30, 9, 32, 4, 816943, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                0.123456789,
                datetime.datetime(
                    2000, 1, 1, 11, 58, 55, 939530, tzinfo=datetime.timezone.utc
                ),
            ),
        )
        for et, dtm in pairs:
            with self.subTest(f'et = {et}'):
                self.assertEqual(self.obj.et2dtm(et), dtm)

    def test_mjd2dtm(self):
        pairs = [
            (
                50000,
                datetime.datetime(1995, 10, 10, 0, 0, tzinfo=datetime.timezone.utc),
            ),
            (
                51234.56789,
                datetime.datetime(
                    1999, 2, 25, 13, 37, 45, 696000, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                60000.1,
                datetime.datetime(2023, 2, 25, 2, 24, tzinfo=datetime.timezone.utc),
            ),
        ]
        for mjd, dtm in pairs:
            with self.subTest(f'mjd = {mjd}'):
                self.assertEqual(self.obj.mjd2dtm(mjd), dtm)

    def test_speed_of_light(self):
        self.assertEqual(self.obj.speed_of_light(), 299792.458)

    def test_calculate_doppler_factor(self):
        pairs = [
            (0, 1),
            (12345.6789, 1.0420647220422994),
            (2e5, 2.2379273771294423),
            (self.obj.speed_of_light() * 0.9, 4.358898943540674),
        ]

        for rv, df in pairs:
            with self.subTest(f'rv = {rv}'):
                self.assertAlmostEqual(self.obj.calculate_doppler_factor(rv), df)

    def test_load_spice_kernels(self):
        self.assertTrue(planetmapper.base._KERNEL_DATA['kernels_loaded'])

    def test_close_loop(self):
        self.assertTrue(
            np.array_equal(
                self.obj.close_loop(np.array([0, 1, 2, 3, 4, 5])),
                np.array([0, 1, 2, 3, 4, 5, 0]),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.obj.close_loop(np.array([[1.1, 1.2], [2.2, 2.3]])),
                np.array([[1.1, 1.2], [2.2, 2.3], [1.1, 1.2]]),
            )
        )

    def test_unit_vector(self):
        a = np.random.rand(3) * 999
        ahat = self.obj.unit_vector(a)
        self.assertAlmostEqual(np.linalg.norm(ahat), 1)  #  type: ignore

    def test_vector_magnitude(self):
        pairs = [
            (np.array([1, 2, 3]), 3.7416573867739413),
            (np.array([-999]), 999),
            (np.array([-1.23, 4.56, 789]), 789.0141358049297),
            (np.array([0, 0, 0, 0]), 0),
            (np.array([0, 0, 0, 42]), 42),
        ]
        for v, magnitude in pairs:
            with self.subTest(v):
                self.assertAlmostEqual(self.obj.vector_magnitude(v), magnitude)
                self.assertAlmostEqual(
                    self.obj.vector_magnitude(v), np.linalg.norm(v)  # type: ignore
                )

        self.assertTrue(np.isnan(self.obj.vector_magnitude(np.array([1, np.nan]))))

    def test_encode_str(self):
        self.assertEqual(self.obj._encode_str('abc'), b'abc')

    def test_angle_conversion(self):
        pair = np.random.rand(2)
        self.assertTrue(
            np.array_equal(self.obj._radian_pair2degrees(*pair), np.rad2deg(pair))
        )
        self.assertTrue(
            np.array_equal(self.obj._degree_pair2radians(*pair), np.deg2rad(pair))
        )
        self.assertTrue(
            np.allclose(
                self.obj._degree_pair2radians(*self.obj._radian_pair2degrees(*pair)),
                pair,
            )
        )

    def test_rotation_matrix_radians(self):
        pairs = [
            (0, np.array([[1.0, 0.0], [-0.0, 1.0]])),
            (np.pi, np.array([[-1.0, -0.0], [0.0, -1.0]])),
            (1, np.array([[0.54030231, 0.84147098], [-0.84147098, 0.54030231]])),
            (
                -12345.6789,
                np.array([[0.71075274, 0.70344192], [-0.70344192, 0.71075274]]),
            ),
        ]

        for radians, matrix in pairs:
            with self.subTest(radians):
                self.assertTrue(
                    np.allclose(self.obj._rotation_matrix_radians(radians), matrix)
                )

    def test_angular_dist(self):
        pairs = [
            ((0, 0, 0, 0), 0),
            ((1, 2, 3, 4), 2.8264172166624126),
            ((-42, 0, 1234.5678, 99), 81.37656372202063),
        ]
        for angles, dist in pairs:
            with self.subTest(angles):
                self.assertAlmostEqual(self.obj.angular_dist(*angles), dist)
        self.assertTrue(np.isnan(self.obj.angular_dist(1, 2, 3, np.nan)))

    def test_progrress_hook(self):
        hook = planetmapper.progress.ProgressHook()
        self.obj._set_progress_hook(hook)
        self.assertEqual(self.obj._get_progress_hook(), hook)
        with self.assertRaises(NotImplementedError):
            self.obj._update_progress_hook(0.5)
        self.obj._remove_progress_hook()
        self.assertIsNone(self.obj._get_progress_hook())


class TestSpiceStringEncoding(unittest.TestCase):
    def setUp(self):
        planetmapper.SpiceBase.load_spice_kernels()
        self.obj = planetmapper.SpiceBase(optimize_speed=True)

    def compare_function_outputs(
        self, fn: Callable[P, Any], *args: P.args, **kw: P.kwargs
    ):
        assert len(kw) == 0
        string_functions = [
            lambda x: x,
            self.obj._encode_str,
        ]
        outputs = []
        for f in string_functions:
            outputs.append(
                fn(
                    *[f(a) if isinstance(a, str) else a for a in args],  # type: ignore
                )
            )
        self.assertEqual(len(outputs[0]), len(outputs[1]))
        for a, b in zip(outputs[0], outputs[1]):
            self.assertEqual(type(a), type(b))
            if isinstance(a, np.ndarray):
                self.assertTrue(np.array_equal(a, b))
            else:
                self.assertEqual(a, b)

    def test_spkezr(self):
        self.compare_function_outputs(
            spice.spkezr,
            'jupiter',
            0,
            'J2000',
            'CN+S',
            'earth',
        )

    def test_subpnt(self):
        self.compare_function_outputs(
            spice.subpnt,
            'INTERCEPT/ELLIPSOID',
            'jupiter',
            1000,
            'IAU_jupiter',
            'CN+S',
            'earth',
        )

    def test_pgrrec(self):
        self.compare_function_outputs(
            spice.pgrrec,
            'jupiter',
            0,
            0,
            0,
            100,
            0.99,
        )

    def test_pxfrm2(self):
        self.compare_function_outputs(
            spice.pxfrm2,
            'IAU_jupiter',
            'J2000',
            10000,
            11000,
        )

    def test_sincpt(self):
        self.compare_function_outputs(
            spice.sincpt,
            'ELLIPSOID',
            'jupiter',
            0,
            'IAU_jupiter',
            'CN+S',
            'earth',
            'J2000',
            np.array([6.25064696e08, 2.76557345e08, 1.03301984e08]),
        )

    def test_recpgr(self):
        self.compare_function_outputs(
            spice.recpgr,
            'jupiter',
            [0, 0, 1000],
            2000,
            0.9,
        )

    def test_immumf(self):
        self.compare_function_outputs(
            spice.illumf,
            'ELLIPSOID',
            'jupiter',
            'sun',
            90000,
            'IAU_jupiter',
            'CN+S',
            'earth',
            np.array([10000, 20000, 3000]),
        )

    def test_spkcpt(self):
        self.compare_function_outputs(
            spice.spkcpt,
            [0, 10000, 200000],
            'jupiter',
            'IAU_jupiter',
            99999,
            'J2000',
            'OBSERVER',
            'CN+S',
            'earth',
        )


class TestKernelPath(unittest.TestCase):
    def setUp(self) -> None:
        planetmapper.base._clear_kernels()

    def tearDown(self) -> None:
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        planetmapper.base._clear_kernels()

    def test_kernel_path(self):
        path = 'abcdef/ghi/jkl'
        planetmapper.set_kernel_path(path)
        self.assertEqual(planetmapper.get_kernel_path(), path)

        self.assertEqual(planetmapper.base.load_kernels(), [])

        planetmapper.set_kernel_path(None)
        self.assertEqual(
            planetmapper.get_kernel_path(), planetmapper.base.DEFAULT_KERNEL_PATH
        )

        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.assertEqual(planetmapper.get_kernel_path(), common_testing.KERNEL_PATH)


class TestBodyBase(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

    def test_init_target(self):
        kw = dict(
            utc='2005-01-01',
            observer='earth',
            aberration_correction='CN+S',
            observer_frame='J2000',
        )
        self.assertEqual(
            BodyBase(target='jupiter', **kw),
            BodyBase(target=599, **kw),
        )

        self.assertEqual(
            BodyBase(target='jupiter', **kw),
            BodyBase(target=' JuPiteR   ', **kw),
        )

    def test_init_utc(self):
        kw = dict(
            target='jupiter',
            observer='earth',
            aberration_correction='CN+S',
            observer_frame='J2000',
        )
        obj = BodyBase(utc='2005-01-01 12:00', **kw)
        self.assertEqual(obj.utc, '2005-01-01T12:00:00.000000')

        self.assertEqual(
            obj,
            BodyBase(utc=datetime.datetime(2005, 1, 1, 12), **kw),
        )
        self.assertEqual(
            obj,
            BodyBase(utc=53371.5, **kw),
        )

        self.assertEqual(
            obj,
            BodyBase(
                utc=datetime.datetime(
                    2005,
                    1,
                    1,
                    15,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=3)),
                ),
                **kw,
            ),
        )

    def test_eq(self):
        obj = BodyBase(
            target='jupiter',
            utc='2005-01-01',
            observer='earth',
            aberration_correction='CN+S',
            observer_frame='J2000',
        )
        self.assertEqual(
            obj,
            BodyBase(
                target='jupiter',
                utc='2005-01-01',
                observer='earth',
                aberration_correction='CN+S',
                observer_frame='J2000',
            ),
        )

        self.assertNotEqual(
            obj,
            BodyBase(
                target='jupiter',
                utc='2005-01-02',
                observer='earth',
                aberration_correction='CN+S',
                observer_frame='J2000',
            ),
        )

    def test_hash(self):
        obj = BodyBase(
            target='jupiter',
            utc='2005-01-01',
            observer='earth',
            aberration_correction='CN+S',
            observer_frame='J2000',
        )
        self.assertEqual(
            hash(obj),
            hash(
                BodyBase(
                    target='jupiter',
                    utc='2005-01-01',
                    observer='earth',
                    aberration_correction='CN+S',
                    observer_frame='J2000',
                )
            ),
        )


class TestCache(unittest.TestCase):
    def setUp(self):
        self._cache = {}
        self._stable_cache = {}
        self.functions_called = []

    @_cache_clearable_result
    def f_clearable(self, a, b=1):
        self.functions_called.append('f_clearable')
        return ('f_clearable', a * b)

    @_cache_stable_result
    def f_stable(self, a, b=1):
        self.functions_called.append('f_stable')
        return ('f_stable', a * b)

    def test_clearable_cache(self):
        self.functions_called = []
        for attempt in range(3):
            with self.subTest(attempt=attempt):
                self._cache.clear()
                self.functions_called = []

                self.assertEqual(self.f_clearable(1), ('f_clearable', 1))
                self.assertEqual(self.functions_called, ['f_clearable'])
                self.assertEqual(self.f_clearable(1), ('f_clearable', 1))
                self.assertEqual(self.functions_called, ['f_clearable'])

                self.assertEqual(self.f_clearable(2), ('f_clearable', 2))
                self.assertEqual(self.functions_called, ['f_clearable'] * 2)
                self.assertEqual(self.f_clearable(2), ('f_clearable', 2))
                self.assertEqual(self.functions_called, ['f_clearable'] * 2)

                self.assertEqual(self.f_clearable(2, b=2), ('f_clearable', 4))
                self.assertEqual(self.functions_called, ['f_clearable'] * 3)
                self.assertEqual(self.f_clearable(2, b=2), ('f_clearable', 4))
                self.assertEqual(self.functions_called, ['f_clearable'] * 3)

                self.assertEqual(self.f_clearable(1), ('f_clearable', 1))
                self.assertEqual(self.f_clearable(2), ('f_clearable', 2))
                self.assertEqual(self.f_clearable(2, b=2), ('f_clearable', 4))
                self.assertEqual(self.functions_called, ['f_clearable'] * 3)

                self.assertEqual(len(self._cache), 3)

    def test_stable_cache(self):
        self.functions_called = []

        self.assertEqual(self.f_stable(1), ('f_stable', 1))
        self.assertEqual(self.functions_called, ['f_stable'])
        self.assertEqual(self.f_stable(1), ('f_stable', 1))
        self.assertEqual(self.functions_called, ['f_stable'])

        self.assertEqual(self.f_stable(2), ('f_stable', 2))
        self.assertEqual(self.functions_called, ['f_stable'] * 2)
        self.assertEqual(self.f_stable(2), ('f_stable', 2))
        self.assertEqual(self.functions_called, ['f_stable'] * 2)

        self.assertEqual(self.f_stable(2, b=2), ('f_stable', 4))
        self.assertEqual(self.functions_called, ['f_stable'] * 3)
        self.assertEqual(self.f_stable(2, b=2), ('f_stable', 4))
        self.assertEqual(self.functions_called, ['f_stable'] * 3)

        self.assertEqual(self.f_stable(1), ('f_stable', 1))
        self.assertEqual(self.f_stable(2), ('f_stable', 2))
        self.assertEqual(self.f_stable(2, b=2), ('f_stable', 4))
        self.assertEqual(self.functions_called, ['f_stable'] * 3)

        self.assertEqual(len(self._stable_cache), 3)


class TestFunctions(unittest.TestCase):
    def test_to_tuple(self):
        pairs = [
            (np.array([1, 2, 3]), (1, 2, 3)),
            (np.array([[1, 2, 3]]), ((1, 2, 3),)),
            (np.array(1), 1.0),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(_to_tuple(a), b)
