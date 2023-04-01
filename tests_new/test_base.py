import unittest
import planetmapper
import datetime
import numpy as np
import spiceypy as spice
from typing import Callable, ParamSpec, Any
import common_testing
P = ParamSpec('P')


class TestSpiceBase(unittest.TestCase):
    def setUp(self):
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.obj = planetmapper.SpiceBase()

    def test_init(self):
        pass
        # TODO

    def test_repr(self):
        pass # TODO

    def test_standardise_body_name(self):
        self.assertEqual(self.obj.standardise_body_name(' JuPiTeR   '), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name('599'), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name(599), 'JUPITER')

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
        pass  # TODO

    def test_speed_of_light(self):
        self.assertEqual(self.obj.speed_of_light(), 299792.458)

    def test_calculate_doppler_factor(self):
        pass  # TODO

    def test_load_spice_kernels(self):
        pass  # TODO

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
        self.assertAlmostEqual(np.linalg.norm(ahat), 1)

    def test_vector_magnitude(self):
        pass  # TODO

    def test_encode_str(self):
        pass  # TODO

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

    def test_rotation_matric_radians(self):
        pass  # TODO

    def test_angular_dist(self):
        pass  # TODO

    def test_progrress_hook(self):
        pass


class TestKernelPath(unittest.TestCase):
    def test_kernel_path(self):
        pass  # TODO

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
            'saturn',
            1000,
            'IAU_SATURN',
            'CN+S',
            'earth',
        )

    def test_pgrrec(self):
        self.compare_function_outputs(
            spice.pgrrec,
            'uranus',
            0,
            0,
            0,
            100,
            0.99,
        )

    def test_pxfrm2(self):
        self.compare_function_outputs(
            spice.pxfrm2,
            'IAU_neptune',
            'J2000',
            10000,
            11000,
        )

    def test_sincpt(self):
        self.compare_function_outputs(
            spice.sincpt,
            'ELLIPSOID',
            'pluto',
            0,
            'IAU_pluto',
            'CN+S',
            'earth',
            'J2000',
            np.array([-1.45130504e09, -4.31817467e09, -9.18250174e08]),
        )

    def test_recpgr(self):
        self.compare_function_outputs(
            spice.recpgr,
            'mercury',
            [0, 0, 1000],
            2000,
            0.9,
        )

    def test_immumf(self):
        self.compare_function_outputs(
            spice.illumf,
            'ELLIPSOID',
            'venus',
            'sun',
            90000,
            'IAU_venus',
            'CN+S',
            'earth',
            np.array([10000, 20000, 3000]),
        )

    def test_spkcpt(self):
        self.compare_function_outputs(
            spice.spkcpt,
            [0, 10000, 200000],
            'mars',
            'IAU_mars',
            99999,
            'J2000',
            'OBSERVER',
            'CN+S',
            'earth',
        )
