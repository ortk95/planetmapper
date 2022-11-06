import unittest
import planetmapper
import numpy as np
import spiceypy as spice
from typing import Callable, ParamSpec, Any

P = ParamSpec('P')


class TestForConsistentResults(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
