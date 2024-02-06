import os
import unittest

import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
KERNEL_PATH = os.path.join(DATA_PATH, 'kernels')
TEMP_PATH = os.path.join(os.path.dirname(__file__), 'temp')


class BaseTestCase(unittest.TestCase):
    def assertArraysEqual(
        self, a: np.ndarray, b: np.ndarray, *, equal_nan: bool = False
    ) -> None:
        self.assertTrue(
            np.array_equal(a, b, equal_nan=equal_nan),
            msg=f'Arrays not equal:\n{a}\n{b}',
        )

    def assertArraysClose(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> None:
        self.assertTrue(
            np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan),
            msg=f'Arrays not close:\n{a}\n{b}',
        )
