import os
import unittest
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
KERNEL_PATH = os.path.join(DATA_PATH, 'kernels')
TEMP_PATH = os.path.join(os.path.dirname(__file__), 'temp')


class BaseTestCase(unittest.TestCase):
    def assertArraysEqual(
        self,
        a: Sequence | np.ndarray,
        b: Sequence | np.ndarray,
        *,
        equal_nan: bool = False,
    ) -> None:
        self.assertTrue(
            np.array_equal(a, b, equal_nan=equal_nan),
            msg=f'Arrays not equal:\n{a!r}\n{b!r}',
        )

    def assertArraysClose(
        self,
        a: Sequence | np.ndarray,
        b: Sequence | np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> None:
        diff = np.abs(np.array(a) - np.array(b))
        aerr = np.nanmax(diff)
        relerr = aerr / np.nanmax(np.abs(b))
        self.assertTrue(
            np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan),
            msg=f'Arrays not close (a={aerr:.2e}, r={relerr:.2e}):\n{a!r}\n{b!r}',
        )

    def _test_wireframe_scaling(
        self,
        wireframe_func: Callable,
        xlabel: str,
        ylabel: str,
        scalings: list[tuple[float | None, tuple[float, float], tuple[float, float]]],
        *,
        atol: float = 1e-5,
        rtol: float = 1e-2,
    ) -> None:
        scale_factor, xlim, ylim = next(
            scaling for scaling in scalings if scaling[0] is None
        )
        scalings.append((1, xlim, ylim))
        scalings.append((1.0, xlim, ylim))

        with self.subTest(wireframe_func=wireframe_func, scale_factor='<default>'):
            fig, ax = plt.subplots()
            wireframe_func(ax)
            self.assertArraysClose(ax.get_xlim(), xlim, atol=atol, rtol=rtol)
            self.assertArraysClose(ax.get_ylim(), ylim, atol=atol, rtol=rtol)
            self.assertEqual(ax.get_xlabel(), xlabel)
            self.assertEqual(ax.get_ylabel(), ylabel)
            plt.close(fig)

        for scale_factor, xlim, ylim in scalings:
            with self.subTest(wireframe_func=wireframe_func, scale_factor=scale_factor):
                fig, ax = plt.subplots()
                wireframe_func(ax, scale_factor=scale_factor)
                self.assertArraysClose(ax.get_xlim(), xlim, atol=atol, rtol=rtol)
                self.assertArraysClose(ax.get_ylim(), ylim, atol=atol, rtol=rtol)
                if scale_factor is None:
                    self.assertEqual(ax.get_xlabel(), xlabel)
                    self.assertEqual(ax.get_ylabel(), ylabel)
                else:
                    self.assertEqual(ax.get_xlabel(), '')
                    self.assertEqual(ax.get_ylabel(), '')
                plt.close(fig)

                for add_axis_labels in (True, False):
                    with self.subTest(
                        wireframe_func=wireframe_func,
                        scale_factor=scale_factor,
                        add_axis_labels=add_axis_labels,
                    ):
                        fig, ax = plt.subplots()
                        wireframe_func(
                            ax,
                            scale_factor=scale_factor,
                            add_axis_labels=add_axis_labels,
                        )
                        self.assertArraysClose(
                            ax.get_xlim(), xlim, atol=atol, rtol=rtol
                        )
                        self.assertArraysClose(
                            ax.get_ylim(), ylim, atol=atol, rtol=rtol
                        )
                        if add_axis_labels:
                            self.assertEqual(ax.get_xlabel(), xlabel)
                            self.assertEqual(ax.get_ylabel(), ylabel)
                        else:
                            self.assertEqual(ax.get_xlabel(), '')
                            self.assertEqual(ax.get_ylabel(), '')
                        plt.close(fig)
