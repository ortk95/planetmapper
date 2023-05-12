import datetime
import os
import unittest
import warnings

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy import array, nan

import planetmapper
from planetmapper import utils


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

    def test_format_radec_axes(self):
        fig, ax = plt.subplots()
        utils.format_radec_axes(ax, 45)
        self.assertAlmostEqual(ax.get_aspect(), 1 / 0.7071067811865476)  #  type: ignore
        self.assertTrue(ax.xaxis_inverted())
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.invert_xaxis()
        utils.format_radec_axes(
            ax, 0, dms_ticks=False, add_axis_labels=False, aspect_adjustable='box'
        )
        self.assertAlmostEqual(ax.get_aspect(), 1.0)  # type: ignore
        self.assertTrue(ax.xaxis_inverted())
        plt.close(fig)

        fig, ax = plt.subplots()
        utils.format_radec_axes(ax, 30)

        limits: list[tuple[float, float]] = [
            (-1, 1),
            (20.12, 20.123),
            (-42.01, -42.010001),
            (0.001, 0.005),
            (-20, 30.1),
            (42, 42.5),
        ]
        for limit in limits:
            ax.set_xlim(*limit)
            ax.set_ylim(*limit)
            fig.canvas.draw()
        plt.close(fig)

    def test_decimal_degrees_to_dms(self):
        pairs = [
            [0, (0, 0, 0)],
            [1, (1, 0, 0)],
            [1.23456789, (1, 14, 4.444404)],
            [-123.456, (-123, 27, 21.6)],
            [360, (360, 0, 0)],
            [-0.1, (0, -6, 0)],
            [-0.001, (0, 0, -3.6)],
        ]
        for decimal_degrees, dms in pairs:
            with self.subTest(decimal_degrees=decimal_degrees):
                d, m, s = utils.decimal_degrees_to_dms(decimal_degrees)
                self.assertEqual(d, dms[0])
                self.assertEqual(m, dms[1])
                self.assertAlmostEqual(s, dms[2])

    def test_decimal_degrees_to_dms_str(self):
        pairs = [
            [0, '0°0′0.0000″'],
            [1, '1°0′0.0000″'],
            [1.23456789, '1°14′4.4444″'],
            [-123.456, '-123°27′21.6000″'],
            [360, '360°0′0.0000″'],
        ]
        for decimal_degrees, dms_str in pairs:
            with self.subTest(decimal_degrees=decimal_degrees):
                self.assertEqual(
                    utils.decimal_degrees_to_dms_str(
                        decimal_degrees, seconds_fmt='.4f'
                    ),
                    dms_str,
                )

    def test_ignore_warnings(self):
        warning_string1 = 'test warning string'
        warning_string2 = 'test warning string 2'

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with self.assertWarns(UserWarning):
                warnings.warn(warning_string1, UserWarning)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with utils.ignore_warnings(warning_string1):
                warnings.warn(warning_string1, UserWarning)

            with utils.ignore_warnings(warning_string1, warning_string2):
                warnings.warn(warning_string1, UserWarning)
                warnings.warn(warning_string1, UserWarning)

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with self.assertWarns(UserWarning):
                warnings.warn(warning_string1, UserWarning)

    def test_filter_fits_comment_warning(self):
        card = (
            'KEY',
            'value',
            'A very very long comment that will create a warning because the card is too long',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with self.assertWarns(UserWarning):
                header = fits.Header()
                header.append(card)
                header.tostring()

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with utils.filter_fits_comment_warning():
                header = fits.Header()
                header.append(card)
                header.tostring()

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with self.assertWarns(UserWarning):
                header = fits.Header()
                header.append(card)
                header.tostring()

    def test_normalise(self):
        pairs = [
            [[1, 2, 3], array([0.0, 0.5, 1.0])],
            [[[nan, -99], [1.23, 45.6]], array([[nan, 0.0], [0.69315353, 1.0]])],
            [[1, 1, 1], array([0, 0, 0])],
        ]
        for a, b in pairs:
            with self.subTest(a):
                self.assertTrue(np.allclose(utils.normalise(a), b, equal_nan=True))

        self.assertTrue(
            np.allclose(
                utils.normalise([1, 1, 1], single_value=42),
                np.array([42, 42, 42]),
                equal_nan=True,
            )
        )

    def test_check_path(self):
        path = os.path.join(
            common_testing.TEMP_PATH,
            datetime.datetime.now().strftime('TEST1_%Y%m%d%H%M%S%f'),
        )
        utils.check_path(path)
        self.assertTrue(os.path.exists(path))
        utils.check_path(path)
        os.rmdir(path)

        path = os.path.join(
            common_testing.TEMP_PATH,
            datetime.datetime.now().strftime('TEST2_%Y%m%d%H%M%S%f'),
        )
        utils.check_path(os.path.join(path, 'test_file.txt'))
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)

        utils.check_path('')
