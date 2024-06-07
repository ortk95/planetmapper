import fnmatch
import os
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import common_testing
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import Observation


class TestObservation(common_testing.BaseTestCase):
    def setUp(self) -> None:
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)
        self.path = os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
        self.observation = Observation(self.path)

    def test_init(self) -> None:
        with self.assertRaises(ValueError):
            Observation()
        with self.assertRaises(ValueError):
            Observation('some/path', data=np.ones((5, 5)))
        with self.assertRaises(ValueError):
            Observation('some/path', header=fits.Header({'key': 'value'}))
        with self.assertRaises(ValueError):
            Observation(
                'some/path', data=np.ones((5, 5)), header=fits.Header({'key': 'value'})
            )

        with self.assertRaises(TypeError):
            Observation(self.path, nx=1)
        with self.assertRaises(TypeError):
            Observation(self.path, ny=1)
        with self.assertRaises(TypeError):
            Observation(self.path, sz=1)

        with self.subTest('image.png+target+observer+utc'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'image.png')
            obs = Observation(
                path,
                target='Jupiter',
                observer='hst',
                utc='2005-01-01T00:00:00',
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.header['OBJECT'], 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header['DATE-OBS'], '2005-01-01T00:00:00.000000')
            self.assertTrue(np.array_equal(obs.data, 100 * np.ones((4, 10, 5))))

        with self.subTest('planmap.fits'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'planmap.fits')
            obs = Observation(path)
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T12:00:00.000000')
            self.assertTrue(
                np.array_equal(
                    obs.data,
                    np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
                )
            )
            self.assertAlmostEqual(obs.get_x0(), 1.1)
            self.assertAlmostEqual(obs.get_y0(), 2.2)
            self.assertAlmostEqual(obs.get_r0(), 3.3)
            self.assertAlmostEqual(obs.get_rotation(), 4.4)

        with self.subTest('Path(planmap.fits)'):
            path = Path(common_testing.DATA_PATH, 'inputs', 'planmap.fits')
            obs = Observation(path)
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, os.fspath(path))
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T12:00:00.000000')
            self.assertTrue(
                np.array_equal(
                    obs.data,
                    np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
                )
            )
            self.assertAlmostEqual(obs.get_x0(), 1.1)
            self.assertAlmostEqual(obs.get_y0(), 2.2)
            self.assertAlmostEqual(obs.get_r0(), 3.3)
            self.assertAlmostEqual(obs.get_rotation(), 4.4)

        with self.subTest('planmap.fits+override'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'planmap.fits')
            obs = Observation(path, observer='EARTH', utc='2005-01-01')
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'EARTH')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertTrue(
                np.array_equal(
                    obs.data,
                    np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
                )
            )
            self.assertAlmostEqual(obs.get_x0(), 1.1)
            self.assertAlmostEqual(obs.get_y0(), 2.2)
            self.assertAlmostEqual(obs.get_r0(), 3.3)
            self.assertAlmostEqual(obs.get_rotation(), 4.4)

        with self.subTest('wcs.fits'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'wcs.fits')
            obs = Observation(path)
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertAlmostEqual(obs.get_x0(), 198.87871682168858, delta=0.2)
            self.assertAlmostEqual(obs.get_y0(), -31.89770255438151, delta=0.2)
            self.assertAlmostEqual(obs.get_r0(), 164.4473594677842, delta=0.2)
            self.assertAlmostEqual(obs.get_rotation(), 260.32237572846986, delta=0.2)

        with self.subTest('extended.fits'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'extended.fits')
            obs = Observation(path)
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertEqual(obs.path, path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T12:00:00.000000')
            self.assertTrue(
                np.array_equal(
                    obs.data,
                    np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
                )
            )

        with self.subTest('data+target+observer+utc'):
            data = np.ones((5, 6, 7))
            obs = Observation(
                data=data,
                target='Jupiter',
                observer='hst',
                utc='2005-01-01T00:00:00',
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('data+header+target+observer+utc'):
            data = np.ones((5, 6, 7))
            header = fits.Header({'key': 'value'})
            obs = Observation(
                data=data,
                header=header,
                target='Jupiter',
                observer='hst',
                utc='2005-01-01T00:00:00',
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header, header)
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('image+header'):
            data = np.ones((5, 6))
            header = fits.Header(
                {'OBJECT': 'jupiter', 'TELESCOP': 'HST', 'DATE-OBS': '2005-01-01'}
            )
            obs = Observation(
                data=data,
                header=header,
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header, header)
            data_expected = data[np.newaxis, :, :]
            self.assertTrue(np.array_equal(obs.data, data_expected))

        with self.subTest('data+header'):
            data = np.ones((5, 6, 7))
            header = fits.Header(
                {'OBJECT': 'jupiter', 'TELESCOP': 'HST', 'DATE-OBS': '2005-01-01'}
            )
            obs = Observation(
                data=data,
                header=header,
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header, header)
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('data+header+mix'):
            data = np.ones((5, 6, 7))
            header = fits.Header({'OBJECT': 'jupiter', 'DATE-OBS': '2005-01-01'})
            obs = Observation(
                data=data,
                header=header,
                observer='HST',
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header, header)
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('data+header+override'):
            data = np.ones((5, 6, 7))
            header = fits.Header(
                {'OBJECT': 'mars', 'TELESCOP': 'HST', 'DATE-OBS': '2005-01-01'}
            )
            obs = Observation(
                data=data,
                header=header,
                target='jupiter',
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T00:00:00.000000')
            self.assertEqual(obs.header, header)
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('data+header (DATE-OBS + TIME-OBS)'):
            data = np.ones((5, 6, 7))
            header = fits.Header(
                {
                    'OBJECT': 'jupiter',
                    'TELESCOP': 'HST',
                    'DATE-OBS': '2005-01-01',
                    'TIME-OBS': '12:34',
                }
            )
            obs = Observation(
                data=data,
                header=header,
            )
            self.assertEqual(obs, obs)
            self.assertNotEqual(obs, self.observation)
            self.assertIsNone(obs.path)
            self.assertEqual(obs.target, 'JUPITER')
            self.assertEqual(obs.observer, 'HST')
            self.assertEqual(obs.utc, '2005-01-01T12:34:00.000000')
            self.assertEqual(obs.header, header)
            self.assertTrue(np.array_equal(obs.data, data))

        with self.subTest('empty.fits'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', 'empty.fits')
            with self.assertRaises(ValueError):
                Observation(path)

        with self.subTest('2d_image.fits (including MJD avg.)'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', '2d_image.fits')
            obs = Observation(path)
            self.assertTrue(
                np.array_equal(obs.data, np.array([[[1.0, 2.0], [3.0, 4.0]]]))
            )
            self.assertEqual(obs.utc, '2000-01-01T12:00:00.000000')

        with self.subTest('2d_image.png'):
            path = os.path.join(common_testing.DATA_PATH, 'inputs', '2d_image.png')
            obs = Observation(path, target='JUPITER', utc='2000-01-01')
            self.assertTrue(np.array_equal(obs.data, np.array([[[1, 2], [3, 4]]])))

    def test_attributes(self):
        self.assertEqual(self.observation.path, self.path)
        self.assertEqual(self.observation.target, 'JUPITER')
        self.assertEqual(self.observation.observer, 'HST')
        self.assertEqual(self.observation.utc, '2005-01-01T00:00:00.000000')
        self.assertEqual(self.observation._nx, 7)
        self.assertEqual(self.observation._ny, 10)

    def test_repr(self):
        self.assertEqual(
            repr(self.observation),
            f"Observation({self.path!r}, target='JUPITER', utc='2005-01-01T00:00:00.000000', observer='HST')",
        )

    def test_to_body_xy(self):
        observation = Observation(
            data=np.ones((6, 5)),
            target='Jupiter',
            observer='HST',
            utc='2005-01-01T00:00:00',
        )
        observation.add_other_bodies_of_interest('amalthea')
        observation.coordinates_of_interest_lonlat.append((0, 0))
        observation.coordinates_of_interest_radec.extend([(0, 0), (1, 1)])

        body_xy = observation.to_body_xy()
        self.assertEqual(
            body_xy,
            planetmapper.BodyXY(
                'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=5, ny=6
            ),
        )

        self.assertEqual(observation.target, body_xy.target)
        self.assertEqual(observation.utc, body_xy.utc)
        self.assertEqual(observation.observer, body_xy.observer)
        self.assertEqual(observation.get_img_size(), body_xy.get_img_size())

        self.assertEqual(
            observation.coordinates_of_interest_lonlat,
            body_xy.coordinates_of_interest_lonlat,
        )
        self.assertEqual(
            observation.coordinates_of_interest_radec,
            body_xy.coordinates_of_interest_radec,
        )
        self.assertEqual(observation.ring_radii, body_xy.ring_radii)

        observation.coordinates_of_interest_radec.clear()
        self.assertNotEqual(
            observation.coordinates_of_interest_radec,
            body_xy.coordinates_of_interest_radec,
        )

    def test_hash(self):
        with self.assertRaises(TypeError):
            hash(self.observation)

    def test_eq(self):
        self.assertEqual(self.observation, self.observation)
        self.assertEqual(self.observation, Observation(self.path))

    def test_copy(self):
        copy = self.observation.copy()
        self.assertEqual(copy.path, self.observation.path)
        self.assertEqual(repr(copy), repr(self.observation))
        self.assertEqual(copy, self.observation)
        self.assertEqual(copy.get_img_size(), self.observation.get_img_size())

    def test_set_img_size(self):
        with self.assertRaises(TypeError):
            self.observation.set_img_size()
        with self.assertRaises(TypeError):
            self.observation.set_img_size(1)
        with self.assertRaises(TypeError):
            self.observation.set_img_size(2, 3)

    def test_disc_from_header(self):
        with self.assertRaises(ValueError):
            self.observation.disc_from_header()

        path = os.path.join(common_testing.DATA_PATH, 'inputs', 'planmap.fits')
        obs = Observation(path)
        self.assertAlmostEqual(obs.get_x0(), 1.1)
        self.assertAlmostEqual(obs.get_y0(), 2.2)
        self.assertAlmostEqual(obs.get_r0(), 3.3)
        self.assertAlmostEqual(obs.get_rotation(), 4.4)

        obs.set_disc_params(0, 0, 1, 0)
        self.assertEqual(obs.get_disc_params(), (0, 0, 1, 0))

        obs.disc_from_header()
        self.assertAlmostEqual(obs.get_x0(), 1.1)
        self.assertAlmostEqual(obs.get_y0(), 2.2)
        self.assertAlmostEqual(obs.get_r0(), 3.3)
        self.assertAlmostEqual(obs.get_rotation(), 4.4)

        data = np.ones((5, 6, 7))
        header = fits.Header(
            {
                'OBJECT': 'jupiter',
                'DATE-OBS': '2005-01-01',
                'HIERARCH PLANMAP DEGREE-INTERVAL': 1,
            }
        )
        obs = Observation(data=data, header=header)
        with self.assertRaises(ValueError):
            obs.disc_from_header()

        del header['HIERARCH PLANMAP DEGREE-INTERVAL']
        header['HIERARCH PLANMAP MAP PROJECTION'] = 'rectangular'
        obs = Observation(data=data, header=header)
        with self.assertRaises(ValueError):
            obs.disc_from_header()

        header['HIERARCH PLANMAP DISC X0'] = 1
        header['HIERARCH PLANMAP DISC Y0'] = 2
        header['HIERARCH PLANMAP DISC R0'] = 3
        header['HIERARCH PLANMAP DISC ROT'] = 4
        obs = Observation(data=data, header=header)
        with self.assertRaises(ValueError):
            obs.disc_from_header()

        del header['HIERARCH PLANMAP MAP PROJECTION']
        obs = Observation(data=data, header=header)
        obs.disc_from_header()
        self.assertAlmostEqual(obs.get_x0(), 1)
        self.assertAlmostEqual(obs.get_y0(), 2)
        self.assertAlmostEqual(obs.get_r0(), 3)
        self.assertAlmostEqual(obs.get_rotation(), 4)

    def test_stuff_from_wcs(self):
        with self.assertRaises(ValueError):
            self.observation.disc_from_wcs(suppress_warnings=True)
        with self.assertRaises(ValueError):
            self.observation.position_from_wcs(suppress_warnings=True)
        with self.assertRaises(ValueError):
            self.observation.rotation_from_wcs(suppress_warnings=True)
        with self.assertRaises(ValueError):
            self.observation.plate_scale_from_wcs(suppress_warnings=True)

        x0 = 198.87871682168858
        y0 = -31.89770255438151
        r0 = 164.4473594677842
        rotation = 260.32237572846986

        path = os.path.join(common_testing.DATA_PATH, 'inputs', 'wcs.fits')
        obs = Observation(path)
        self.assertTrue(
            np.allclose(obs.get_disc_params(), (x0, y0, r0, rotation), atol=0.2)
        )

        obs.set_disc_params(0, 0, 1, 0)
        self.assertEqual(obs.get_disc_params(), (0, 0, 1, 0))

        obs.disc_from_wcs(suppress_warnings=True)
        self.assertEqual(obs.get_disc_method(), 'wcs')
        self.assertTrue(
            np.allclose(obs.get_disc_params(), (x0, y0, r0, rotation), atol=0.2)
        )

        obs.set_disc_params(0, 0, 1, 0)
        obs.position_from_wcs(suppress_warnings=True)
        self.assertEqual(obs.get_disc_method(), 'wcs_position')
        self.assertAlmostEqual(obs.get_x0(), x0, delta=0.2)
        self.assertAlmostEqual(obs.get_y0(), y0, delta=0.2)

        obs.set_disc_params(0, 0, 1, 0)
        obs.rotation_from_wcs(suppress_warnings=True)
        self.assertEqual(obs.get_disc_method(), 'wcs_rotation')
        self.assertAlmostEqual(obs.get_rotation(), rotation, delta=0.2)

        obs.set_disc_params(0, 0, 1, 0)
        obs.plate_scale_from_wcs(suppress_warnings=True)
        self.assertEqual(obs.get_disc_method(), 'wcs_plate_scale')
        self.assertAlmostEqual(obs.get_r0(), r0, delta=0.2)

        data = np.ones((5, 6, 7))
        header = fits.Header(
            {
                'OBJECT': 'jupiter',
                'DATE-OBS': '2005-01-01',
                'CRPIX1': 1,
                'CRPIX2': 1,
                'CRVAL1': 196.3,
                'CRVAL2': -5.5,
                'CDELT1': 1,
                'CDELT2': 1,
                'CTYPE1': 'RA---TAN',
                'CTYPE2': 'DEC--TAN',
                'CUNIT1': 'deg',
                'CUNIT2': 'deg',
            }
        )
        obs = Observation(data=data, header=header)
        obs.disc_from_wcs(suppress_warnings=True)

        x0_before = obs.get_x0()
        y0_before = obs.get_y0()

        data = np.ones((5, 6, 7))
        h2 = header.copy()
        h2['HIERARCH NAV RA_OFFSET'] = 1
        h2['HIERARCH NAV DEC_OFFSET'] = -2.5
        obs = Observation(data=data, header=h2)
        obs.disc_from_wcs(suppress_warnings=True)
        self.assertNotEqual(obs.get_x0(), x0_before)
        self.assertNotEqual(obs.get_y0(), y0_before)

        obs.add_arcsec_offset(-1, 2.5)  # undo the header offsets
        self.assertAlmostEqual(obs.get_x0(), x0_before, delta=0.2)
        self.assertAlmostEqual(obs.get_y0(), y0_before, delta=0.2)

        obs.disc_from_wcs(suppress_warnings=True)
        self.assertNotEqual(obs.get_x0(), x0_before)
        self.assertNotEqual(obs.get_y0(), y0_before)

        obs.disc_from_wcs(suppress_warnings=True, use_header_offsets=False)
        self.assertAlmostEqual(obs.get_x0(), x0_before, delta=0.2)
        self.assertAlmostEqual(obs.get_y0(), y0_before, delta=0.2)

        h2 = header.copy()
        h2['CTYPE1'] = 'DEC--TAN'
        obs = Observation(data=data, header=h2)
        with self.assertRaises(ValueError):
            obs.disc_from_wcs(suppress_warnings=True)

        h2 = header.copy()
        h2['A_ORDER'] = 2
        h2['B_ORDER'] = 2
        for n in range(3):
            for m in range(3):
                h2[f'A_{n}_{m}'] = 0.1
                h2[f'B_{n}_{m}'] = 0.2
        obs = Observation(data=data, header=h2)
        with self.assertRaises(ValueError):
            obs.disc_from_wcs(suppress_warnings=True)
        obs.disc_from_wcs(validate=False, suppress_warnings=True)

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            with self.assertWarns(AstropyWarning):
                obs.disc_from_wcs(validate=False, suppress_warnings=False)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            obs.position_from_wcs(validate=False, suppress_warnings=True)

    def test_wcs_offset(self):
        with self.assertRaises(ValueError):
            self.observation.get_wcs_offset(suppress_warnings=True)
        with self.assertRaises(ValueError):
            self.observation.get_wcs_arcsec_offset(suppress_warnings=True)

        x0 = 198.87871682168858
        y0 = -31.89770255438151
        r0 = 164.4473594677842
        rotation = 260.32237572846986

        path = os.path.join(common_testing.DATA_PATH, 'inputs', 'wcs.fits')
        obs = Observation(path)

        obs.disc_from_wcs(suppress_warnings=True)
        self.assertTrue(
            np.allclose(obs.get_disc_params(), (x0, y0, r0, rotation), atol=0.2)
        )

        adjustment = (1.23, -4.56, 7.89, 10.11)
        obs.adjust_disc_params(*adjustment)
        self.assertTrue(
            np.allclose(obs.get_wcs_offset(suppress_warnings=True), adjustment),
            msg=f'{obs.get_wcs_offset(suppress_warnings=True)} != {adjustment}',
        )
        obs.adjust_disc_params(dx=10)
        adjustment = (1.23 + 10, -4.56, 7.89, 10.11)
        self.assertTrue(
            np.allclose(obs.get_wcs_offset(suppress_warnings=True), adjustment)
        )

        obs.disc_from_wcs(suppress_warnings=True)
        obs.add_arcsec_offset(1, 2.5)
        self.assertTrue(
            np.allclose(obs.get_wcs_arcsec_offset(suppress_warnings=True), (1, 2.5))
        )
        obs.add_arcsec_offset(10)
        self.assertArraysClose(
            obs.get_wcs_arcsec_offset(suppress_warnings=True), (11, 2.5), atol=1e-3
        )

        obs.disc_from_wcs(suppress_warnings=True)
        obs.adjust_disc_params(dr=10)
        with self.assertRaises(ValueError):
            obs.get_wcs_arcsec_offset(suppress_warnings=True)
        obs.get_wcs_arcsec_offset(
            suppress_warnings=True, check_is_position_offset_only=False
        )

        obs.disc_from_wcs(suppress_warnings=True)
        obs.adjust_disc_params(drotation=123)
        with self.assertRaises(ValueError):
            obs.get_wcs_arcsec_offset(suppress_warnings=True)
        obs.get_wcs_arcsec_offset(
            suppress_warnings=True, check_is_position_offset_only=False
        )

        # check don't get wraparound errors for small -ve drotation
        obs.disc_from_wcs(suppress_warnings=True)
        obs.adjust_disc_params(drotation=-1e-6)
        obs.get_wcs_arcsec_offset(suppress_warnings=True)

    def test_fit_disc(self):
        data = np.ones((5, 10, 8))
        data[:, 3:5, 2:4] = 10
        obs = Observation(
            data=data,
            target='Jupiter',
            observer='hst',
            utc='2005-01-01T00:00:00',
        )
        obs.set_disc_params(0, 0, 99, 99)

        obs.fit_disc_position()
        self.assertAlmostEqual(obs.get_x0(), 2.5)
        self.assertAlmostEqual(obs.get_y0(), 3.5)
        self.assertEqual(obs.get_disc_method(), 'fit_position')

        obs.fit_disc_radius()
        self.assertAlmostEqual(obs.get_r0(), 1.5)
        self.assertEqual(obs.get_disc_method(), 'fit_r0')

        self.assertAlmostEqual(obs.get_rotation(), 99)

        obs = Observation(
            data=np.ones((300, 300)),
            target='Jupiter',
            observer='hst',
            utc='2005-01-01T00:00:00',
        )
        obs.set_disc_params(x0=-1)
        with self.assertRaises(ValueError):
            obs.fit_disc_radius()

        obs.set_disc_params(x0=1, y0=301)
        with self.assertRaises(ValueError):
            obs.fit_disc_radius()

        obs.set_disc_params(x0=150, y0=150)
        obs.fit_disc_radius()

    # get_mapped_data tested against output references

    def test_append_to_header(self):
        obs = Observation(
            data=np.ones((5, 10, 8)),
            target='Jupiter',
            observer='hst',
            utc='2005-01-01T00:00:00',
        )

        obs.append_to_header('TESTING', 123, 'Testing comment')
        self.assertEqual(obs.header['HIERARCH PLANMAP TESTING'], 123)
        self.assertEqual(
            obs.header.comments['HIERARCH PLANMAP TESTING'], 'Testing comment'
        )

        header = fits.Header()
        obs.append_to_header('TESTING', 123, 'Testing comment', header=header)
        self.assertEqual(header['HIERARCH PLANMAP TESTING'], 123)
        self.assertEqual(header.comments['HIERARCH PLANMAP TESTING'], 'Testing comment')
        self.assertNotIn('TESTING', header)

        header = fits.Header()
        obs.append_to_header(
            'TESTING', 123, 'Testing comment', header=header, hierarch_keyword=False
        )
        self.assertEqual(header['TESTING'], 123)
        self.assertEqual(header.comments['TESTING'], 'Testing comment')
        self.assertNotIn('HIERARCH PLANMAP TESTING', header)

        header = fits.Header()
        obs.append_to_header('A', 0, header=header, hierarch_keyword=False)
        obs.append_to_header('B', 1, header=header, hierarch_keyword=False)
        obs.append_to_header('A', 1, header=header, hierarch_keyword=False)
        self.assertEqual(header['A'], 1)
        self.assertEqual(list(header.keys()), ['B', 'A'])

        header = fits.Header()
        obs.append_to_header('A', 0, header=header, hierarch_keyword=False)
        obs.append_to_header('B', 1, header=header, hierarch_keyword=False)
        obs.append_to_header(
            'A', 1, header=header, hierarch_keyword=False, remove_existing=False
        )
        self.assertEqual(header['A'], 0)
        self.assertEqual(list(header.keys()), ['A', 'B', 'A'])

        for n in range(100):
            with self.subTest(n=n):
                s = 'x' * n
                obs.append_to_header('TESTING', s)
                if n >= 53:
                    s = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...'
                self.assertEqual(obs.header['HIERARCH PLANMAP TESTING'], s)

        with self.assertRaises(ValueError):
            obs.append_to_header('TESTING', 'x' * 100, truncate_strings=False)
            obs.header.tostring()

    def test_add_header_metadata(self):
        obs = planetmapper.Observation(
            data=np.ones((5, 10, 8)),
            target='Jupiter',
            observer='hst',
            utc='2005-01-01T00:00:00',
        )
        obs.add_header_metadata()
        self.assertNotIn('HIERARCH PLANMAP INFILE', obs.header)

        path = os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
        obs = planetmapper.Observation(path)
        obs.add_header_metadata()
        self.assertEqual(obs.header['HIERARCH PLANMAP INFILE'], os.path.split(path)[1])

        # add_header_metadata also tested against output references

    def test_make_filename(self):
        self.assertEqual(
            self.observation.make_filename(), 'JUPITER_2005-01-01T000000.fits'
        )
        self.assertEqual(
            self.observation.make_filename(extension='.txt'),
            'JUPITER_2005-01-01T000000.txt',
        )
        self.assertEqual(
            self.observation.make_filename(prefix='pre_', suffix='_post'),
            'pre_JUPITER_2005-01-01T000000_post.fits',
        )

    def test_save_observation(self):
        self.observation.set_disc_params(2.5, 3.1, 3.9, 123.456)
        self.observation.set_disc_method('<<<test>>>')

        path = os.path.join(common_testing.TEMP_PATH, 'test_nav.fits')

        # test skip wireframe here
        self.observation.save_observation(
            path,
            show_progress=True,
            include_wireframe=False,
        )
        self.compare_fits_to_reference(path, skip_wireframe=True)

        # test print info here
        self.observation.save_observation(
            path, print_info=True, wireframe_kwargs=dict(output_size=20, dpi=20)
        )
        self.compare_fits_to_reference(path)

        # test progress bar here too
        self.observation.save_observation(
            path, show_progress=True, wireframe_kwargs=dict(output_size=20, dpi=20)
        )
        self.compare_fits_to_reference(path)

        # test PathLike
        self.observation.save_observation(
            Path(path),
            show_progress=True,
            include_wireframe=False,
        )
        self.compare_fits_to_reference(path, skip_wireframe=True)

    def test_save_mapped_observation(self):
        self.observation.set_disc_params(2.5, 3.1, 3.9, 123.456)
        self.observation.set_disc_method('<<<test>>>')

        map_kwargs = {
            'rectangular-nearest': dict(
                degree_interval=30, interpolation='nearest', show_progress=True
            ),
            'rectangular-linear': dict(
                degree_interval=30, interpolation='linear', include_wireframe=False
            ),
            'rectangular-quadratic': dict(
                degree_interval=30,
                interpolation='quadratic',
                include_backplanes=False,
                include_wireframe=False,
            ),
            'rectangular-cubic': dict(
                degree_interval=30,
                interpolation='cubic',
                include_backplanes=False,
                include_wireframe=False,
            ),
            'rectangular-interpolation': dict(
                degree_interval=30,
                interpolation=(1, 3),
                spline_smoothing=1.23,
                include_backplanes=False,
                include_wireframe=False,
            ),
            'orthographic-1': dict(
                projection='orthographic', size=10, include_wireframe=False
            ),
            'orthographic-2': dict(
                projection='orthographic',
                lat=90,
                size=5,
            ),
            'orthographic-3': dict(
                projection='orthographic',
                lat=-21.3,
                lon=-42,
                size=4,
                include_wireframe=False,
            ),
            'azimuthal-1': dict(
                projection='azimuthal', size=10, include_wireframe=False
            ),
            'azimuthal-2': dict(
                projection='azimuthal',
                lat=-90,
                size=5,
            ),
            'azimuthal-3': dict(
                projection='azimuthal',
                lat=42,
                lon=12.345,
                size=4,
                include_wireframe=False,
            ),
        }

        for map_type, map_kw in map_kwargs.items():
            with self.subTest(
                map_type=map_type,
            ):
                path = os.path.join(common_testing.TEMP_PATH, f'map_{map_type}.fits')
                self.observation.save_mapped_observation(
                    path, **map_kw, wireframe_kwargs=dict(output_size=20, dpi=20)
                )
                self.compare_fits_to_reference(path)

        with self.subTest('PathLike'):
            map_type = 'rectangular-nearest'
            map_kw = map_kwargs[map_type]
            path = os.path.join(common_testing.TEMP_PATH, f'map_{map_type}.fits')
            self.observation.save_mapped_observation(
                Path(path), **map_kw, wireframe_kwargs=dict(output_size=20, dpi=20)
            )
            self.compare_fits_to_reference(path)

    def compare_fits_to_reference(self, path: str, skip_wireframe: bool = False):
        filename = os.path.basename(path)
        path_ref = os.path.join(common_testing.DATA_PATH, 'outputs', filename)
        with fits.open(path) as hdul, fits.open(path_ref) as hdul_ref:
            if skip_wireframe:
                hdul_ref = fits.HDUList(
                    [hdu for hdu in hdul_ref if hdu.name != 'WIREFRAME']
                )
            with self.subTest('Number of backplanes', filename=filename):
                self.assertEqual(len(hdul), len(hdul_ref))

            with self.subTest('Backplane names', filename=filename):
                self.assertEqual(
                    set(hdu.name for hdu in hdul),
                    set(hdu.name for hdu in hdul_ref),
                )

            for hdu_ref in hdul_ref:
                extname = hdu_ref.name
                hdu = hdul[extname]
                with self.subTest('HDU data', filename=filename, extname=extname):
                    data = hdu.data
                    data_ref = hdu_ref.data
                    self.assertEqual(data.shape, data_ref.shape)

                    # Significantly increase tolerance for wireframe as it is generated
                    # from a Matplotlib plot, so is sensitive to the OS/environment
                    # (e.g. fonts available), and is only a cosmetic backplane anyway
                    # so the actual values don't matter anywhere near as much as the
                    # other backplanes.
                    atol = 64 if extname == 'WIREFRAME' else 1e-6
                    self.assertArraysClose(data, data_ref, atol=atol, equal_nan=True)

                header = hdu.header
                header_ref = hdu_ref.header
                with self.subTest('HDU header', filename=filename, extname=extname):
                    self.assertEqual(set(header.keys()), set(header_ref.keys()))

                keys_to_skip = {'*DATE*', '*VERSION*'}
                for key in header.keys():
                    if any(
                        fnmatch.fnmatch(key.casefold(), pattern.casefold())
                        for pattern in keys_to_skip
                    ):
                        continue
                    value = header[key]
                    value_ref = header_ref[key]
                    with self.subTest(
                        'HDU keader key', filename=filename, extname=extname, key=key
                    ):
                        if isinstance(value, float):
                            self.assertAlmostEqual(value, value_ref)
                        else:
                            self.assertEqual(value, value_ref)

    @patch('planetmapper.gui.GUI.run')
    def test_run_gui(self, mock_run: MagicMock):
        out = self.observation.run_gui()
        self.assertEqual(out, [])
        mock_run.assert_called_once()
