import fnmatch
import os
import unittest

import common_testing
import numpy as np
from astropy.io import fits

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import Observation


class TestObservation(unittest.TestCase):
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

    def test_attributes(self):
        self.assertEqual(self.observation.path, self.path)
        self.assertEqual(self.observation.target, 'JUPITER')
        self.assertEqual(self.observation.observer, 'HST')
        self.assertEqual(self.observation.utc, '2005-01-01T00:00:00.000000')
        self.assertEqual(self.observation._nx, 7)
        self.assertEqual(self.observation._ny, 10)

    def test_repr(self):
        self.assertEqual(repr(self.observation), f'Observation({self.path!r})')

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

    # add_header_metadata tested against output references

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

    def compare_fits_to_reference(self, path: str):
        filename = os.path.basename(path)
        path_ref = os.path.join(common_testing.DATA_PATH, 'outputs', filename)
        with fits.open(path) as hdul, fits.open(path_ref) as hdul_ref:
            with self.subTest('Number of backplanes', filename=filename):
                self.assertEqual(len(hdul), len(hdul_ref))
            for hdu, hdu_ref in zip(hdul, hdul_ref):
                self.assertEqual(hdu.name, hdu_ref.name)
                extname = hdu.name
                with self.subTest(filename=filename, extname=extname):
                    data = hdu.data
                    data_ref = hdu_ref.data
                    self.assertEqual(data.shape, data_ref.shape)

                    # Significantly increase tolerance for wireframe as it is generated
                    # from a Matplotlib plot, so is sensitive to the OS/environment
                    # (e.g. fonts available), and is only a cosmetic backplane anyway
                    # so the actual values don't matter anywhere near as much as the
                    # other backplanes.
                    atol = 64 if extname == 'WIREFRAME' else 1e-8  # 1e-8 is the default
                    self.assertTrue(
                        np.allclose(data, data_ref, atol=atol, equal_nan=True)
                    )

                header = hdu.header
                header_ref = hdu_ref.header
                with self.subTest(filename=filename, extname=extname):
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
                    with self.subTest(filename=filename, extname=extname, key=key):
                        if isinstance(value, float):
                            self.assertAlmostEqual(value, value_ref)
                        else:
                            self.assertEqual(value, value_ref)
