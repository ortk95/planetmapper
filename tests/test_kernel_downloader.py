import os
import shutil
import time
import unittest

import common_testing

import planetmapper
from planetmapper import kernel_downloader


class TestKernelDownloader(unittest.TestCase):
    def setUp(self):
        self.kernel_path = os.path.join(common_testing.TEMP_PATH, 'kernels')
        planetmapper.set_kernel_path(self.kernel_path)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.kernel_path)
        except FileNotFoundError:
            pass
        planetmapper.set_kernel_path(common_testing.KERNEL_PATH)

    def test_root(self):
        self.assertEqual(kernel_downloader.URL_ROOT, 'https://naif.jpl.nasa.gov/pub/')

    def test_download_urls(self):
        with self.subTest('single url'):
            kernel_downloader.download_urls(
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/aareadme.txt'
            )
            local_path = os.path.join(
                self.kernel_path, 'naif', 'VIKING', 'kernels', 'aareadme.txt'
            )
            self.assertTrue(os.path.exists(local_path))
            with open(local_path, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(
                lines[1].strip(),
                'SPICE Data for Viking Mission (Orbiters and Landers) (06/1996 to 07/1980)',
            )
            self.assertEqual(len(lines), 14)

        with self.subTest('multiple urls'):
            kernel_downloader.download_urls(
                'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk'
            )
            local_path = os.path.join(
                self.kernel_path, 'naif', 'generic_kernels', 'lsk'
            )
            self.assertTrue(os.path.exists(local_path))
            files = os.listdir(local_path)
            self.assertGreater(len(files), 2)
            self.assertIn('latest_leapseconds.tls', files)

        with self.subTest('existing url'):
            local_path = os.path.join(
                self.kernel_path, 'naif', 'VIKING', 'kernels', 'aareadme.txt'
            )
            t = os.path.getmtime(local_path)
            if time.time() - t < 1:
                time.sleep(1)  # ensure at least 1s between downloads
            kernel_downloader.download_kernel(
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/aareadme.txt'
            )
            self.assertEqual(os.path.getmtime(local_path), t)
            kernel_downloader.download_kernel(
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/aareadme.txt',
                force_download=True,
            )
            self.assertGreater(os.path.getmtime(local_path), t)

    def test_get_kernel_paths_from_webpage(self):
        self.assertEqual(
            set(
                kernel_downloader.get_kernel_paths_from_webpage(
                    'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk'
                )
            ),
            {
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/mar033-7.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vl1.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vl2.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vo1_ext_gem.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vo1_rcon.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vo1_sedr.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vo2_rcon.bsp',
                'https://naif.jpl.nasa.gov/pub/naif/VIKING/kernels/spk/vo2_sedr.bsp',
            },
        )
