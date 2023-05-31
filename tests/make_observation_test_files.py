#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image

import planetmapper

path = os.path.join(common_testing.DATA_PATH, 'inputs', 'image.png')
image = np.ones((10, 5, 4), dtype=np.uint8) * 100
plt.imsave(path, image)


path = os.path.join(common_testing.DATA_PATH, 'inputs', 'planmap.fits')
image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float)
header = fits.Header(
    {
        'TARGET': 'jupiter',
        'TELESCOP': 'HST',
        'DATE-OBS': '2005-01-01',
        'TIME-OBS': '12:00:00',
        'HIERARCH PLANMAP DISC X0': 1.1,
        'HIERARCH PLANMAP DISC Y0': 2.2,
        'HIERARCH PLANMAP DISC R0': 3.3,
        'HIERARCH PLANMAP DISC ROT': 4.4,
    }
)
fits.writeto(path, data=image, header=header, overwrite=True)

path = os.path.join(common_testing.DATA_PATH, 'inputs', 'wcs.fits')
image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float)
header = fits.Header(
    {
        'TARGET': 'jupiter',
        'TELESCOP': 'HST',
        'DATE-OBS': '2005-01-01 00:00',
        'CRPIX1': 1,
        'CRPIX2': 1,
        'CRVAL1': 196.37,
        'CRVAL2': -5.56,
        'CDELT1': 3e-5,
        'CDELT2': 3e-5,
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'PC1_1': 0.1710829588022924,
        'PC1_2': -0.99852566270812154,
        'PC2_1': -0.99852566270812154,
        'PC2_2': -0.1710829588022924,
    }
)
fits.writeto(path, data=image, header=header, overwrite=True)


path = os.path.join(common_testing.DATA_PATH, 'inputs', 'extended.fits')
image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float)
primary_hdu = fits.PrimaryHDU(
    header=fits.Header(
        {
            'TARGET': 'jupiter',
            'TELESCOP': 'HST',
        }
    )
)
sci_hdu = fits.ImageHDU(
    data=image, header=fits.Header({'DATE-OBS': '2005-01-01 12:00'})
)
hdul = fits.HDUList([primary_hdu, sci_hdu])
hdul.writeto(path, overwrite=True)


path = os.path.join(common_testing.DATA_PATH, 'inputs', 'test.fits')
header = fits.Header(
    {
        'TARGET': 'jupiter',
        'TELESCOP': 'HST',
        'DATE-OBS': '2005-01-01',
        'TIME-OBS': '00:00:00',
        'CUSTOM': '<<<  testing  >>>',
    }
)
cube = np.ones((10, 10, 7))
cube[1] = np.nan
cube[2, ::2] = np.nan
cube[3, ::2] = np.nan
cube[3, :, ::2] = np.nan
cube[4, 5, 3] = np.nan
cube[5] = 42
cube[6, ::2] = 2
cube[6, :, ::2] *= 1.234
cube[7, -1, -1] = 9999.99
cube[7, 3, 4] = 1.234
cube[7, 6, 3] = 43
cube[7, 3, 3] = -123
cube[8, 4] = -1
cube[9, 0] = np.nan
cube[9, -1] = np.nan
cube[9, :, 0] = np.nan
cube[9, :, -1] = np.nan
fits.writeto(path, data=cube, header=header, overwrite=True)


path = os.path.join(common_testing.DATA_PATH, 'inputs', 'empty.fits')
hdul = fits.HDUList([fits.PrimaryHDU()])
hdul.writeto(path, overwrite=True)


path = os.path.join(common_testing.DATA_PATH, 'inputs', '2d_image.fits')
hdul = fits.HDUList(
    [
        fits.PrimaryHDU(
            data=np.array([[1, 2], [3, 4]], dtype=float),
            header=fits.Header(
                {
                    'TARGET': 'jupiter',
                    'TELESCOP': 'HST',
                    'MJD-BEG': 51544,
                    'MJD-END': 51545,
                }
            ),
        ),
    ]
)
hdul.writeto(path, overwrite=True)


path = os.path.join(common_testing.DATA_PATH, 'inputs', '2d_image.png')
image = np.array([[3, 4], [1, 2]], dtype=np.uint8)
im = Image.fromarray(image)
im.save(path)
