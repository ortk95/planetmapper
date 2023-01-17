#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np
import spiceypy as spice

p = '/Users/ortk1/Dropbox/PhD/data/jwst/Uranus_2023jan08/lon2/stage3/d1/Level3_ch4-short_s3d.fits'
# p = '/Users/ortk1/Dropbox/PhD/data/jwst/NIRSPEC_IFU/2023-01-08_Uranus/lon2/data/level2_nav/jw01248004001_03103_00004_nrs1_s3d_nav.fits'
body = planetmapper.Observation(p, show_progress=False)

ax = body.plot_wireframe_xy()
img = np.nanmedian(body.data, axis=0)
ax.imshow(img, origin='lower')
plt.show()
img = np.nan_to_num(img)

for interpolation in ('nearest', 'linear', 'quadratic', 'cubic'):
    map_img = body.map_img(img, interpolation=interpolation)
    fig, ax = plt.subplots()
    body.imshow_map(map_img, ax=ax)
    ax.set_title(interpolation)
    plt.show()
