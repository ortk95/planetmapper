#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np

try:
    observation: planetmapper.Observation = observation  #  type: ignore
except NameError:
    p = '/Users/ortk1/Dropbox/science/jwst_data/MIRI_IFU/Saturn_2022nov13/SATURN-15N/stage3/d2_fringe_nav/Level3_ch1-short_s3d_nav.fits'
    observation = planetmapper.Observation(p)

# body.plot_backplane_map('pixel-x')
# plt.show()

img = observation.data[187]
img[img.shape[0]//2, img.shape[1]//2] = np.nan
mapped = observation.map_img(img, interpolation='linear', degree_interval=1)

fig, ax = plt.subplots()
observation.imshow_map(mapped, ax=ax,
                                   vmin=np.nanpercentile(img, 1),
            vmax=np.nanpercentile(img, 99),
)

ax.set_xlim(np.nanmax(observation.get_lon_img()), np.nanmin(observation.get_lon_img()))
ax.set_ylim(np.nanmin(observation.get_lat_img()), np.nanmax(observation.get_lat_img()))

plt.show()

# mapped = body.map_img(img, interpolation='linear', mask_nan=False)
# body.imshow_map(mapped)
# plt.show()
