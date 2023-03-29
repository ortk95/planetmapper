#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
import planetmapper.progress
import scipy.interpolate

p = '/Users/ortk1/Downloads/URA_NIRSPEC_F290LP_G395H_NRS2_Lon3_2_nav.fits'

observation = planetmapper.Observation(p)


img = np.nanmean(observation.data, axis=0)

fig, ax = plt.subplots()
observation.plot_wireframe_xy(
    ax=ax, indicate_equator=True, indicate_prime_meridian=True
)
ax.imshow(img, origin='lower')
plt.show()

x_out, y_out, img_out, transformer = observation.project_img_orthographic(
    img,
    lat=90,
    interpolation='cubic',
)

fig, ax = plt.subplots()
ax.pcolormesh(x_out, y_out, img_out)

# Add gridlines
npts = 360
for lat in np.arange(-90, 90, 30):
    x, y = transformer.transform(np.linspace(0, 360, npts), lat * np.ones(npts))
    ax.plot(x, y, color='k', alpha=0.5, linestyle='-' if lat == 0 else ':')

npts = 180
for lon in np.arange(0, 360, 30):
    x, y = transformer.transform(lon * np.ones(npts), np.linspace(-90, 90, npts))
    ax.plot(x, y, color='k', alpha=0.5, linestyle='-' if lon == 0 else ':')

ax.set_xlabel('Distance (km)')
ax.set_ylabel('Distance (km)')
ax.ticklabel_format(style='sci', scilimits=(-3, 3))
ax.set_aspect('equal', adjustable='box')
ax.set_title('Northern hemisphere')
