#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np

# gui = planetmapper.gui.GUI()
# gui.run()

p = '/Users/ortk1/Desktop/iew608inq_drz.fits'
observation = planetmapper.Observation(p, target='uranus')
# observation.plot_wireframe_radec()
# plt.show()

header = observation.header
# wcs = observation._get_wcs_from_header()
wcs = astropy.wcs.WCS(header).celestial

ra, dec = observation.limb_radec()

img = np.nanmean(observation.data, axis=0)
plt.imshow(img, origin='lower', cmap='Greys_r')
x, y = np.array(wcs.world_to_pixel_values(np.array([ra, dec]).T)).T
plt.plot(x, y)

observation.disc_from_wcs(True)
x1, y1 = observation.limb_xy()
plt.plot(x1, y1)
plt.show()
