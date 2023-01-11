#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np

p = '/Users/ortk1/Desktop/nav.fits'
observation = planetmapper.Observation(p, target='uranus')

img = np.nanmean(observation.data, axis=0)
plt.imshow(img, origin='lower', cmap='Greys_r')

x1, y1 = observation.limb_xy()
plt.plot(x1, y1)
# # plt.show()


# observation.run_gui()

# gui = planetmapper.gui.GUI()
# gui.run()
