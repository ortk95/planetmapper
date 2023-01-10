#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os

p = '/Users/ortk1/Desktop/iew608inq_drz.fits'
observation = planetmapper.Observation(p, aberration_correction='CN', target='uranus')
observation.plot_wireframe_radec()
observation.run_gui()
# observation.disc_from_wcs(True, False)
# observation.plot_wireframe_xy()
# plt.show()

# plt.plot(*observation.limb_xy(), label='Auto')
# wcs = observation._get_wcs_from_header(suppress_warnings=True)

# x, y = np.array(wcs.world_to_pixel_values(np.array(observation.limb_radec()).T)).T

# plt.plot(x, y, label='WCS')
