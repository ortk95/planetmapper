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
# observation.run_gui()
# observation.save_mapped_observation('~/test.fits')
# observation.save_observation('~/test.fits')
fig, ax = plt.subplots()
observation.plot_wireframe_xy(
    ax=ax, indicate_equator=True, indicate_prime_meridian=True
)
ax.imshow(np.nanmean(observation.data, axis=0), origin='lower')
plt.show()


body = planetmapper.Body('Jupiter', utc='2022-01-01T7:00')
body.plot_wireframe_radec(
    indicate_equator=True, indicate_prime_meridian=True, grid_interval=30
)
