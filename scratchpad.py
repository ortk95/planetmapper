#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt

p = '/Users/ortk1/Dropbox/PhD/data/jwst/NIRSPEC_IFU/2023-01-08_Uranus/lon2/data/level2_nav/jw01248004001_03101_00001_nrs1_s3d_nav.fits'
observation = planetmapper.Observation(p, aberration_correction='CN')
observation.disc_from_wcs(True, False)
observation.plot_wireframe_xy()
plt.show()
