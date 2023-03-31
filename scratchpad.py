#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np

p = '/Users/ortk1/Downloads/URA_NIRSPEC_F290LP_G395H_NRS2_Lon3_2_nav.fits'
# p = '/Users/ortk1/Dropbox/science/jwst_data/Shared/2023-03-27 Enceladus/Enceladus_2022-11-14/stage3/d1_fringe/Level3_ch1-long_s3d.fits'
observation = planetmapper.Observation(p)
# observation.plot_backplane_img('KM-X', show=True)
# observation.plot_backplane_img('KM-Y', show=True)
# observation.plot_backplane_map('KM-X', show=True)
# observation.plot_backplane_map('KM-Y', show=True)

# observation.plot_backplane_map('PIXEL-X', projection='azimuthal', size=500,lat=90)
mapped_cube = observation.get_mapped_data(projection='orthographic', size=50, lat=90)
plt.imshow(mapped_cube[0])
