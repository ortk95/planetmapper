#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np

p = '/Users/ortk1/Downloads/URA_NIRSPEC_F290LP_G395H_NRS2_Lon3_2_nav.fits'
observation = planetmapper.Observation(p)
# observation.plot_backplane_img('KM-X', show=True)
# observation.plot_backplane_img('KM-Y', show=True)
# observation.plot_backplane_map('KM-X', show=True)
# observation.plot_backplane_map('KM-Y', show=True)
# observation.plot_backplane_map('PIXEL-X', projection='azimuthal', size=500,lat=90)

map_kw = dict(projection='perspective', size=100, lat=90, altitude=500)

img = observation.map_img(np.nanmean(observation.data, axis=0), **map_kw)
observation.plot_map(img, **map_kw)
