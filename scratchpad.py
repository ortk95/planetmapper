#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt

p = '/Users/ortk1/Dropbox/PhD/data/jwst/Uranus_2023jan08/lon2/stage3/d1/Level3_ch4-short_s3d.fits'
p = '/Users/ortk1/Dropbox/PhD/data/jwst/saturn/SATURN-75N/stage6_flat/d1_fringe_nav/Level3_ch1-short_s3d_nav.fits'
body = planetmapper.Observation(p)
# body.plot_backplane_map('pixel-x')
# plt.show()

img = body.data[0]
mapped = body.map_img(img, interpolation='cubic')
body.imshow_map(mapped)
plt.show()

# mapped = body.map_img(img, interpolation='linear', mask_nan=False)
# body.imshow_map(mapped)
# plt.show()