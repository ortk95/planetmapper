#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np
import spiceypy as spice

lon = 0
lat = -18

body = planetmapper.BodyXY('saturn', sz=50, utc='2023-01-11T19:00')
body.set_r0(10)
body.coordinates_of_interest_lonlat.append((lon, lat))
# body.plot_wireframe_xy()
# plt.show()

radius, _, _ = body.ring_plane_coordinates(*body.lonlat2radec(lon, lat))
ring_data = planetmapper.data_loader.get_ring_radii()['SATURN']
for name, radii in ring_data.items():
    if min(radii) < radius < max(radii):
        print(f'Point obscured by {name} ring')
        break
else:
    print('Point not obscured by rings')

body.plot_backplane_img('RING-RADIUS')
body.plot_backplane_map('RING-RADIUS')