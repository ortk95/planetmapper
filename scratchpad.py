#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime

planetmapper.set_kernel_path(
    '/Users/ortk1/Dropbox/science/planetmapper/tests_new/data/kernels'
)

body = planetmapper.BodyXY(
    'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
)
body_zero_size = planetmapper.BodyXY(
    'Jupiter', observer='HST', utc='2005-01-01T00:00:00'
)

lons = np.linspace(0, 360, 20)
lats=  np.linspace(-90, 90, 10)
lons, lats = np.meshgrid(lons, lats)

body.get_emission_angle_map(projection='manual', lon_coords=lons, lat_coords=lats)

# print(datetime.datetime.now())
# body.get_emission_angle_map()
# print(datetime.datetime.now())
# body.get_emission_angle_map()
# print(datetime.datetime.now())
