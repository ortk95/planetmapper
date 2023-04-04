#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.io import fits

planetmapper.set_kernel_path(
    '/Users/ortk1/Dropbox/science/planetmapper/tests/data/kernels'
)

body = planetmapper.BodyXY(
    'Jupiter', observer='HST', utc='2005-01-01T00:00:00', nx=15, ny=10
)

print(datetime.datetime.now())
body.get_emission_angle_map()
print(datetime.datetime.now())
body.get_emission_angle_map()
print(datetime.datetime.now())
