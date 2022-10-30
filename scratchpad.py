#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import utils
from multiprocessing import Pool
import time
import spiceypy as spice
import numpy as np
import ctypes


target = 'jupiter'
dates = ('2022-10-31 00:00:00', '2022-10-31 01:00:00')

backplane = 'lon'
sz = 50
for d in dates:
    body = mapper.BodyXY(target, d, nx=sz, ny=sz)
    body.set_params(x0=0.5 * sz, y0=0.5 * sz, r0=0.45 * sz)
    body.plot_backplane(backplane)
    im = body.get_backplane_img(backplane)
    print(np.nanmean(im))
