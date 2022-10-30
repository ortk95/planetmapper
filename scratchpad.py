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
import datetime
import utils


target = 'jupiter'
sz = 100

backplane = 'lon'
for b in (True, False):
    utils.print_progress()
    body = mapper.BodyXY(target, datetime.datetime.now(), nx=sz, ny=sz)
    body._do_pixel_radius_short_circuit = b
    body.set_params(x0=0.5 * sz, y0=0.5 * sz, r0=0.1 * sz)
    im = body.get_backplane_img(backplane)
    utils.print_progress(str(b))
