#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper.data_loader
import planetmapper
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.patheffects as path_effects
import astropy.io.fits
import matplotlib.ticker
from functools import lru_cache

if False:
    if True:
        gui = planetmapper.gui.GUI(
            'data/saturn.jpg',
            target='saturn',
            utc='2001-12-08T04:39:30.449',
        )
        gui.observation.set_disc_params(x0=650, y0=540, r0=200)
        gui.observation.add_other_bodies_of_interest('Tethys')
    else:
        gui = planetmapper.gui.GUI(
            'data/jupiter.jpg',
            target='jupiter',
            utc='2020-08-25 02:30:40',
        )
    gui.observation.plot_wireframe_radec()
    # gui.run()


body = planetmapper.BodyXY('saturn', datetime.datetime.now())
planetmapper.utils.print_progress()
for bp in body.backplanes.values():
    # body.plot_backplane_map(bp.name)
    img = bp.get_img()
    img = bp.get_map()
# 
# print(body.graphic2centric_lonlat(10, 10))
# print(body.centric2graphic_lonlat(10, 10))


# print(body.prograde, body.positive_planetographic_longitude_direction)
