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
    gui.run()
else:
    p = '/Users/ortk1/Dropbox/PhD/data/jwst/saturn/reference/cgo.jpg'

    # obs = planetmapper.Observation(p, observer='JWST', aberration_correction='CN')

    # obs.disc_from_wcs(supress_warnings=True)
    # obs.adjust_disc_params(dy=-5)
    # obs.plot_wireframe_xy()
    # wcs = obs._get_wcs_from_header()

    gui = planetmapper.gui.GUI(p, target='saturn', utc='2022-11-14T10:28')
    # gui.observation.disc_from_wcs()


    gui.observation.set_disc_params(
        x0=368.9999999999998, y0=568.0, r0=152.7, rotation=6.300000000000003
    )

    gui.run()



    # ax = gui.observation.plot_wireframe_xy()
    # for x, y in gui.observation.visible_lonlat_grid_xy(90):
    #     ax.plot(x,y)
print('DONE')
