#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper.data_loader
import planetmapper
from planetmapper import utils
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.patheffects as path_effects
import astropy.io.fits
import matplotlib.ticker
from functools import lru_cache
import scipy.interpolate

# gui = planetmapper.gui.GUI(
#             'data/jupiter.jpg',
#             target='jupiter',
#             utc='2020-08-25 02:30:40',
#                     )
# gui.run()
try:
    obs  #  type: ignore
    raise NameError
except NameError:
    obs = planetmapper.Observation(
        'data/jupiter_small.jpg',
        target='jupiter',
        utc='2020-08-25 02:30:40',
    )
    obs.set_disc_params(
        x0=135.626383468748,
        y0=118.52747744865971,
        r0=81.5,
        rotation=352.0,
    )
# obs.plot_backplane_img('lat-graphic')
# obs.plot_backplane_map('pixel-x')
utils.print_progress('start')
img = np.nansum(obs.data, axis=0)

plt.imshow(obs.map_img(img), origin='lower')
plt.show()
