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
    p = '/Users/ortk1/Dropbox/PhD/data/jwst/saturn/jw01247-o330_t600_miri_ch1-shortmediumlong/jw01247-o330_t600_miri_ch1-shortmediumlong_s3d.fits'

    obs = planetmapper.Observation(p, observer='JWST')

    ax = obs.plot_wireframe_radec(show=False)
    for k, c in [('DATE-BEG', 'r'), ('DATE-END', 'b')]:
        utc = obs.header[k]
        obs = planetmapper.Observation(p, observer='JWST', utc=utc)
        obs.plot_wireframe_radec(color=c, show=False)
        ax.plot(np.nan, np.nan, color=c, label=f'{k}: {utc}')
    ax.legend()
    plt.show()
print('DONE')
