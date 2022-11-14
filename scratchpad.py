#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
from planetmapper import observation
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

    # obs = planetmapper.Observation(p, observer='JWST')
    # observation.disc_from_wcs()

    obs = planetmapper.Body('moon', datetime.datetime.now())
    print(obs.north_pole_angle())
    # img = np.nansum(obs.data, axis=0)

    # Plot an observed image on an RA/Dec axis with a wireframe of the target
    # ax = obs.plot_wireframe_radec()
    ax = obs.plot_wireframe_km()
    for x in [-obs.r_eq, 0, obs.r_eq]:
        ax.axvline(x)
    for y in [-obs.r_polar, 0, obs.r_polar]:
        ax.axhline(y)
    plt.show()

    # ax.autoscale(False)
    # ax.imshow(img, origin='lower', transform=obs.matplotlib_xy2radec_transform(ax))

print('DONE')
