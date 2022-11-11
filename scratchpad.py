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

if False:
    planetmapper.utils.print_progress()
    gui = planetmapper.gui.GUI(
        'data/saturn.jpg',
        target='saturn',
        utc='2001-12-08T04:39:30.449',
    )
    gui.observation.set_disc_params(x0=650, y0=540, r0=200)
    gui.observation.add_other_bodies_of_interest('Tethys')
    gui.run()

else:
    p = '/Users/ortk1/Dropbox/PhD/data/jwst/pandora/jw01247-o312_t617_nirspec_prism-clear/jw01247-o312_t617_nirspec_prism-clear_s3d.fits'
    # cube, hdr = astropy.io.fits.getdata(p, header=True) # type: ignore
    observation = planetmapper.Observation(
        p, target='pandora', utc='2022-11-08T21:43:13.707'
    )
    observation.disc_from_wcs()
    # wcs = observation._get_wcs_from_header()
    # coords = [
    #     [0, 0],
    #     [1, 1],
    #     [90, 3298],
    # ]

    # for x, y in coords:
    #     w = wcs.pixel_to_world_values(x, y)
    #     o = observation.xy2radec(x, y)
    #     print(np.array(w) % 360, np.array(o) % 360)
