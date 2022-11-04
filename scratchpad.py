#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper
import numpy as np
import tools
import datetime
import spiceypy as spice

# # obs = planetmapper.gui.InteractiveObservation('data/europa.fits.gz')
# obs = planetmapper.gui.InteractiveObservation(
#     'data/jupiter_small.jpg',
#     target='jupiter',
#     utc='2020-08-25 02:30:40',
# )
# obs.observation.add_other_bodies_of_interest('Io', 'Europa', 'Ganymede', 'Callisto')
# obs.observation.set_disc_params(x0=137.0, y0=119.0, r0=80.0, rotation=354.0)
# obs.observation.plot_wireframe_xy()
# # obs.run()

# planetmapper.PlanetMapperTool().load_spice_kernels()

# x = spice.bodvcd(699, 'RING1', 10)
# print(x)


body = planetmapper.Body('saturn', '2022-01-01T00:00:00')
lonlat = (90, 0)
body.coordinates_of_interest_lonlat.append(lonlat)

body.plot_wireframe_radec()

targvec = body.lonlat2targvec(*lonlat)*10

phase, incdnc, emissn, visibl, lit = body._illumf_from_targvec_radians(targvec)

print(visibl)
