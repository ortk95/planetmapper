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

planetmapper.utils.print_progress()


body = planetmapper.BodyXY('Jupiter', '2000-01-01', sz=500)
body.set_disc_params(x0=250, y0=250, r0=200)
planetmapper.utils.print_progress()
body.get_backplane_img('LON') # Takes ~15s to execute
planetmapper.utils.print_progress()
body.get_backplane_img('LAT') # Executes instantly
planetmapper.utils.print_progress()

# This changes the disc location, so the cache is cleared
body.set_r0(190)


planetmapper.utils.print_progress()

body.get_backplane_img('LAT') # Takes ~15s to execute


planetmapper.utils.print_progress()



