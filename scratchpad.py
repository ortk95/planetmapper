#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper
import numpy as np
import tools

# obs = planetmapper.gui.InteractiveObservation('data/europa.fits.gz')
obs = planetmapper.gui.InteractiveObservation(
    'data/jupiter_small.jpg',
    target='jupiter',
    utc='2020-08-25 02:30:40',
)
obs.observation.add_other_bodies('Io', 'Europa', 'Ganymede', 'Callisto')
obs.observation.set_disc_params(x0=137.0, y0=119.0, r0=80.0, rotation=354.0)
obs.run()
