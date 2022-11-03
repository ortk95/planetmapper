#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper
import numpy as np
import tools

# obs = planetmapper.gui.InteractiveObservation('data/europa.fits.gz')
obs = planetmapper.gui.InteractiveObservation('data/jupiter_small.jpg',
target='jupiter',
utc='August 25, 2020',
)

obs.run()
