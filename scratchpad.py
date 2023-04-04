#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.io import fits

observation = planetmapper.Observation(
    '/Users/ortk1/Dropbox/science/jwst_data/MIRI_IFU/Saturn_2022nov13/SATURN-45N/stage6_background/d1_fringe_nav/Level3_ch1-long_s3d_nav.fits'
)
observation.run_gui()
