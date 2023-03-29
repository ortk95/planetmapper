#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
import planetmapper.progress
import scipy.interpolate

p = '/Users/ortk1/Downloads/URA_NIRSPEC_F290LP_G395H_NRS2_Lon3_2_nav.fits'

observation = planetmapper.Observation(p)
# observation.run_gui()
observation.save_mapped_observation('~/test.fits')