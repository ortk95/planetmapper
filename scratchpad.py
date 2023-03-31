#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np

p = '/Users/ortk1/Downloads/URA_NIRSPEC_F290LP_G395H_NRS2_Lon3_2_nav.fits'

observation = planetmapper.Observation(p)
# observation.plot_backplane_img('KM-X', show=True)
# observation.plot_backplane_img('KM-Y', show=True)
# observation.plot_backplane_map('KM-X', show=True)
# observation.plot_backplane_map('KM-Y', show=True)

observation.plot_backplane_map('LON-GRAPHIC')