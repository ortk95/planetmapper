#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np
import spiceypy as spice

p = '/Users/ortk1/Dropbox/PhD/data/jwst/Uranus_2023jan08/lon2/stage3/d1/Level3_ch4-short_s3d.fits'
body = planetmapper.Observation(p)
body.plot_wireframe_radec(show=True, add_axis_labels=False, dms_ticks=False)
body.plot_wireframe_km(show=True, aspect_adjustable='box', add_title=False)
body.plot_wireframe_xy(show=True)
