#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import astropy.wcs
import numpy as np
import spiceypy as spice

body = planetmapper.Body('uranus')
body.add_satellites_to_bodies_of_interest()
body.plot_wireframe_radec()
