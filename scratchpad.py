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
import matplotlib.ticker
from functools import lru_cache

body = planetmapper.Body('Saturn', datetime.datetime.now())
body.plot_wireframe_radec(color='r')
body = planetmapper.Body(
    'Saturn', datetime.datetime.now() + datetime.timedelta(hours=1)
)
body.plot_wireframe_radec(color='b')
print('DONE')


from planetmapper.kernel_downloader import download_urls

# Download all kernel files in generic_kernels/pck
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/')

# Download specific kernel file
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls')

# Download multiple sets of kernel files
download_urls(
    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/',
    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/',
)
