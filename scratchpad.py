#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
from planetmapper import utils, mapper
import numpy as np
from functools import wraps
import tools

times = [
    '2022-07-28T06:03:59.373',
    '2022-07-28T08:03:59.373',
]
for t in times:
    body = mapper.BodyXY(
        'jupiter',
        t,
        observer='JWST',
        sz=50,
    )
    # ax = body.plot_backplane('doppler')
    ax = body.plot_backplane('lon')
    print(
        body.radial_velocity_from_lonlat(30, 0),
        body.radial_velocity_from_lonlat(30 + 12, 0),
    )