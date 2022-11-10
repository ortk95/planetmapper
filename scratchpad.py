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

body = planetmapper.BodyXY(
    target='saturn',
    utc='2022-11-04T21:31:23.939',
    observer='JWST',
    sz=50,
)
body.add_other_bodies_of_interest('titan')


fig, ax = plt.subplots(figsize=(10, 10))
body.plot_wireframe_radec(ax=ax)
