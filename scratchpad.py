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

planetmapper.utils.print_progress()
gui = planetmapper.gui.GUI(
    'data/saturn.jpg',
    target='neptune',
    utc='2001-12-08T04:39:30.449',
)
gui.observation.set_disc_params(x0=650, y0=540, r0=200)
gui.observation.add_other_bodies_of_interest('Tethys')
gui.run()
