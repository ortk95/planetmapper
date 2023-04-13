#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.io import fits


image = np.random.rand(100, 100)
observation = planetmapper.Observation(data=image, target='JUPITER')
observation.run_gui()
