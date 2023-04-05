#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.io import fits

observation = planetmapper.Observation('~/Downloads/test.fits')
observation.run_gui()
