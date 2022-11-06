#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper.data
import planetmapper
import numpy as np
import datetime

body = planetmapper.Body('SATURN', datetime.datetime.now())

body.ring_radii.add(122340)  # Add new ring radius to plot
body.ring_radii.add(136780)  # Add new ring radius to plot


body.plot_wireframe_radec()
