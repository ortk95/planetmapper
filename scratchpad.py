#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper.data_loader
import planetmapper
import numpy as np
import datetime

body = planetmapper.Body('JUPITER', datetime.datetime.now())
body.plot_wireframe_radec()
body.ring_radii.add(122500)
body.ring_radii.add(129000)
body.plot_wireframe_radec()

body = planetmapper.Body('SATURN', datetime.datetime.now())
body.plot_wireframe_radec()
