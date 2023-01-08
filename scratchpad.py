#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt


# for rotation, color in [(0, 'k'), (90, 'r')]:
#     body = planetmapper.BodyXY('Uranus', '2023-01-07 06:11:32', observer='HST', sz=50)
#     body.set_rotation(rotation)
#     body.plot_wireframe_xy(color=color)

# plt.show()
p = '/Users/ortk1/Desktop/iew608inq_drz.fits'
observation = planetmapper.Observation(p)