#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper


body = mapper.BodyXY(
    'jupiter', '2022-07-28T06:03:59.373', sz=25, observer='JWST'
)
ax = body.plot_backplane('radial_velocity')
ax = body.plot_backplane('doppler')
