#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import numpy as np

body = mapper.BodyXY(
    'jupiter',
    '2022-07-28T06:03:59.373',
    sz=25,
    observer='JWST',
)
ax = body.plot_backplane('radial_velocity')
ax = body.plot_backplane('doppler')

img = body.get_backplane_img('doppler')

fmt  = '.3e'
print('Avg doppler shift:', format(np.nanmean(img), fmt))
print('Min doppler shift:', format(np.nanmin(img), fmt))
print('Max doppler shift:', format(np.nanmax(img), fmt))

