#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import numpy as np
import utils

# utils.print_progress()
# body = mapper.BodyXY(
#     'jupiter',
#     '2022-07-28T06:03:59.373',
#     sz=50,
#     observer='JWST',
# )
# ax = body.plot_backplane('radial_velocity')
# ax = body.plot_backplane('doppler')

# img = body.get_backplane_img('doppler')

# fmt = '.3e'
# print('Avg doppler shift:', format(np.nanmean(img), fmt))
# print('Min doppler shift:', format(np.nanmin(img), fmt))
# print('Max doppler shift:', format(np.nanmax(img), fmt))

body = mapper.BodyXY('saturn', '2001-02-03', nx=4, ny=6)
body.set_x0(2)
body.set_y0(4.5)
body.set_r0(3.14)
body.set_rotation(42)

for bp in body.backplanes:
    s = '{bp}: {img},'.format(
        bp=repr(bp),
        img=repr(body.get_backplane_img(bp))
    )
    print(s)