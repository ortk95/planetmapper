#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import mapper
import datetime
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import tqdm

body = mapper.Body('moon', datetime.datetime.now())
ax = body.plot_wirefeame_radec(show=False)

state, lt = spice.spkezr(
    body.target,
    body.et,
    body.observer_frame,
    body.aberration_correction,
    body.observer,
)
obsvec = state[:3]
velocity = state[3:]
v_target = np.dot(obsvec, velocity) / np.linalg.norm(obsvec)
print(v_target)

print(spice.vdot(velocity, spice.vhat(obsvec)))


# lons = np.linspace(0, 360, 100)
# lats = np.linspace(-80, 80, 50)

# ra_vals = []
# dec_vals = []
# v_vals = []

# for lon in tqdm.tqdm(lons):
#     for lat in lats:
#         if body.test_if_lonlat_visible_degrees(lon, lat):
#             targvec = body.lonlat2targvec(*body._degree_pair2radians(lon, lat))
#             state, lt = spice.spkcpt(
#                 trgpos=targvec,
#                 trgctr=body.target,
#                 trgref=body.target_frame,
#                 et=body.et,
#                 outref=body.observer_frame,
#                 refloc='OBSERVER',
#                 abcorr=body.aberration_correction,
#                 obsrvr=body.observer,
#             )
#             obsvec = state[:3]
#             velocity = state[3:]

#             v = np.dot(obsvec, velocity) / np.linalg.norm(obsvec)

#             ra, dec = body.lonlat2radec_degrees(lon, lat)
#             ra_vals.append(ra)
#             dec_vals.append(dec)
#             v_vals.append(v)

# sc = ax.scatter(ra_vals, dec_vals, c=v_vals, cmap='turbo', marker='.', zorder=0)
# plt.colorbar(sc)
# plt.show()
# print('v_target', v_target, 'km/s')
