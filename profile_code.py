#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for profiling code during development (TODO delete in final version)"""
import cProfile
import pstats
import planetmapper
from planetmapper import utils

nn = 50
with cProfile.Profile() as pr:
    utils.print_progress()
    o = planetmapper.BodyXY('Jupiter', '2022-01-01', nx=nn, ny=nn)
    o.set_r0(10)
    utils.print_progress('__init__')
    img = o.get_lon_img()
    utils.print_progress('img')

stats = pstats.Stats(pr)

stats.strip_dirs()
stats.sort_stats('time')
stats.print_stats(0.1)
# stats.print_callers(0.1)
o.plot_wireframe_xy()
