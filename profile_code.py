#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for profiling code during development (TODO delete in final version)"""
import line_profiler
from inspect import isclass, isfunction
import utils

print_progress = lambda x='': utils.print_progress(x, c1='c')
print_progress('importing...')
import mapper

print_progress('setting up...')

# FUNCTION TO TIME ---------------------------------------------------------------------


def fn():
    # p = '/Users/ortk1/Dropbox/PhD/data/reduced/sphere_irdis/europa/combined/SPHER.2014-12-09T075436.092_irdis.fits.gz'
    # o = mapper.Observation.from_fits(p)
    utils.print_progress()
    for abcorr in ['NONE', 'CN', 'CN+S', 'LT', 'LT+S']:
        o = mapper.BodyXY(
            'Jupiter', '2022-01-01', nx=100, ny=100, aberration_correction=abcorr
        )
        img = o.get_lon_img()
        utils.print_progress(abcorr)

    # o = mapper.BodyXY('Jupiter', '2022-01-01', nx=100, ny=100)
    # img = o.get_lon_img()
    # utils.print_progress('lon')

    # o = mapper.BodyXY('Jupiter', '2022-01-01', nx=100, ny=100, aberration_correction='LT')
    # img = o.get_lon_img()
    # utils.print_progress('lon')

    # img = o.get_ra_img()
    # utils.print_progress('ra')

    # img = o.get_doppler_img()
    # utils.print_progress('doppler')


# SETTINGS -----------------------------------------------------------------------------

output_unit = 1
stripzeros = True
cutoff = 10

# TIMING INTERNALS ---------------------------------------------------------------------

lp = line_profiler.LineProfiler()

objects_to_profile = [
    mapper.Body,
    mapper.BodyXY,
    mapper.Observation,
]

for obj in objects_to_profile:
    for k, v in obj.__dict__.items():
        if isfunction(v):
            lp.add_function(v)

lp.add_function(fn)
wrapped = lp(fn)
print_progress('running...')
wrapped()
print_progress('processing stats...')

stats = lp.get_stats()
total_time = lambda x: sum(l[2] for l in x)

print_progress('printing stats...')

for (fn, lineno, name), timings in sorted(
    stats.timings.items(), key=lambda x: total_time(x[1])
):
    if total_time(timings) * stats.unit < cutoff:
        continue
    line_profiler.show_func(
        fn,
        lineno,
        name,
        timings,
        stats.unit,
        output_unit=output_unit,
        stripzeros=stripzeros,
    )
