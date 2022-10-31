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
    nn = 100
    utils.print_progress()
    o = mapper.BodyXY('Jupiter', '2022-01-01', nx=nn, ny=nn)
    utils.print_progress('__init__')
    img = o.get_lon_img()
    utils.print_progress('img')


# SETTINGS -----------------------------------------------------------------------------

output_unit = 1
stripzeros = True
cutoff = 0

# TIMING INTERNALS ---------------------------------------------------------------------

lp = line_profiler.LineProfiler()
lp.add_function(mapper.BodyXY._get_targvec_img)
objects_to_profile = [
    # mapper.Body,
    # mapper.BodyXY,
    # mapper.Observation,
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

skipped = 0
for (fn, lineno, name), timings in sorted(
    stats.timings.items(), key=lambda x: total_time(x[1])
):
    if total_time(timings) * stats.unit < cutoff:
        skipped += 1
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

if skipped:
    utils.cprint(f'{skipped} functions skipped due to cutoff', fg='m')
