#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for timing code snippets. (TODO delete this in final version)

Specify the snippets in the SETUP section below, then run the script to
calculate the timings. Adjust the value of repeat and number as appropriate to
perform the timings in a reasonable amount of time. The snippets timed to see
how long they take to execute `number` times, and the fastest time from
`repeat` repeats is taken as the optimal time for that snippet.
"""
import timeit
import math
import utils
import numpy as np

# SETUP --------------------------------------------------------------------------------
# Adjust repeat & number as needed to ensure the timing doesn't take too long. Higher
# values will take longer but will generally give more reliable results.
repeat = 100  # Number of times to repeat timing loop
number = 100  # Number of times statement is called in each timing loop


# Define any variables, module imports etc. to use in the snippets here...
import math

x, y, x0, y0 = np.random.rand(4)
r2 = np.random.rand() ** 2


def f1():
    return (x - x0) ** 2 + (y - y0) ** 2 > r2


def f2():
    return (x - x0) * (x - x0) + (y - y0) * (y - y0) > r2


def f3():
    dx = x - x0
    dy = y - y0
    return dx * dx + dy * dy > r2


def f4():
    dx = x - x0
    if dx*dx > r2:
        return True
    dy = y - y0
    if dy*dy > r2:
        return True
    return dx * dx + dy * dy > r2


def f5():
    dx = x - x0
    dy = y - y0
    return dx * dx > r2 or dy * dy > r2 or dx * dx + dy * dy > r2


# Define code snippets as a list of strings to execute here...
statements = [
    '(x - x0)**2 + (y - y0)**2 > r2',
    'f1()',
    'f2()',
    'f3()',
    'f4()',
    'f5()',
]

statements = ['out = ' + s for s in statements]
# statements = [f'out = str({s})' for s in statements]
# statements = ['out = sum(' + s + ')' for s in statements]
# statements = ['out = np.sum(' + s + ')' for s in statements]
# statements = [f'out = list(({s})[:2])' for s in statements]


# PERFORM TIMINGS AND ANALYSIS ---------------------------------------------------------
# Nothing below here should generally need modifying

global_vals = locals()  # Get copy of local variables for use in function statements

print(f'Testing code timings with repeats={repeat} and number={number}')
print()

check_output = all(('out = ' in s or 'out=' in s) for s in statements)
if check_output:
    print(
        f'{"TIME (s)":^12s}|{"COMMAND":^{max(len(s) for s in statements)+2}s}| OUTPUT'
    )
else:
    print(f'{"TIME (s)":^12s}|{"COMMAND":^{max(len(s) for s in statements)+2}s}')

times = []
out_old = None
for s in statements:
    t = (
        min(timeit.repeat(s, repeat=repeat, number=number, globals=global_vals))
        / number
    )
    times.append(t)

    if check_output:
        gv = global_vals.copy()
        exec(s, gv)
        out = gv['out']
        if isinstance(out, np.ndarray):
            out = str(out)  # hack to work with numpy array comparison
        print(f'{t:.5e} | {s:{max(len(s) for s in statements)}s} | ', end='')
        if out_old is None:
            fg = None
        elif out == out_old or out is out_old:
            fg = 'g'
        else:
            fg = 'r'
            try:
                if math.isclose(out, out_old):  # type: ignore
                    fg = 'c'
                elif math.isclose(out, out_old, rel_tol=1e-6, abs_tol=1e-9):  # type: ignore
                    fg = 'm'
            except TypeError:
                pass
        out_old = out
        utils.cprint(f'{repr(out)}', fg=fg, style='b', flush=True)
    else:
        print(f'{t:.5e} | {s}', flush=True)

print()
utils.print_bar_chart(statements, times, sort=True)