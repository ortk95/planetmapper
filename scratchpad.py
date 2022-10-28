#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import utils
from multiprocessing import Pool
import time
import spiceypy as spice
import numpy as np


def fn(x):
    # print('running for >', x)
    # time.sleep(1)
    # return x * x
    body = mapper.Body.standardise_body_name('jupiter')
    try:
        spoint, trgepc, srfvec = spice.sincpt(
            'ELLIPSOID',
            'JUPITER',
            720239408.9007025,
            'IAU_JUPITER',
            'CN+S',
            'EARTH',
            'J2000',
            np.array([6.14248967e08, 3.25492730e06, -1.69053304e07]),
        )
    except mapper.NotFoundError:
        return None
    return spoint


def init():
    mapper.Body.load_spice_kernels()


def main():
    n = int(5e5)
    utils.print_progress(format(n, '.0e'))
    init()
    args = list(range(n))
    utils.print_progress('set up')

    with Pool(5, initializer=init) as pool:
        utils.print_progress('set up pool')
        results = list(pool.map(fn, args, chunksize=10000))
        # print(results)
        utils.print_progress('parallel')
    utils.print_progress('pool closed')
    results = list(map(fn, args))
    # print(results)
    utils.print_progress('serial')


if __name__ == '__main__':
    main()
