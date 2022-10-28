#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import utils
from multiprocessing import Pool
import time
import spiceypy as spice
import numpy as np
import ctypes

mapper.SpiceTool.load_spice_kernels()

et = spice.utc2et('2022-01-01')
target = 'jupiter'
observer_frame = 'J2000'
aberration_correction = 'CN+S'
observer = 'earth'

functions = [
    lambda x: x.encode(),
    lambda x: x,
    # lambda x: bytearray([1,2,3]),
]
for fn in functions:
    starg, lt = spice.spkezr(
        fn(target),
        et,
        fn(observer_frame),
        fn(aberration_correction),
        fn(observer),
    )
    print(starg, lt)

# for fn in functions:
#     starg = spice.stypes.empty_double_vector(6)
#     lt = ctypes.c_double()

#     spice.libspice.spkezr_c(
#         fn(target),
#         ctypes.c_double(et),
#         fn(observer_frame),
#         fn(aberration_correction),
#         fn(observer),
#         starg,
#         ctypes.byref(lt),
#     )
#     print('>', starg, lt)
