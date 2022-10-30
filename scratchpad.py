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
import datetime
import utils

import glob
import os

kernel_path = mapper.KERNEL_PATH

kernel_path = os.path.expanduser(kernel_path)
pcks = sorted(glob.glob(kernel_path + 'pck/*.tpc'))
spks1 = sorted(glob.glob(kernel_path + 'spk/planets/de*.bsp'))
spks2 = sorted(glob.glob(kernel_path + 'spk/satellites/*.bsp'))
fks = sorted(glob.glob(kernel_path + 'fk/planets/*.tf'))
lsks = sorted(glob.glob(kernel_path + 'lsk/naif*.tls'))
kernels = [pcks[-1], spks1[-1], *spks2, lsks[-1]]
for kernel in kernels:
    print(kernel)
