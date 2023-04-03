#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt
import numpy as np

planetmapper.set_kernel_path(
    '/Users/ortk1/Dropbox/science/planetmapper/tests_new/data/kernels'
)

body = planetmapper.Body('Jupiter', observer='HST', utc='2005-01-01T00:00:00')
