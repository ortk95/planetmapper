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


target = 'jupiter'
observer_frame = 'J2000'
aberration_correction = 'CN+S'
observer = 'earth'


class BodySlow(mapper.Body):
    @staticmethod
    def _encode_str(s: str):
        return s


body1 = mapper.Body('jupiter', '2022-01-01')
body2 = BodySlow('jupiter', '2022-01-01')
