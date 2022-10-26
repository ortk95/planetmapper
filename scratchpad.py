#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import mapper
import datetime
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import tqdm




class C:

    def __init__(self):
        self._cache = {}

    @staticmethod
    def decorator(fn):
        def decorated(self, *args, **kwargs):
            fn_name = fn.__name__
            print('decorated', fn_name)
            if fn_name not in self._cache:
                self._cache[fn_name] = fn(self, *args, **kwargs)
            return self._cache[fn_name]
        return decorated


    @decorator
    def f(self):
        print('calling f')
        return 1234567890


c = C()

c.f()
c.f()
c.f()

c._cache.clear()

c.f()
c.f()
c.f()
