#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
from planetmapper import utils, mapper
import numpy as np
from functools import wraps


class C:
    CONST = False

    @classmethod
    def set_const(cls, v):
        cls.CONST = v

    def get_const(self):
        return self.CONST


a = C()
b = C()
print(a.get_const())
print(b.get_const())

a.set_const('abcdef')
print(a.get_const())
print(b.get_const())
