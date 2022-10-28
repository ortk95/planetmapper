#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import datetime
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from typing import Callable, Iterable, TypeVar, ParamSpec, cast, Any
from functools import wraps

T = TypeVar('T')
P = ParamSpec('P')


class A:
    def __init__(self, x: int):
        print(x)


class B(A):
    def __init__(self, x):
        super().__init__(x=x)
