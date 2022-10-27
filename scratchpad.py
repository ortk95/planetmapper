#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
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


class C:
    DEFAULT_BACKPLANES = []

    @staticmethod
    def backplane(name: str, description: str):
        def decorator(fn: Callable[P, np.ndarray]) -> Callable[P, np.ndarray]:
            C.DEFAULT_BACKPLANES.append((fn, name, description))
            print(name)
            @wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs):
                return fn(*args, **kwargs)

            wrapper.name = name
            wrapper.description = description
            return wrapper

        return decorator

    @backplane('x', 'y')
    def img(self):
        return np.random.rand(10, 10)

print(0)
c = C()
print(1)
plt.imshow(c.img())
print(c.img.description)
