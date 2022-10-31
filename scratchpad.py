#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import mapper
import numpy as np
import utils
from functools import wraps


class C:
    @staticmethod
    def add_method(fn):
        # @wraps(
        #     fn,
        # )
        def wrapped(
            self,
            *args,
            method='spam',
            **kwargs,
        ):
            print(args, kwargs)
            self.set_method(method)
            return fn(self, *args, **kwargs)

        return wrapped

    @add_method
    def set_x0(self, x0: float):
        print('> setting x0', x0)

    def set_method(self, method):
        print('method', method)


c = C()


c.set_x0(123, method='123')
