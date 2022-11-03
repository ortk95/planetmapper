#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper
import numpy as np
from functools import wraps
import tools

times = [
    '2022-07-28T06:03:59.373',
    '2022-07-28T08:03:59.373',
]
body = planetmapper.BodyXY(
    'jupiter',
    times[0],
    observer='JWST',
    sz=50,
)
body.print_backplanes()

print(f'{len(body.backplanes)} backplanes currently registered:')
for bp in body.backplanes.values():
    print(f'    {bp.name}: {bp.description}')
