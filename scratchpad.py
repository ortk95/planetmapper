#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
from planetmapper import utils, mapper
import numpy as np
from functools import wraps

st = mapper.SpiceTool()
st.standardise_body_name('bob')