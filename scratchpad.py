#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import mapper
import datetime

import numpy as np

# testing case where no pole is visible
body = mapper.Body('jupiter', datetime.datetime.now(), 'europa')
body.plot_wirefeame_radec()