#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import mapper
import datetime

obs = mapper.Observation('jupiter', datetime.datetime.now(), 'europa')
obs.plot_wirefeame_radec()
obs.plot_wirefeame_xy()