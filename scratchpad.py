#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import planetmapper
import matplotlib.pyplot as plt

p = '/Users/ortk1/Desktop/iew608inq_drz.fits'
observation = planetmapper.Observation(p, aberration_correction='CN', target='uranus')
observation.run_gui()