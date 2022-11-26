#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development"""
import tqdm
import tkinter as tk
from tkinter import ttk
from time import sleep
import planetmapper
import planetmapper.progress
import matplotlib.pyplot as plt
from planetmapper import utils

body = planetmapper.BodyXY('Saturn', '2022-01-01', sz=500)
body = planetmapper.Observation('data/europa.fits.gz', show_progress=True)

body.save_mapped_observation('data/test.fits.gz')
