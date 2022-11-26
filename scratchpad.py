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
body = planetmapper.Observation(
    'data/jupiter_small.jpg', target='jupiter', utc='2022-01-01'
)
# body = planetmapper.Observation(
#     '/Users/ortk1/Dropbox/PhD/data/jwst/saturn/SATURN-75N/stage3/d1_fringe_nav/Level3_ch1-long_s3d_nav.fits'
# )
# body._set_progress_hook(planetmapper.progress.CLIProgressHook(leave=True))
body.save_observation('data/test.fits.gz', show_progress=True, print_info=True)
