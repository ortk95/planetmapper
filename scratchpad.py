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
# body = planetmapper.Observation('data/europa.fits.gz')

body._set_progress_hook(planetmapper.progress.TqdmProgressHook())
utils.print_progress()
for bp in body.backplanes.values():
    bp.get_img()
    utils.print_progress(bp.name)

for bp in body.backplanes.values():
    bp.get_img()
    utils.print_progress(bp.name)
