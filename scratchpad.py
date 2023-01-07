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
import numpy as np


body = planetmapper.Body('Uranus')
body.add_other_bodies_of_interest(*range(701, 711))

body.plot_wireframe_radec()