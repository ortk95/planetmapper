#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import planetmapper.data_loader
import planetmapper
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.patheffects as path_effects

planetmapper.utils.print_progress()
gui = planetmapper.gui.GUI(
    'data/saturn.jpg',
    target='saturn',
    utc='2001-12-08T04:39:30.449',
)
gui.observation.set_disc_params(x0=650, y0=540, r0=200)
gui.observation.add_other_bodies_of_interest('Tethys')
gui.run()


# import tkinter as tk
# from tkinter import ttk

# root = tk.Tk()
# root.geometry('500x500')

# frame = ttk.Frame(root).pack()


# s = ttk.Style(root)
# # s.theme_use('default')
# for element in ['TEntry', 'TCombobox', 'TSpinbox', 'TButton']:
#     s.configure(
#         element,
#         foreground='black',
#         insertcolor='black',
#         fieldbackground='white',
#         selectbackground='#b3d8ff',
#         selectforeground='black',
#     )
# # s.configure('TEntry', foreground='red', insertcolor='black')
# # s.configure('TCombobox', foreground='green')
# # s.configure('TSpinbox', foreground='blue')


# sv = tk.StringVar(value='abcdef')
# ent = ttk.Entry(frame, textvariable=sv)
# ent.pack()
# ttk.Combobox(frame, textvariable=sv).pack()
# ttk.Spinbox(frame, textvariable=sv).pack()
# ttk.Button(frame, text='bjkdhdjfh').pack()
# root.mainloop()
