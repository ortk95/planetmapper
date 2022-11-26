#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for testing stuff during development (TODO delete in final version)"""
import tqdm
import tkinter as tk
from tkinter import ttk
from time import sleep


root = tk.Tk()
root.geometry('500x500+25+25')
style = ttk.Style(root)
style.theme_use('default')

style.theme_use('default')
for element in ['TEntry', 'TCombobox', 'TSpinbox', 'TButton', 'TLabel']:
    style.configure(
        element,
        foreground='black',
        insertcolor='black',
        fieldbackground='white',
        selectbackground='#bdf',
        selectforeground='black',
    )


frame = ttk.Frame(root)
frame.pack(fill='both', expand=True)


bar = ttk.Progressbar(frame)
bar.pack(fill='x', padx=10, pady=10)

def f():
    bar['value'] = 0
    for _ in tqdm.tqdm(range(100)):
        bar['value'] += 1
        root.update_idletasks()
        sleep(0.1)

ttk.Button(frame, text='Start', command=f).pack()


root.mainloop()
