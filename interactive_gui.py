#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import sys
import tkinter as tk
from tkinter import ttk
from typing import TypeVar
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mapper

Widget = TypeVar('Widget', bound=tk.Widget)


def main(*args):

    io = InteractiveObservation(
        'data/jupiter.jpg', target='jupiter', utc=datetime.datetime(2020, 8, 25, 12)
    )
    io.run()


class InteractiveObservation:
    DEFAULT_GEOMETRY = '800x600+10+10'

    def __init__(self, *args, **kwargs) -> None:
        self.handles = []

        self.observation = mapper.Observation(*args, **kwargs)
        self.image = np.flipud(np.moveaxis(self.observation.data, 0, 2))

        self.step_size = 10

    def __repr__(self) -> str:
        return f'InteractiveObservation()'

    def run(self) -> None:
        self.build_gui()
        self.root.mainloop()

    # GUI Building
    def build_gui(self) -> None:
        self.root = tk.Tk()
        self.root.geometry(self.DEFAULT_GEOMETRY)
        self.root.title(self.observation.get_description(newline=False))
        self.configure_style()

        self.hint_frame = tk.Frame(self.root)
        self.hint_frame.pack(side='bottom', fill='x')
        self.build_help_hint()

        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side='left', fill='y')
        self.build_controls()

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side='top', fill='both', expand=True)
        self.build_plot()

    def configure_style(self) -> None:
        self.style = ttk.Style(self.root)
        self.style.theme_use('default')

    def build_controls(self) -> None:
        self.notebook = ttk.Notebook(self.controls_frame)
        self.notebook.pack(fill='both', expand=True)
        self.build_main_controls()
        self.build_settings_controls()

    def build_main_controls(self):
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Controls')

        ttk.Scale(
            frame,
            orient='horizontal',
            from_=0.1,
            to=10,
            value=10,
            command=self.set_step_size,
        ).pack()

        pos_controls = ttk.LabelFrame(frame, text='Position')
        pos_controls.pack(fill='x')

        rot_controls = ttk.LabelFrame(frame, text='Rotation')
        rot_controls.pack(fill='x')

        sz_controls = ttk.LabelFrame(frame, text='Size')
        sz_controls.pack(fill='x')

        for s, fn in (
            ('up ↑', self.move_up),
            ('down ↓', self.move_down),
            ('left ←', self.move_left),
            ('right →', self.move_right),
        ):
            self.add_tooltip(
                ttk.Button(pos_controls, text=s.capitalize(), command=fn),
                f'Move fitted disc {s}',
            ).pack()

        for s, fn in (
            ('clockwise ↻', self.rotate_right),
            ('anticlockwise ↺', self.rotate_left),
        ):
            self.add_tooltip(
                ttk.Button(rot_controls, text=s.capitalize(), command=fn),
                f'Rotate fitted disc {s}',
            ).pack()

        for s, fn in (
            ('increase +', self.increase_radius),
            ('decrease -', self.decrease_radius),
        ):
            self.add_tooltip(
                ttk.Button(sz_controls, text=s.capitalize(), command=fn),
                f'{s.capitalize()} fitted disc radius',
            ).pack()

    def build_settings_controls(self) -> None:
        settings = ttk.Frame(self.notebook)
        settings.pack()
        self.notebook.add(settings, text='Settings')

    def build_plot(self) -> None:
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot()

        self.ax.imshow(self.image, origin='lower', zorder=0)
        self.plot_wireframe()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def build_help_hint(self) -> None:
        self.help_hint = tk.Label(
            self.hint_frame, text='', foreground='black'
        )
        self.help_hint.pack(side='left')

    def add_tooltip(self, widget: Widget, msg: str) -> Widget:
        def f_enter(event):
            self.help_hint.configure(text=msg)

        def f_leave(event):
            self.help_hint.configure(text='')

        widget.bind('<Enter>', f_enter)
        widget.bind('<Leave>', f_leave)
        return widget

    def update_plot(self) -> None:
        while self.handles:
            self.handles.pop().remove()
        self.observation.update_transform()
        self.canvas.draw()
        print(
            'x0={x0}, y0={y0}, r0={r0}, rot={rot}'.format(
                x0=self.observation.get_x0(),
                y0=self.observation.get_y0(),
                r0=self.observation.get_r0(),
                rot=self.observation.get_rotation(),
            )
        )

    def plot_wireframe(self) -> None:
        ax = self.ax
        transform = self.observation.get_matplotlib_radec2xy_transform() + ax.transData

        limb_color = 'w'

        ax.plot(
            *self.observation.limb_radec(),
            color=limb_color,
            linewidth=0.5,
            transform=transform,
            zorder=5,
        )
        ax.plot(
            *self.observation.terminator_radec(),
            color=limb_color,
            linestyle='--',
            transform=transform,
            zorder=5,
        )

        (
            ra_day,
            dec_day,
            ra_night,
            dec_night,
        ) = self.observation.limb_radec_by_illumination()
        ax.plot(ra_day, dec_day, color=limb_color, transform=transform, zorder=5)

        for ra, dec in self.observation.visible_latlon_grid_radec(30):
            ax.plot(
                ra,
                dec,
                color='k',
                linestyle=':',
                transform=transform,
                zorder=4,
            )

        for lon, lat, s in self.observation.get_poles_to_plot():
            ra, dec = self.observation.lonlat2radec(lon, lat)
            ax.text(
                ra,
                dec,
                s,
                ha='center',
                va='center',
                weight='bold',
                color='k',
                path_effects=[
                    path_effects.Stroke(linewidth=3, foreground='w'),
                    path_effects.Normal(),
                ],
                transform=transform,
                zorder=5,
            )  # TODO make consistent with elsewhere

    # Buttons
    def move_up(self) -> None:
        self.observation.set_y0(self.observation.get_y0() + self.step_size)
        self.update_plot()

    def move_down(self) -> None:
        self.observation.set_y0(self.observation.get_y0() - self.step_size)
        self.update_plot()

    def move_right(self) -> None:
        self.observation.set_x0(self.observation.get_x0() + self.step_size)
        self.update_plot()

    def move_left(self) -> None:
        self.observation.set_x0(self.observation.get_x0() - self.step_size)
        self.update_plot()

    def rotate_left(self) -> None:
        self.observation.set_rotation(self.observation.get_rotation() - self.step_size)
        self.update_plot()

    def rotate_right(self) -> None:
        self.observation.set_rotation(self.observation.get_rotation() + self.step_size)
        self.update_plot()

    def increase_radius(self) -> None:
        self.observation.set_r0(self.observation.get_r0() + self.step_size)
        self.update_plot()

    def decrease_radius(self) -> None:
        self.observation.set_r0(self.observation.get_r0() - self.step_size)
        self.update_plot()

    def set_step_size(self, value: str) -> None:
        print(f'>> Setting step size to {value}')
        self.step_size = float(value)


if __name__ == '__main__':
    main(*sys.argv[1:])
