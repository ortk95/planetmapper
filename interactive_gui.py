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
    io = InteractiveObservation()
    io.run()


class InteractiveObservation:
    def __init__(self) -> None:
        self.handles = []

        p = 'jupiter_test.jpg'
        self.observation = mapper.Observation(
            'jupiter', datetime.datetime(2020, 8, 25, 12)
        )
        self.image = np.flipud(plt.imread(p))

        self.observation.set_x0(self.image.shape[0] / 2)
        self.observation.set_y0(self.image.shape[1] / 2)
        self.observation.set_r0(self.image.shape[0] / 4)
        self.observation._set_rotation_radians(0)

        self.step_size = 10

        self.ax: Axes
        self.canvas: FigureCanvasTkAgg
        # print(self.observation.get_x0())
        # print(self.observation.get_y0())
        # print(self.observation.get_r0())
        # print(self.observation.get_rotation())

    def __repr__(self) -> str:
        return f'InteractiveObservation()'

    def run(self) -> None:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use('default')

        panel_bottom = tk.Frame(root)
        panel_bottom.pack(side='bottom', fill='x')
        help_hint = tk.Label(panel_bottom, text='', foreground='black')
        help_hint.pack()
        self.help_hint = help_hint

        panel_right = tk.Frame(root)
        panel_right.pack(side='right', fill='y')

        notebook = ttk.Notebook(panel_right)
        notebook.pack(fill='both', expand=True)

        controls = ttk.Frame(notebook)
        controls.pack()
        notebook.add(controls, text='Controls')

        settings = ttk.Frame(notebook)
        settings.pack()
        notebook.add(settings, text='Settings')

        # ttk.Scale(
        #     controls,
        #     orient='horizontal',
        #     # label='Step size',
        #     from_=-2,
        #     to=2,
        #     # resolution=1,
        #     # showvalue=False,
        # ).pack()
        # ttk.Spinbox(controls, values=('0.01', '0.1', '1', '10', '100')).pack()
        ttk.Scale(
            controls,
            orient='horizontal',
            # label='Step size',
            from_=0.1,
            to=10,
            value=10,
            # resolution=1,
            command=self.set_step_size,
        ).pack()

        pos_controls = ttk.LabelFrame(controls, text='Position')
        pos_controls.pack(fill='x')

        rot_controls = ttk.LabelFrame(controls, text='Rotation')
        rot_controls.pack(fill='x')

        sz_controls = ttk.LabelFrame(controls, text='Size')
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

        # ent = ttk.Entry(pos_controls, validate='key')
        # ent.pack()
        # ent.insert(0, '1345')

        fig = plt.figure(figsize=(5, 4), dpi=200)
        self.ax = fig.add_subplot()
        self.ax.imshow(self.image, origin='lower', zorder=0)
        self.plot_wireframe()

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()

        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        root.geometry('800x600+10+10')
        root.title(self.observation.get_description(newline=False))

        # self.replot()
        fig.tight_layout()
        root.mainloop()

    def add_tooltip(self, widget: Widget, msg: str) -> Widget:
        def f_enter(event):
            self.help_hint.configure(text=msg)

        def f_leave(event):
            self.help_hint.configure(text='')

        widget.bind('<Enter>', f_enter)
        widget.bind('<Leave>', f_leave)
        return widget

    def replot(self) -> None:
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
        transform = (
            self.observation._get_matplotlib_radec2xy_transform_radians() + ax.transData
        )

        ax.plot(
            *self.observation._limb_radec_radians(),
            color='w',
            linewidth=0.5,
            transform=transform,
            zorder=5,
        )
        ax.plot(
            *self.observation._terminator_radec_radians(),
            color='w',
            linestyle='--',
            transform=transform,
            zorder=5,
        )

        (
            ra_day,
            dec_day,
            ra_night,
            dec_night,
        ) = self.observation._limb_radec_by_illumination_radians()
        ax.plot(ra_day, dec_day, color='w', transform=transform, zorder=5)

        for ra, dec in self.observation.visible_latlon_grid_radec(30):
            ax.plot(
                np.deg2rad(ra),
                np.deg2rad(dec),
                color='k',
                linestyle=':',
                transform=transform,
                zorder=4,
            )
        print(ra_day[0], dec_day[0])
        for lon, lat, s in ((0, 90, 'N'), (0, -90, 'S')):
            if self.observation.test_if_lonlat_visible(lon, lat):
                ra, dec = self.observation._lonlat2radec_radians(
                    np.deg2rad(lon), np.deg2rad(lat)
                )
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

    def move_up(self) -> None:
        self.observation.set_y0(self.observation.get_y0() + self.step_size)
        self.replot()

    def move_down(self) -> None:
        self.observation.set_y0(self.observation.get_y0() - self.step_size)
        self.replot()

    def move_right(self) -> None:
        self.observation.set_x0(self.observation.get_x0() + self.step_size)
        self.replot()

    def move_left(self) -> None:
        self.observation.set_x0(self.observation.get_x0() - self.step_size)
        self.replot()

    def rotate_left(self) -> None:
        self.observation.set_rotation(self.observation.get_rotation() - self.step_size)
        self.replot()

    def rotate_right(self) -> None:
        self.observation.set_rotation(self.observation.get_rotation() + self.step_size)
        self.replot()

    def increase_radius(self) -> None:
        self.observation.set_r0(self.observation.get_r0() + self.step_size)
        self.replot()

    def decrease_radius(self) -> None:
        self.observation.set_r0(self.observation.get_r0() - self.step_size)
        self.replot()

    def set_step_size(self, value: str) -> None:
        print(f'>> Setting step size to {value}')
        self.step_size = float(value)


if __name__ == '__main__':
    main(*sys.argv[1:])
