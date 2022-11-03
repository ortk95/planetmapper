import datetime
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
from typing import TypeVar, Callable, Any
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import functools
from . import utils
from .observation import Observation

Widget = TypeVar('Widget', bound=tk.Widget)


class InteractiveObservation:
    DEFAULT_GEOMETRY = '800x600+10+10'

    def __init__(self, path: str | None = None, *args, **kwargs) -> None:
        if path is None:
            path = tkinter.filedialog.askopenfilename(title='Open FITS file')
            # TODO add configuration for target, date etc.
        self.observation = Observation(path, *args, **kwargs)

        self.image = np.flipud(np.moveaxis(self.observation.data, 0, 2))
        if self.image.shape[2] != 3:
            self.image = np.nansum(self.image, axis=2)
            # TODO get image better

        self.step_size = 10

        self.shortcuts: dict[Callable[[], Any], list[str]] = {
            self.increase_step: [']'],
            self.decrease_step: ['['],
            self.move_up: ['<Up>', 'w'],
            self.move_down: ['<Down>', 's'],
            self.move_right: ['<Right>', 'd'],
            self.move_left: ['<Left>', 'a'],
            self.rotate_right: ['>', '.'],
            self.rotate_left: ['<less>', ','],
            self.increase_radius: ['+', '='],
            self.decrease_radius: ['-', '_'],
            # self.observation.centre_disc: ['<Control-c>'],
        }
        self.handles = []

    def __repr__(self) -> str:
        return f'InteractiveObservation()'

    def run(self) -> None:
        self.build_gui()
        self.bind_keyboard()
        self.root.mainloop()
        # TODO do something when closed to kill figure etc.?

    # GUI Building
    def build_gui(self) -> None:
        self.root = tk.Tk()
        self.root.geometry(self.DEFAULT_GEOMETRY)
        self.root.title(self.observation.get_description(multiline=False))
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
        # self.build_settings_controls()

    def build_main_controls(self):
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Controls')

        # step_size_frame = ttk.LabelFrame(frame, text='Step size')
        # step_size_frame.pack(fill='x')

        pos_frame = ttk.LabelFrame(frame, text='Position')
        pos_frame.pack(fill='x')

        rot_frame = ttk.LabelFrame(frame, text='Rotation')
        rot_frame.pack(fill='x')

        sz_frame = ttk.LabelFrame(frame, text='Size')
        sz_frame.pack(fill='x')

        save_frame = ttk.LabelFrame(frame, text='Output')
        save_frame.pack(fill='x')

        # ttk.Scale(
        #     step_size_frame,
        #     orient='horizontal',
        #     from_=0.1,
        #     to=10,
        #     value=10,
        #     command=self.set_step_size,
        # ).pack()

        pos_button_frame = ttk.Frame(pos_frame)
        pos_button_frame.pack()
        for arrow, hint, fn, column, row in (
            ('↑', 'up', self.move_up, 1, 0),
            ('↗', 'up and right', self.move_up_right, 2, 0),
            ('→', 'right', self.move_right, 2, 1),
            ('↘', 'down and right', self.move_down_right, 2, 2),
            ('↓', 'down', self.move_down, 1, 2),
            ('↙', 'down and left', self.move_down_left, 0, 2),
            ('←', 'left', self.move_left, 0, 1),
            ('↖', 'up and left', self.move_up_left, 0, 0),
        ):
            self.add_tooltip(
                ttk.Button(pos_button_frame, text=arrow, command=fn, width=2),
                f'Move fitted disc {hint}',
            ).grid(column=column, row=row, ipadx=5, ipady=5)

        for arrow, fn in (
            ('clockwise ↻', self.rotate_right),
            ('anticlockwise ↺', self.rotate_left),
        ):
            self.add_tooltip(
                ttk.Button(rot_frame, text=arrow.capitalize(), command=fn),
                f'Rotate fitted disc {arrow}',
            ).pack()

        for arrow, fn in (
            ('increase +', self.increase_radius),
            ('decrease -', self.decrease_radius),
        ):
            self.add_tooltip(
                ttk.Button(sz_frame, text=arrow.capitalize(), command=fn),
                f'{arrow.capitalize()} fitted disc radius',
            ).pack()

        self.add_tooltip(
            ttk.Button(save_frame, text='Save', command=self.save),
            f'Save FITS file with backplane data',
        ).pack()

    def build_settings_controls(self) -> None:
        settings = ttk.Frame(self.notebook)
        settings.pack()
        self.notebook.add(settings, text='Settings')

    def build_plot(self) -> None:
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot()

        self.ax.imshow(self.image, origin='lower', zorder=0)
        self.ax.set_xlim(-0.5, self.observation._nx - 0.5)
        self.ax.set_ylim(-0.5, self.observation._ny - 0.5)
        self.plot_wireframe()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def build_help_hint(self) -> None:
        self.help_hint = tk.Label(self.hint_frame, text='', foreground='black')
        self.help_hint.pack(side='left')
        # TODO add keybinginds to hint

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
            'x0={x0}, y0={y0}, r0={r0}, rotation={rotation}'.format(
                x0=self.observation.get_x0(),
                y0=self.observation.get_y0(),
                r0=self.observation.get_r0(),
                rotation=self.observation.get_rotation(),
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
            )
        for body in self.observation.other_bodies:
            ra = body.target_ra
            dec = body.target_dec
            ax.text(
                ra,
                dec,
                body.target + '\n',
                size='small',
                ha='center',
                va='center',
                color='grey',
                transform=transform,
                clip_on=True,
            )
            ax.scatter(
                ra,
                dec,
                marker='+',  # type: ignore
                color='w',
                transform=transform,
            )
        # TODO make this code consistent with elsewhere?

    # Keybindings
    def bind_keyboard(self) -> None:
        for fn, events in self.shortcuts.items():
            handler = lambda e, f=fn: f()
            for event in events:
                self.root.bind(event, handler)

    # Buttons
    def increase_step(self) -> None:
        self.step_size *= 10
        print(self.step_size)

    def decrease_step(self) -> None:
        self.step_size /= 10
        print(self.step_size)

    def move_up(self) -> None:
        self.observation.set_y0(self.observation.get_y0() + self.step_size)
        self.update_plot()

    def move_up_right(self) -> None:
        self.observation.set_y0(self.observation.get_y0() + self.step_size)
        self.observation.set_x0(self.observation.get_x0() + self.step_size)
        self.update_plot()

    def move_right(self) -> None:
        self.observation.set_x0(self.observation.get_x0() + self.step_size)
        self.update_plot()

    def move_down_right(self) -> None:
        self.observation.set_y0(self.observation.get_y0() - self.step_size)
        self.observation.set_x0(self.observation.get_x0() + self.step_size)
        self.update_plot()

    def move_down(self) -> None:
        self.observation.set_y0(self.observation.get_y0() - self.step_size)
        self.update_plot()

    def move_down_left(self) -> None:
        self.observation.set_y0(self.observation.get_y0() - self.step_size)
        self.observation.set_x0(self.observation.get_x0() - self.step_size)
        self.update_plot()

    def move_left(self) -> None:
        self.observation.set_x0(self.observation.get_x0() - self.step_size)
        self.update_plot()

    def move_up_left(self) -> None:
        self.observation.set_y0(self.observation.get_y0() + self.step_size)
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

    def save(self) -> None:
        path = tkinter.filedialog.asksaveasfilename(
            title='Save FITS file',
            parent=self.root,
            # defaultextension='.fits',
            initialfile=self.observation.make_filename(),
            # filetypes=[
            #     ('FITS', '*.fits'),
            #     ('Compressed FITS', '*.fits.gz')
            # ],
        )
        # TODO add some validation
        # TODO add some progress UI
        print(path)
        utils.print_progress(c1='c')
        self.observation.save(path)
        utils.print_progress('saved', c1='c')
