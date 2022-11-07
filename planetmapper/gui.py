import datetime
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.colorchooser
from typing import TypeVar, Callable, Any, Literal, TypeAlias
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import utils
from .observation import Observation

Widget = TypeVar('Widget', bound=tk.Widget)
SETTER_KEY = Literal['x0', 'y0', 'r0', 'rotation', 'step']
PLOT_KEY = Literal[
    'image',
    'limb',
    'limb_dayside',
    'terminator',
    'grid',
    'rings',
    'poles',
    'coordinates_lonlat',
    'coordinates_radec',
    'other_bodies',
    'other_bodies_labels',
]

DEFAULT_PLOT_SETTINGS: dict[PLOT_KEY, dict] = {
    'limb': dict(color='w', linewidth=0.5, linestyle='-'),
    'limb_dayside': dict(color='w', linewidth=1, linestyle='-'),
    'terminator': dict(color='w', linewidth=1, linestyle='--'),
    'grid': dict(color='dimgray', linewidth=1, linestyle=':'),
    'rings': dict(color='w', linewidth=0.5, linestyle='-'),
    'poles': dict(color='k', outline_color='w'),
    'other_bodies_labels': dict(color='grey'),
}


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

        self.setter_callbacks: dict[SETTER_KEY, list[Callable[[float], Any]]] = {
            'x0': [self.observation.set_x0],
            'y0': [self.observation.set_y0],
            'r0': [self.observation.set_r0],
            'rotation': [self.observation.set_rotation],
            'step': [self.set_step, lambda s: print(s)],
        }
        self.getters: dict[SETTER_KEY, Callable[[], float]] = {
            'x0': self.observation.get_x0,
            'y0': self.observation.get_y0,
            'r0': self.observation.get_r0,
            'rotation': self.observation.get_rotation,
            'step': lambda: self.step_size,
        }
        self.plot_handles: defaultdict[PLOT_KEY, list[Artist]] = defaultdict(list)
        self.plot_settings: defaultdict[PLOT_KEY, dict] = defaultdict(dict)
        for k, v in DEFAULT_PLOT_SETTINGS.items():
            self.plot_settings[k] = v.copy()

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

        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side='left', fill='y')

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side='top', fill='both', expand=True)

        self.build_plot()
        self.build_help_hint()
        self.build_controls()
        self.update_plot()

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

        # Position controls
        label_frame = ttk.LabelFrame(frame, text='Position')
        label_frame.pack(fill='x')

        entry_frame = self.add_tooltip(
            ttk.Frame(label_frame), 'Set pixel coordinates of the centre of the disc'
        )
        entry_frame.pack()
        NumericEntry(self, entry_frame, 'x0')
        NumericEntry(self, entry_frame, 'y0')

        button_frame = ttk.Frame(label_frame)
        button_frame.pack()
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
                ttk.Button(button_frame, text=arrow, command=fn, width=2),
                f'Move fitted disc {hint}',
            ).grid(column=column, row=row, ipadx=5, ipady=5)

        # Rotation controls
        label_frame = ttk.LabelFrame(frame, text='Rotation')
        label_frame.pack(fill='x')

        entry_frame = self.add_tooltip(
            ttk.Frame(label_frame), 'Set the rotation (in degrees) of the disc'
        )
        entry_frame.pack()
        NumericEntry(self, entry_frame, 'rotation', label='Rotation (°)')

        button_frame = ttk.Frame(label_frame)
        button_frame.pack()
        for arrow, hint, fn, column in (
            ('↻', 'clockwise', self.rotate_right, 0),
            ('↺', 'anticlockwise', self.rotate_left, 1),
        ):
            self.add_tooltip(
                ttk.Button(button_frame, text=arrow.capitalize(), command=fn, width=2),
                f'Rotate fitted disc {hint}',
            ).grid(column=column, row=0, ipadx=5, ipady=5)

        # Size controls
        label_frame = ttk.LabelFrame(frame, text='Size')
        label_frame.pack(fill='x')

        entry_frame = self.add_tooltip(
            ttk.Frame(label_frame), 'Set the (equatorial) radius in pixels of the disc'
        )
        entry_frame.pack()
        NumericEntry(self, entry_frame, 'r0')
        # TODO add plate scale option

        button_frame = ttk.Frame(label_frame)
        button_frame.pack()
        for arrow, hint, fn, column in (
            ('-', 'decrease', self.decrease_radius, 0),
            ('+', 'increase', self.increase_radius, 1),
        ):
            self.add_tooltip(
                ttk.Button(button_frame, text=arrow.capitalize(), command=fn, width=2),
                f'{hint.capitalize()} fitted disc radius',
            ).grid(column=column, row=0, ipadx=5, ipady=5)

        # Step controls
        label_frame = ttk.LabelFrame(frame, text='Step size')
        label_frame.pack(fill='x')

        entry_frame = self.add_tooltip(
            ttk.Frame(label_frame), 'Set the step size when clicking buttons'
        )
        entry_frame.pack()
        NumericEntry(self, entry_frame, 'step')

        button_frame = ttk.Frame(label_frame)
        button_frame.pack()
        for arrow, hint, fn, column in (
            ('÷', 'decrease', self.decrease_step, 0),
            ('×', 'increase', self.increase_step, 1),
        ):
            self.add_tooltip(
                ttk.Button(button_frame, text=arrow.capitalize(), command=fn, width=2),
                f'{hint.capitalize()} step size',
            ).grid(column=column, row=0, ipadx=5, ipady=5)

        # IO controls
        label_frame = ttk.LabelFrame(frame, text='Output')
        label_frame.pack(fill='x')

        self.add_tooltip(
            ttk.Button(label_frame, text='Save', command=self.save),
            f'Save FITS file with backplane data',
        ).pack()

    def build_settings_controls(self) -> None:
        menu = ttk.Frame(self.notebook)
        menu.pack()
        self.notebook.add(menu, text='Settings')
        self.notebook.select(menu)

        frame = ttk.LabelFrame(menu, text='Plot')
        frame.pack(fill='x')

        PlotLineSetting(self, frame, 'limb', label='Limb', hint='the target\'s limb')
        PlotLineSetting(
            self,
            frame,
            'limb_dayside',
            label='Limb (dayside)',
            hint='the illuminated part of the target\'s limb',
        )
        PlotLineSetting(
            self,
            frame,
            'terminator',
            label='Terminator',
            hint='the line between the dayside and nightside regions on the target',
        )
        PlotLineSetting(
            self,
            frame,
            'grid',
            label='Gridlines',
            hint='the longitude/latitude grid on the target',
        )
        PlotLineSetting(
            self,
            frame,
            'rings',
            label='Rings',
            hint='rings around the target (click Format to define ring radii)',
        )
        PlotTextSetting(self, frame, 'poles', label='Poles', hint='the target\'s poles')
        PlotScatterSetting(
            self,
            frame,
            'coordinates_lonlat',
            label='Lon/Lat POI',
            hint='points of interest on the surface of the target (click Format to define POI)',
        )
        PlotScatterSetting(
            self,
            frame,
            'coordinates_radec',
            label='RA/Dec POI',
            hint='points of interest in the sky (click Format to define POI)',
        )
        PlotScatterSetting(
            self,
            frame,
            'other_bodies',
            label='Other bodies',
            hint='other bodies of interest (click Format to specify other bodies to show, e.g. moons)',
        )
        PlotTextSetting(
            self,
            frame,
            'other_bodies_labels',
            label='Other body labels',
            hint='labels for other bodies of interest',
        )

    def build_plot(self) -> None:
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot()

        self.plot_handles['image'] = [
            self.ax.imshow(self.image, origin='lower', zorder=0)
        ]
        self.ax.set_xlim(-0.5, self.observation._nx - 0.5)
        self.ax.set_ylim(-0.5, self.observation._ny - 0.5)
        self.ax.xaxis.set_tick_params(labelsize='x-small')
        self.ax.yaxis.set_tick_params(labelsize='x-small')

        self.plot_wireframe()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
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
        self.observation.update_transform()
        for k, settings in self.plot_settings.items():
            if settings:
                plt.setp(self.plot_handles[k], **settings)
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
        # Formatting shold be done when settings controls are created
        ax = self.ax
        transform = self.observation.get_matplotlib_radec2xy_transform() + ax.transData

        self.plot_handles['limb'].extend(
            ax.plot(*self.observation.limb_radec(), transform=transform, zorder=5)
        )
        self.plot_handles['terminator'].extend(
            ax.plot(*self.observation.terminator_radec(), transform=transform, zorder=5)
        )

        (
            ra_day,
            dec_day,
            ra_night,
            dec_night,
        ) = self.observation.limb_radec_by_illumination()
        self.plot_handles['limb_dayside'].extend(
            ax.plot(ra_day, dec_day, transform=transform, zorder=5)
        )

        for ra, dec in self.observation.visible_latlon_grid_radec(30):
            self.plot_handles['grid'].extend(
                ax.plot(ra, dec, transform=transform, zorder=4)
            )

        for lon, lat, s in self.observation.get_poles_to_plot():
            ra, dec = self.observation.lonlat2radec(lon, lat)
            self.plot_handles['poles'].append(
                ax.add_artist(
                    OutlinedText(
                        ra,
                        dec,
                        s,
                        ha='center',
                        va='center',
                        weight='bold',
                        transform=transform,
                        zorder=5,
                        clip_on=True,
                    )
                )
            )

        for lon, lat in self.observation.coordinates_of_interest_lonlat:
            if self.observation.test_if_lonlat_visible(lon, lat):
                ra, dec = self.observation.lonlat2radec(lon, lat)
                self.plot_handles['coordinates_lonlat'].extend(
                    ax.scatter(
                        ra,
                        dec,
                        marker='x',  # type: ignore
                        color='k',
                        transform=transform,
                    )
                )

        for ra, dec in self.observation.coordinates_of_interest_radec:
            self.plot_handles['coordinates_radec'].extend(
                ax.scatter(
                    ra,
                    dec,
                    marker='+',  # type: ignore
                    color='k',
                    transform=transform,
                )
            )

        for radius in self.observation.ring_radii:
            ra, dec = self.observation.ring_radec(radius)
            self.plot_handles['rings'].extend(
                ax.plot(ra, dec, transform=transform, zorder=5)
            )

        for body in self.observation.other_bodies_of_interest:
            ra = body.target_ra
            dec = body.target_dec

            self.plot_handles['other_bodies_labels'].append(
                ax.text(
                    ra,
                    dec,
                    body.target + '\n',
                    size='small',
                    ha='center',
                    va='center',
                    transform=transform,
                    clip_on=True,
                    zorder=6,
                )
            )
            self.plot_handles['other_bodies'].append(
                ax.scatter(
                    ra,
                    dec,
                    marker='+',  # type: ignore
                    color='w',
                    transform=transform,
                    zorder=7,
                )
            )
        # TODO make this code consistent with elsewhere?
        # TODO make sure everything is plotted
        # TODO tidy up zorder etc.

    # Keybindings
    def bind_keyboard(self) -> None:
        for fn, events in self.shortcuts.items():
            handler = lambda e, f=fn: f()
            for event in events:
                self.root.bind(event, handler)

    # API
    def set_value(
        self, key: SETTER_KEY, value: float, update_plot: bool = True
    ) -> None:
        for fn in self.setter_callbacks.get(key, []):
            fn(value)

        if update_plot:
            self.update_plot()

    def set_step(self, step: float) -> None:
        self.step_size = step
        print(self.step_size)

    # Buttons
    def increase_step(self) -> None:
        self.set_value('step', self.step_size * 10, update_plot=False)

    def decrease_step(self) -> None:
        self.set_value('step', self.step_size / 10, update_plot=False)

    def move_up(self) -> None:
        self.set_value('y0', self.observation.get_y0() + self.step_size)

    def move_up_right(self) -> None:
        self.set_value(
            'y0', self.observation.get_y0() + self.step_size, update_plot=False
        )
        self.set_value('x0', self.observation.get_x0() + self.step_size)

    def move_right(self) -> None:
        self.set_value('x0', self.observation.get_x0() + self.step_size)

    def move_down_right(self) -> None:
        self.set_value(
            'y0', self.observation.get_y0() - self.step_size, update_plot=False
        )
        self.set_value('x0', self.observation.get_x0() + self.step_size)

    def move_down(self) -> None:
        self.set_value('y0', self.observation.get_y0() - self.step_size)

    def move_down_left(self) -> None:
        self.set_value(
            'y0', self.observation.get_y0() - self.step_size, update_plot=False
        )
        self.set_value('x0', self.observation.get_x0() - self.step_size)

    def move_left(self) -> None:
        self.set_value('x0', self.observation.get_x0() - self.step_size)

    def move_up_left(self) -> None:
        self.set_value(
            'y0', self.observation.get_y0() + self.step_size, update_plot=False
        )
        self.set_value('x0', self.observation.get_x0() - self.step_size)

    def rotate_left(self) -> None:
        self.set_value('rotation', self.observation.get_rotation() - self.step_size)

    def rotate_right(self) -> None:
        self.set_value('rotation', self.observation.get_rotation() + self.step_size)

    def increase_radius(self) -> None:
        self.set_value('r0', self.observation.get_r0() + self.step_size)

    def decrease_radius(self) -> None:
        self.set_value('r0', self.observation.get_r0() - self.step_size)

    # File IO
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


class ArtistSetting:
    def __init__(
        self,
        gui: InteractiveObservation,
        parent: tk.Widget,
        key: PLOT_KEY,
        label: str | None = None,
        hint: str | None = None,
        row: int | None = None,
    ):
        self.parent = parent
        self.key: PLOT_KEY = key
        self.gui = gui
        self._enable_callback = True
        if label is None:
            label = key
        self.label = label

        if row is None:
            row = parent.grid_size()[1]

        self.enabled = tk.IntVar()
        self.enabled.set(self.gui.plot_settings[self.key].get('visible', True))
        self.enabled.trace_add('write', self.checkbutton_toggle)
        self.checkbutton = ttk.Checkbutton(
            parent, text=self.label, variable=self.enabled
        )
        self.checkbutton.grid(column=0, row=row, sticky='nsew')

        self.button = ttk.Button(
            parent, text='Format', width=6, command=self.button_click
        )
        self.button.grid(column=1, row=row)

        if hint:
            self.gui.add_tooltip(self.checkbutton, 'Show ' + hint)
            self.gui.add_tooltip(self.button, 'Format ' + hint)

    def checkbutton_toggle(self, *_) -> None:
        enabled = bool(self.enabled.get())
        if enabled:
            self.button['state'] = 'normal'
        else:
            self.button['state'] = 'disable'
        for artist in self.gui.plot_handles[self.key]:
            artist.set_visible(enabled)
        self.gui.update_plot()

    def button_click(self) -> None:
        self.window = tk.Toplevel(self.gui.root)
        self.window.title('Format: ' + self.label)
        self.window.grab_set()
        self.window.transient(self.gui.root)

        x, y = (int(s) for s in self.gui.root.geometry().split('+')[1:])
        self.window.geometry(
            '300x400+{x:.0f}+{y:.0f}'.format(
                x=x + 50,
                y=y + 50,
            )
        )
        self.make_format_menu()

    def make_format_menu(self):
        raise NotImplementedError


class PlotLineSetting(ArtistSetting):
    pass


class PlotScatterSetting(ArtistSetting):
    pass


class PlotTextSetting(ArtistSetting):
    pass


class NumericEntry:
    # TODO add validation
    def __init__(
        self,
        gui: InteractiveObservation,
        parent: tk.Widget,
        key: SETTER_KEY,
        label: str | None = None,
        row: int | None = None,
    ):
        self.parent = parent
        self.key: SETTER_KEY = key
        self.gui = gui
        self._enable_callback = True

        if label is None:
            label = key
        if row is None:
            row = parent.grid_size()[1]
        self.label = ttk.Label(parent, text=label + ' = ')
        self.label.grid(row=row, column=0)

        self.sv = tk.StringVar()
        self.sv.trace_add('write', self.text_input)
        self.entry = ttk.Entry(parent, width=10, textvariable=self.sv)
        self.entry.grid(row=row, column=1)

        self.gui.setter_callbacks[self.key].append(self.update_text)
        self.update_text(self.gui.getters[self.key]())

    def update_text(self, value: float) -> None:
        if not self._enable_callback:
            return
        self._enable_callback = False
        value = self.gui.getters[self.key]()
        self.sv.set(format(value, '.5g'))
        self._enable_callback = True

    def text_input(self, *_) -> None:
        if not self._enable_callback:
            return
        self._enable_callback = False
        value = self.sv.get()
        try:
            self.gui.set_value(self.key, float(value))
            self.entry.configure(foreground='black')
        except ValueError:
            self.entry.configure(foreground='red')
        self._enable_callback = True


class OutlinedText(Text):
    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        *args,
        outline_color: str = 'none',
        **kwargs,
    ):
        super().__init__(x, y, text, *args, **kwargs)
        self.set_outline_color(outline_color)

    def set_outline_color(self, c):
        self.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=c),
                path_effects.Normal(),
            ]  # type: ignore
        )
