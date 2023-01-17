import math
import os
import tkinter as tk
import tkinter.colorchooser
import tkinter.filedialog
import tkinter.messagebox
import tkinter.scrolledtext
from collections import defaultdict
from tkinter import ttk
from typing import Any, Callable, Literal, TypeVar
from functools import lru_cache

import matplotlib.cm
import matplotlib.colors
import matplotlib.markers
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.text import Text
import matplotlib as mpl

from matplotlib.backends._backend_tk import NavigationToolbar2Tk  # TODO delete this

from . import base, data_loader, utils
from .body import Body, NotFoundError
from .observation import Observation
from . import progress


Widget = TypeVar('Widget', bound=tk.Widget)
SETTER_KEY = Literal[
    'x0', 'y0', 'r0', 'rotation', 'step', 'plate_scale_arcsec', 'plate_scale_km'
]
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
    'marked_coord',
    '_',
]
IMAGE_MODE = Literal['sum', 'single', 'rgb']

DEFAULT_PLOT_SETTINGS: dict[PLOT_KEY, dict] = {
    'grid': dict(zorder=3.1, color='#333', linewidth=1, linestyle='dotted'),
    'terminator': dict(zorder=3.2, color='w', linewidth=1, linestyle='dashed'),
    'limb': dict(zorder=3.3, color='w', linewidth=0.5, linestyle='solid'),
    'limb_dayside': dict(zorder=3.31, color='w', linewidth=1, linestyle='solid'),
    'rings': dict(zorder=3.4, color='w', linewidth=0.5, linestyle='solid'),
    'poles': dict(zorder=3.5, color='k', outline_color='w'),
    'coordinates_lonlat': dict(zorder=3.6, marker='x', color='k', s=36),
    'coordinates_radec': dict(zorder=3.7, marker='+', color='k', s=36),
    'other_bodies': dict(zorder=3.8, marker='+', color='w', s=36),
    'other_bodies_labels': dict(zorder=3.81, color='grey'),
    'marked_coord': dict(zorder=4, color='cyan', linewidth=0.5, linestyle='dotted'),
    'image': dict(zorder=0.9, cmap='inferno', vmin=0, vmax=100),
    '_': dict(
        grid_interval=30,
        image_mode='single',
        image_idx_single=0,
        image_idx_r=0,
        image_idx_g=1,
        image_idx_b=2,
        image_gamma=1,
    ),
}


LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot']
MARKERS = ['x', '+', 'o', '.', '*', 'v', '^', '<', '>', ',', 'D', 'd', '|', '_']
GRID_INTERVALS = ['10', '30', '45', '90']
CMAPS = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

MAP_INTERPOLATIONS = ('nearest', 'linear', 'quadratic', 'cubic')

DEFAULT_HINT = (
    'Use the various options in "Find disc" to automatically adjust the disc position'
)


def _main(*args):
    """Called with `planetmapper` from the command line"""
    print('Launching planetmapper...')
    gui = GUI()
    if args:
        gui.set_observation(Observation(args[0]))
    gui.run()


class Quit(Exception):
    pass


class GUI:
    """
    Class to create and run graphical user interface to fit observations.

    This class does not usually need to be run directly, as a GUI can be created
    directly from an :class:`planetmapper.Observation` object using
    :func:`planetmapper.Observation.run_gui`, or by calling `planetmapper` from the
    command line.
    """

    MINIMUM_SIZE = (600, 600)
    DEFAULT_GEOMETRY = '800x650+15+15'

    def __init__(self, allow_open: bool = True) -> None:
        self.allow_open = allow_open

        self._observation: Observation | None = None
        self.step_size = 1

        self.click_locations: list[tuple[float, float]] = []
        """
        List of click locations marked on the plot in `(x, y)` pixel coordinates.

        This list is cleared whenever a new observation is opened.
        """

        self.last_click_location: tuple[float, float] | None = None
        self.coords_formatted_str: str | None = None
        self.coords_machine_str: str | None = None

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
            self.save_button: ['<Control-s>'],
            self.load_observation: ['<Control-o>'],
            self.copy_machine_coord_values: ['<Control-c>'],
        }
        self.shortcuts_to_keep_in_entry = ['<Control-s>', '<Control-o>']

        self.setter_callbacks: defaultdict[
            SETTER_KEY, list[Callable[[float], Any]]
        ] = defaultdict(
            list,
            {
                'x0': [lambda f: self.get_observation().set_x0(f)],
                'y0': [lambda f: self.get_observation().set_y0(f)],
                'r0': [lambda f: self.get_observation().set_r0(f)],
                'rotation': [lambda f: self.get_observation().set_rotation(f)],
                'step': [lambda f: self.set_step(f)],
                'plate_scale_arcsec': [
                    lambda f: self.get_observation().set_plate_scale_arcsec(f)
                ],
                'plate_scale_km': [
                    lambda f: self.get_observation().set_plate_scale_km(f)
                ],
            },
        )
        self.ui_callbacks: defaultdict[
            SETTER_KEY, set[Callable[[], Any]]
        ] = defaultdict(set)

        self.getters: dict[SETTER_KEY, Callable[[], float]] = {
            'x0': lambda: self.get_observation().get_x0(),
            'y0': lambda: self.get_observation().get_y0(),
            'r0': lambda: self.get_observation().get_r0(),
            'rotation': lambda: self.get_observation().get_rotation(),
            'step': lambda: self.step_size,
            'plate_scale_arcsec': lambda: self.get_observation().get_plate_scale_arcsec(),
            'plate_scale_km': lambda: self.get_observation().get_plate_scale_km(),
        }
        self.plot_handles: defaultdict[PLOT_KEY, list[Artist]] = defaultdict(list)
        self.plot_settings: defaultdict[PLOT_KEY, dict] = defaultdict(dict)
        for k, v in DEFAULT_PLOT_SETTINGS.items():
            self.plot_settings[k] = v.copy()

        self.disc_finding_routines: dict[
            str, list[tuple[Callable[[], None], str, str]]
        ] = {
            'Reset position': [
                (
                    lambda: self.get_observation().centre_disc(),
                    'Centre disc in image',
                    'Centre the target\'s planetary disc and make it fill ~90% of the observation',
                ),
            ],
            'Use WCS data from FITS header': [
                (
                    lambda: self.get_observation().disc_from_wcs(True, False),
                    'Use WCS position, rotation & scale',
                    'Set all disc parameters using approximate WCS information in the observation\'s FITS header',
                ),
                (
                    lambda: self.get_observation().position_from_wcs(True, False),
                    'Use WCS position',
                    'Set disc position using approximate WCS information in the observation\'s FITS header',
                ),
                (
                    lambda: self.get_observation().rotation_from_wcs(True, False),
                    'Use WCS rotation',
                    'Set disc rotation using approximate WCS information in the observation\'s FITS header',
                ),
                (
                    lambda: self.get_observation().plate_scale_from_wcs(True, False),
                    'Use WCS plate scale',
                    'Set plate scale using approximate WCS information in the observation\'s FITS header',
                ),
            ],
            'Fit observation': [
                (
                    lambda: self.get_observation().fit_disc_position(),
                    'Fit disc position',
                    'Set x0 and y0 so that the planet\'s disc is fit to the brightest part of the data (this may take a few seconds)',
                ),
                (
                    lambda: self.get_observation().fit_disc_radius(),
                    'Fit disc radius',
                    'Set r0 by calculating the radius around (x0, y0) where the brightness decrease is the fastest (this may take a few seconds)',
                ),
            ],
            'Use FITS header metadata': [
                (
                    lambda: self.get_observation().disc_from_header(),
                    'Use PlanetMapper metadata',
                    'Set disc parameters using information in the observation\'s FITS header generated by any previous runs of PlanetMapper',
                ),
            ],
        }

        self.kernels: list[str] = [
            os.path.join(base.get_kernel_path(), pattern)
            for pattern in base._KERNEL_DATA['kernel_patterns']
        ]

        self.event_time_to_ignore = None
        self.gui_built = False

    def __repr__(self) -> str:
        return f'InteractiveObservation()'

    def run(self) -> None:
        """
        Run the GUI.
        """
        print('Running user interface...')
        try:
            self.get_observation()
        except Quit:
            print('App quit')
            return
        # Disable keyboard shortcuts
        context = {}
        for k in mpl.rcParams:
            if k.startswith('keymap.'):
                context[k] = []
        with mpl.rc_context(context):
            self.build_gui()
            self.bind_keyboard()
            self.root.mainloop()
            # TODO do something when closed to kill figure etc.?

    def load_observation(self) -> None:
        if self.allow_open:
            OpenObservation(gui=self, first_run=self._observation is None)
        if self._observation is None:
            raise Quit

    def set_observation(self, observation: Observation) -> None:
        """
        Set the observation used in the GUI.

        For example, to run the GUI with the data in `'europa.fits'`, use: ::

            gui = planetmapper.gui.GUI()
            gui.set_observation(planetmapper.Observation('europa.fits'))
            gui.run()

        Args:
            observation: Observation to fit.
        """
        self._observation = observation

        self.image_modes: dict[IMAGE_MODE, tuple[Callable[[], np.ndarray], str]] = {
            'single': (self.image_single, 'Single wavelength'),
            'sum': (self.image_sum, 'Sum all wavelengths'),
            'rgb': (self.image_rgb, 'RGB composite'),
        }
        n_wavl = self.get_observation().data.shape[0]
        if n_wavl < 2:
            del self.image_modes['sum']
        if n_wavl < 3:
            del self.image_modes['rgb']

        if n_wavl == 1:
            self.plot_settings['_']['image_mode'] = 'single'
        elif n_wavl == 3 and not self.gui_built:
            self.plot_settings['_']['image_mode'] = 'rgb'
        else:
            self.plot_settings['_']['image_mode'] = 'sum'

        if self.gui_built:
            self.run_all_ui_callbacks()
            self.rebuild_plot()
            self.root.title(self.get_observation().get_description(multiline=False))

        self.click_locations = []
        self.clear_click_location()

    def get_observation(self) -> Observation:
        if self._observation is None:
            self.load_observation()
        assert self._observation is not None
        return self._observation

    # GUI Building
    def build_gui(self) -> None:
        self.root = tk.Tk()
        self.root.geometry(self.DEFAULT_GEOMETRY)
        self.root.minsize(*self.MINIMUM_SIZE)
        self.configure_style(self.root)
        self.root.title(self.get_observation().get_description(multiline=False))

        self.hint_frame = ttk.Frame(self.root)
        self.hint_frame.pack(side='bottom', fill='x')

        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(side='left', fill='y')

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side='top', fill='both', expand=True)

        self.build_plot()
        self.build_help_hint()
        self.build_controls()
        self.update_plot()

        self.gui_built = True

    def configure_style(self, root: tk.Tk | None) -> None:
        if root is None:
            root = self.root
        self.style = ttk.Style(root)
        self.style.theme_use('default')
        # TODO add padding etc. here
        for element in ['TEntry', 'TCombobox', 'TSpinbox', 'TButton', 'TLabel']:
            self.style.configure(
                element,
                foreground='black',
                insertcolor='black',
                fieldbackground='white',
                selectbackground='#bdf',
                selectforeground='black',
            )

    def build_controls(self) -> None:
        self.notebook = ttk.Notebook(self.controls_frame)
        self.notebook.pack(fill='both', expand=True)
        self.build_main_controls_tab()
        self.build_disc_finding_controls_tab()
        self.build_plot_settings_controls_tab()
        self.build_coords_tab()

    def build_main_controls_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Controls')

        buttons = [
            (
                'Open...',
                'Open a new observation, change target/date/observer, and adjust kernel settings',
                self.load_observation,
                0,
                0,
            ),
            (
                'Save...',
                'Save FITS files of the observation and mapped observation with backplane data',
                self.save_button,
                1,
                0,
            ),
        ]
        if not self.allow_open:
            del buttons[0]

        self.build_main_controls_section(
            frame=frame,
            label='File',
            buttons=buttons,
            button_tooltip_base='{hint}',
            entry_tooltip='',
            numeric_entries=[],
        )

        self.build_main_controls_section(
            frame=frame,
            label='Position',
            buttons=[
                ('↖', 'up and left', self.move_up_left, 0, 0),
                ('↑', 'up', self.move_up, 1, 0),
                ('↗', 'up and right', self.move_up_right, 2, 0),
                ('←', 'left', self.move_left, 0, 1),
                ('→', 'right', self.move_right, 2, 1),
                ('↙', 'down and left', self.move_down_left, 0, 2),
                ('↓', 'down', self.move_down, 1, 2),
                ('↘', 'down and right', self.move_down_right, 2, 2),
            ],
            button_tooltip_base='Move fitted disc {hint}',
            entry_tooltip='Set pixel coordinates of the centre of the disc',
            numeric_entries=['x0', 'y0'],
            ipadx=20,
        )

        self.build_main_controls_section(
            frame=frame,
            label='Rotation',
            buttons=[
                ('↺', 'anticlockwise', self.rotate_left, 0, 0),
                ('↻', 'clockwise', self.rotate_right, 1, 0),
            ],
            button_tooltip_base='Rotate fitted disc {hint}',
            entry_tooltip='Set the rotation (in degrees) of the disc',
            numeric_entries=[('rotation', 'Rotation (°)')],
        )

        self.build_main_controls_section(
            frame=frame,
            label='Size',
            buttons=[
                ('-', 'Decrease', self.decrease_radius, 0, 0),
                ('+', 'Increase', self.increase_radius, 1, 0),
            ],
            button_tooltip_base='{hint} fitted disc radius',
            entry_tooltip='Set the equatorial radius, r0, in pixels of the disc',
            numeric_entries=[
                ('r0', 'r0'),
                ('plate_scale_arcsec', 'arcsec/pixel'),
                ('plate_scale_km', 'km/pixel'),
            ],
            add_callbacks=['r0', 'plate_scale_arcsec', 'plate_scale_km'],
        )

        self.build_main_controls_section(
            frame=frame,
            label='Step size',
            buttons=[
                ('÷', 'Decrease', self.decrease_step, 0, 0),
                ('×', 'Increase', self.increase_step, 1, 0),
            ],
            button_tooltip_base='{hint} step size when clicking buttons',
            entry_tooltip='Set the step size when clicking buttons',
            numeric_entries=['step'],
        )

    def build_main_controls_section(
        self,
        frame: ttk.Frame,
        label: str,
        buttons: list[tuple[str, str, Callable[[], None], int, int]],
        button_tooltip_base: str,
        entry_tooltip: str,
        numeric_entries: list[SETTER_KEY | tuple[SETTER_KEY, str]],
        ipadx=30,
        ipady=1,
        add_callbacks: list[SETTER_KEY] | None = None,
        **kw,
    ) -> None:
        label_frame = ttk.LabelFrame(frame, text=label)
        label_frame.pack(fill='x', pady=3, ipadx=1, ipady=1)

        button_frame = ttk.Frame(label_frame)
        button_frame.pack()
        for arrow, hint, fn, column, row in buttons:
            self.add_tooltip(
                ttk.Button(button_frame, text=arrow, command=fn, width=1),
                button_tooltip_base.format(hint=hint),
                fn,
            ).grid(column=column, row=row, ipadx=ipadx, ipady=ipady, padx=2, pady=2)

        entry_frame = self.add_tooltip(ttk.Frame(label_frame), entry_tooltip)
        entry_frame.pack(pady=2)
        for ne in numeric_entries:
            if isinstance(ne, str):
                NumericEntry(
                    self, entry_frame, ne, pady=2, add_callbacks=add_callbacks, **kw
                )
            else:
                NumericEntry(
                    self,
                    entry_frame,
                    ne[0],
                    ne[1],
                    pady=2,
                    add_callbacks=add_callbacks,
                    **kw,
                )

    def build_plot_settings_controls_tab(self) -> None:
        menu = ttk.Frame(self.notebook)
        menu.pack()
        self.notebook.add(menu, text='Settings')
        # self.notebook.select(menu)  # TODO delete this

        # Image
        frame = ttk.LabelFrame(menu, text='Observation')
        frame.pack(fill='x', pady=5)
        frame.grid_columnconfigure(0, weight=1)
        PlotImageSetting(
            self,
            frame,
            'image',
            label='Observed image',
            hint='the image of your observation',
            callbacks=[self.replot_image],
        )

        # Plot features
        frame = ttk.LabelFrame(menu, text='Plotted features')
        frame.pack(fill='x', pady=5)
        frame.grid_columnconfigure(0, weight=1)
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
        PlotGridSetting(
            self,
            frame,
            'grid',
            label='Gridlines',
            hint='the longitude/latitude grid on the target',
            callbacks=[self.replot_grid],
        )
        PlotRingsSetting(
            self,
            frame,
            'rings',
            label='Rings',
            hint='rings around the target (click Edit to define ring radii)',
            callbacks=[self.replot_rings],
        )
        PlotOutlinedTextSetting(
            self,
            frame,
            'poles',
            label='Poles',
            hint='the target\'s poles',
        )
        PlotCoordinatesSetting(
            self,
            frame,
            'coordinates_lonlat',
            label='Lon/Lat POI',
            hint='points of interest on the surface of the target (click Edit to define POI)',
            callbacks=[self.replot_coordinates_lonlat],
            coordinate_list=self.get_observation().coordinates_of_interest_lonlat,
            menu_label='\n'.join(
                [
                    'List of Lon/Lat points of interest.',
                    'Coordinates should be written as comma',
                    'separated "lon, lat" values, with each',
                    'coordinate pair on a new line:',
                ]
            ),
        )
        PlotCoordinatesSetting(
            self,
            frame,
            'coordinates_radec',
            label='RA/Dec POI',
            hint='points of interest in the sky (click Edit to define POI)',
            callbacks=[self.replot_coordinates_radec],
            coordinate_list=self.get_observation().coordinates_of_interest_radec,
            menu_label='\n'.join(
                [
                    'List of RA/Dec points of interest.',
                    'Coordinates should be written as comma',
                    'separated "ra, dec" values, with each',
                    'coordinate pair on a new line:',
                ]
            ),
        )
        PlotOtherBodyScatterSetting(
            self,
            frame,
            'other_bodies',
            label='Other bodies',
            hint='other bodies of interest (click Edit to specify other bodies to show, e.g. moons)',
            callbacks=[self.replot_other_bodies],
        )
        PlotOtherBodyTextSetting(
            self,
            frame,
            'other_bodies_labels',
            label='Other body labels',
            hint='labels for other bodies of interest (click Edit to specify other bodies to show, e.g. moons)',
            callbacks=[self.replot_other_bodies],
        )

        # Marked coords
        frame = ttk.LabelFrame(menu, text='Marked coords')
        frame.pack(fill='x', pady=5)
        frame.grid_columnconfigure(0, weight=1)
        PlotLineSetting(
            self,
            frame,
            'marked_coord',
            label='Click location',
            hint='the location on the image clicked when to calculate coordinates (see the Coords tab)',
            callbacks=[self.replot_marked_coord],
        )

    def build_disc_finding_controls_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Find disc')
        for label, routines in self.disc_finding_routines.items():
            label_frame = ttk.LabelFrame(frame, text=label)
            label_frame.pack(fill='x', pady=10)
            for fn, name, description in routines:
                self.add_tooltip(
                    ttk.Button(
                        label_frame,
                        text=name,
                        command=self.make_disc_finding_fn(fn),
                    ),
                    description,
                ).pack(fill='x', pady=2, padx=5)

    def make_disc_finding_fn(self, fn: Callable[[], None]) -> Callable[[], None]:
        def button_command():
            try:
                fn()
            except ValueError as e:
                tkinter.messagebox.showwarning(
                    title='Error finding disc',
                    message=str(e),
                )
            self.run_all_ui_callbacks(update_plot=True)

        return button_command

    def build_help_hint(self) -> None:
        frame = ttk.Frame(self.hint_frame)
        frame.pack(fill='x', padx=5, pady=1)
        self.help_hint = ttk.Label(frame, text=DEFAULT_HINT)
        self.help_hint.pack(side='left')

    def add_tooltip(
        self, widget: Widget, msg: str, shortcut_fn: Callable | None = None
    ) -> Widget:
        if shortcut_fn is not None:
            keys = self.shortcuts.get(shortcut_fn, None)
            if keys is not None:
                key = keys[0]
                key = key.replace('<less>', '<').upper()
                if key[0] == '<' and key[-1] == '>' and len(key) > 2:
                    key = key[1:-1]
                msg = f'{msg} (keyboard shortcut: {key})'

        def f_enter(event):
            self.help_hint.configure(text=msg)

        def f_leave(event):
            self.help_hint.configure(text='')

        widget.bind('<Enter>', f_enter)
        widget.bind('<Leave>', f_leave)
        return widget

    # Coords
    def build_coords_tab(self):
        top_level_frame = ttk.Frame(self.notebook)
        top_level_frame.pack()
        self.notebook.add(top_level_frame, text='Coords')
        # self.notebook.select(top_level_frame)  # TODO delete this

        frame = ttk.Frame(top_level_frame)
        frame.pack(padx=5, fill='x')
        self.coords_tab_labels: dict[str, ttk.Label] = {}
        self.coords_labels: dict[str, list[str | tuple[str, str]]] = {
            'Pixel coordinates': ['x', 'y'],
            'Celestial coordinates': [
                ('ra', 'Right ascension'),
                ('dec', 'Declination'),
            ],
            'Planetographic coordinates': [('lon', 'Longitude'), ('lat', 'Latitude')],
            'Illumination angles': ['phase', 'incidence', 'emission', 'azimuth'],
        }
        for name, part_labels in self.coords_labels.items():
            label_frame = ttk.LabelFrame(frame, text=name)
            label_frame.pack(fill='x', pady=5)
            for col in range(2):
                label_frame.grid_columnconfigure(col, weight=1, uniform='a')
            for row, kl in enumerate(part_labels):
                if isinstance(kl, tuple):
                    key, label = kl
                else:
                    key = kl
                    label = kl.capitalize()
                label = label + ':'
                l1 = ttk.Label(label_frame, text=label)
                l1.grid(row=row, column=0, sticky='e', pady=2, padx=2)

                l2 = ttk.Label(label_frame, text='')
                l2.grid(row=row, column=1, sticky='w', pady=2, padx=2)

                self.coords_tab_labels[key] = l2

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=5)

        self.coords_copy_formatted_button = self.add_tooltip(
            ttk.Button(
                button_frame,
                text='Copy formatted values',
                command=self.copy_formatted_coord_values,
                state='disable',
            ),
            'Copy formatted coordinate values',
            self.copy_formatted_coord_values,
        )
        self.coords_copy_formatted_button.pack(fill='x', pady=2, padx=2)
        self.coords_copy_machine_button = self.add_tooltip(
            ttk.Button(
                button_frame,
                text='Copy machine readable values',
                command=self.copy_machine_coord_values,
                state='disable',
            ),
            'Copy machine readable coordinate values, compatible with Python, JSON, etc.',
            self.copy_machine_coord_values,
        )
        self.coords_copy_machine_button.pack(fill='x', pady=2, padx=2)

        message = [
            '',
            'Click on the plot to get coordinates',
            'Right click on the plot to clear',
            '',
            'Note that most of these values change',
            'when you adjust the disc position',
            '',
        ]
        ttk.Label(top_level_frame, text='\n'.join(message), justify='center').pack(
            fill='x', padx=5
        )

    def get_click_coords(self) -> dict[str, float]:
        if self.last_click_location is None:
            return {}
        out: dict[str, float] = {}
        observation = self.get_observation()
        x, y = self.last_click_location
        out['x'] = x
        out['y'] = y
        out['ra'], out['dec'] = observation.xy2radec(x, y)
        try:
            targvec = observation._xy2targvec(x, y)
            out['lon'], out['lat'] = observation.targvec2lonlat(targvec)
            (
                phase,
                incdnc,
                emissn,
            ) = observation._illumination_angles_from_targvec_radians(targvec)
            az = observation._azimuth_angle_from_gie_radians(phase, incdnc, emissn)
            (
                out['phase'],
                out['incidence'],
                out['emission'],
                out['azimuth'],
            ) = np.rad2deg((phase, incdnc, emissn, az))
        except NotFoundError:
            pass
        return out

    def update_coords(self, print_coords: bool = False) -> None:
        if self.last_click_location is None:
            for k, label in self.coords_tab_labels.items():
                label.configure(text='')
            return

        coords = self.get_click_coords()
        coords_strs = self.get_click_coords_formatted_strings(coords)
        if print_coords:
            # Print with trailing comma so can be copied straight into a list
            print(self.make_click_json_string(coords) + ',')

        self.coords_machine_str = self.make_click_json_string(
            coords, fmt='', fmt_radec=''
        )
        self.coords_formatted_str = self.make_click_formatted_string(coords_strs)

        for k, label in self.coords_tab_labels.items():
            label.configure(text=coords_strs.get(k, ''))

    def get_click_coords_formatted_strings(
        self, coords: dict[str, float], fmt: str = '.2f', dms_fmt: str = '.3f'
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        x, y = coords['x'], coords['y']
        observation = self.get_observation()

        out['x'] = f'{x:{fmt}}'
        out['y'] = f'{y:{fmt}}'

        ra, dec = coords['ra'], coords['dec']
        out['ra'] = utils.decimal_degrees_to_dms_str(ra, dms_fmt)
        out['dec'] = utils.decimal_degrees_to_dms_str(dec, dms_fmt)

        try:
            # Use targvec for a bit more speed here
            lon, lat = coords['lon'], coords['lat']
            ew = observation.positive_longitude_direction
            ns = 'N' if lat >= 0 else 'S'
            out['lon'] = f'{lon:{fmt}}°{ew}'
            out['lat'] = f'{abs(lat):{fmt}}°{ns}'

            out['phase'] = f'{coords["phase"]:{fmt}}°'
            out['incidence'] = f'{coords["incidence"]:{fmt}}°'
            out['emission'] = f'{coords["emission"]:{fmt}}°'
            out['azimuth'] = f'{coords["azimuth"]:{fmt}}°'
        except KeyError:
            pass
        return out

    def make_click_formatted_string(self, coords_strs: dict[str, str]) -> str:
        msg = []
        for name, part_labels in self.coords_labels.items():
            msg.append(name)
            for row, kl in enumerate(part_labels):
                if isinstance(kl, tuple):
                    key, label = kl
                else:
                    key = kl
                    label = kl.capitalize()
                value = coords_strs.get(key, '')
                msg.append(f'  - {label}: {value}')
        return '\n'.join(msg)

    def make_click_json_string(
        self, coords: dict[str, float], fmt: str = '.2f', fmt_radec: str = '.6f'
    ) -> str:
        x, y = coords['x'], coords['y']
        ra, dec = coords['ra'], coords['dec']
        parts = [
            f'"xy": [{x:{fmt}}, {y:{fmt}}]',
            f'"radec": [{ra:{fmt_radec}}, {dec:{fmt_radec}}]',
        ]

        try:
            # Use targvec for a bit more speed here
            lon, lat = coords['lon'], coords['lat']
            parts.extend(
                [
                    f'"lonlat": [{lon:{fmt}}, {lat:{fmt}}]',
                    f'"phase": {coords["phase"]:{fmt}}',
                    f'"incidence": {coords["incidence"]:{fmt}}',
                    f'"emission": {coords["emission"]:{fmt}}',
                    f'"azimuth": {coords["azimuth"]:{fmt}}',
                ]
            )
        except KeyError:
            pass  # Not on disc
        return '{' + ', '.join(parts) + '}'

    def figure_click_callback(self, event: MouseEvent) -> None:
        if not event.inaxes or event.dblclick:
            return

        try:
            # Disable when panning/zooming
            if self.toolbar.mode._navigate_mode is not None:
                return
        except:
            pass

        if event.button == MouseButton.RIGHT:
            self.clear_click_location()

        if event.button == MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            self.set_click_location(x, y)
        self.replot_marked_coord()
        self.update_plot(print_coords=True)

    def set_click_location(self, x: float, y: float) -> None:
        self.click_locations.append((x, y))
        self.last_click_location = (x, y)
        for button in [
            self.coords_copy_formatted_button,
            self.coords_copy_machine_button,
        ]:
            button['state'] = 'normal'

    def clear_click_location(self) -> None:
        self.last_click_location = None
        self.coords_formatted_str = None
        self.coords_machine_str = None
        try:
            for button in [
                self.coords_copy_formatted_button,
                self.coords_copy_machine_button,
            ]:
                button['state'] = 'disable'
        except AttributeError:
            pass

    def copy_formatted_coord_values(self) -> None:
        self.copy_to_clipboard(self.coords_formatted_str)

    def copy_machine_coord_values(self) -> None:
        self.copy_to_clipboard(self.coords_machine_str)

    def copy_to_clipboard(self, s: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(s)

    # Plotting
    def update_plot(self, print_coords: bool = False) -> None:
        self.get_observation().update_transform()
        self.canvas.draw()
        self.update_coords(print_coords=print_coords)

    def build_plot(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.06, 0.03, 0.93, 0.96])
        self.transform = (
            self.get_observation().matplotlib_radec2xy_transform() + self.ax.transData
        )

        self.replot_all()
        self.format_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        toolbar_frame = tk.Frame(self.plot_frame)
        toolbar_frame.pack(side='bottom', fill='x')
        tk.Label(toolbar_frame, text='\N{NO-BREAK SPACE}').pack(side='left')
        self.toolbar = CustomNavigationToolbar(
            self.canvas,
            toolbar_frame,
            pack_toolbar=False,
            gui=self,
        )
        self.toolbar.pack(side='bottom', fill='x')

        self.fig.canvas.callbacks.connect(
            'button_press_event', self.figure_click_callback
        )

    def rebuild_plot(self) -> None:
        self.transform = (
            self.get_observation().matplotlib_radec2xy_transform() + self.ax.transData
        )
        self.replot_all()
        self.format_plot()
        self.update_plot()

    def replot_all(self) -> None:
        self.replot_image()
        self.replot_grid()
        self.replot_terminator()
        self.replot_limb()
        self.replot_rings()
        self.replot_poles()
        self.replot_coordinates_lonlat()
        self.replot_coordinates_radec()
        self.replot_other_bodies()

    def format_plot(self):
        self.fig.set_dpi(100)
        self.ax.set_xlim(-0.5, self.get_observation()._nx - 0.5)
        self.ax.set_ylim(-0.5, self.get_observation()._ny - 0.5)
        self.ax.xaxis.set_tick_params(labelsize='x-small')
        self.ax.yaxis.set_tick_params(labelsize='x-small')
        self.ax.set_facecolor('0.1')
        self.ax.set_axisbelow(True)

    def replot_image(self):
        self.remove_artists('image')

        mode = self.plot_settings['_'].setdefault('image_mode', 'single')
        image = self.image_modes[mode][0]()  # type: ignore

        self.plot_handles['image'].append(
            self.ax.imshow(
                image,
                origin='lower',
                **self.plot_settings['image'],
            )
        )

    def replot_limb(self):
        self.remove_artists('limb')
        self.remove_artists('limb_dayside')
        self.plot_handles['limb'].extend(
            self.ax.plot(
                *self.get_observation().limb_radec(),
                transform=self.transform,
                **self.plot_settings['limb'],
            )
        )
        (
            ra_day,
            dec_day,
            ra_night,
            dec_night,
        ) = self.get_observation().limb_radec_by_illumination()
        self.plot_handles['limb_dayside'].extend(
            self.ax.plot(
                ra_day,
                dec_day,
                transform=self.transform,
                **self.plot_settings['limb_dayside'],
            )
        )

    def replot_terminator(self):
        self.remove_artists('terminator')
        self.plot_handles['terminator'].extend(
            self.ax.plot(
                *self.get_observation().terminator_radec(),
                transform=self.transform,
                **self.plot_settings['terminator'],
            )
        )

    def replot_poles(self):
        self.remove_artists('poles')
        for lon, lat, s in self.get_observation().get_poles_to_plot():
            ra, dec = self.get_observation().lonlat2radec(lon, lat)
            self.plot_handles['poles'].append(
                self.ax.add_artist(
                    OutlinedText(
                        ra,
                        dec,
                        s,
                        ha='center',
                        va='center',
                        weight='bold',
                        size='small',
                        transform=self.transform,
                        clip_on=True,
                        **self.plot_settings['poles'],
                    )
                )
            )

    def replot_grid(self) -> None:
        self.remove_artists('grid')
        interval = self.plot_settings['_'].setdefault('grid_interval', 30)
        for ra, dec in self.get_observation().visible_lonlat_grid_radec(interval):
            self.plot_handles['grid'].extend(
                self.ax.plot(
                    ra,
                    dec,
                    transform=self.transform,
                    **self.plot_settings['grid'],
                )
            )

    def replot_coordinates_lonlat(self) -> None:
        self.remove_artists('coordinates_lonlat')
        for lon, lat in self.get_observation().coordinates_of_interest_lonlat:
            if self.get_observation().test_if_lonlat_visible(lon, lat):
                ra, dec = self.get_observation().lonlat2radec(lon, lat)
                self.plot_handles['coordinates_lonlat'].append(
                    self.ax.scatter(
                        ra,
                        dec,
                        transform=self.transform,
                        **self.plot_settings['coordinates_lonlat'],
                    )
                )

    def replot_coordinates_radec(self) -> None:
        self.remove_artists('coordinates_radec')
        for ra, dec in self.get_observation().coordinates_of_interest_radec:
            self.plot_handles['coordinates_radec'].append(
                self.ax.scatter(
                    ra,
                    dec,
                    transform=self.transform,
                    **self.plot_settings['coordinates_radec'],
                )
            )

    def replot_rings(self) -> None:
        self.remove_artists('rings')
        for radius in self.get_observation().ring_radii:
            ra, dec = self.get_observation().ring_radec(radius)
            self.plot_handles['rings'].extend(
                self.ax.plot(
                    ra,
                    dec,
                    transform=self.transform,
                    **self.plot_settings['rings'],
                )
            )

    def replot_other_bodies(self) -> None:
        self.remove_artists('other_bodies_labels')
        self.remove_artists('other_bodies')
        for body in self.get_observation().other_bodies_of_interest:
            ra = body.target_ra
            dec = body.target_dec

            self.plot_handles['other_bodies_labels'].append(
                self.ax.text(
                    ra,
                    dec,
                    body.target + '\n',
                    size='small',
                    ha='center',
                    va='center',
                    transform=self.transform,
                    clip_on=True,
                    **self.plot_settings['other_bodies_labels'],
                )
            )
            self.plot_handles['other_bodies'].append(
                self.ax.scatter(
                    ra,
                    dec,
                    transform=self.transform,
                    **self.plot_settings['other_bodies'],
                )
            )

    def replot_marked_coord(self) -> None:
        self.remove_artists('marked_coord')
        if self.last_click_location is not None:
            x, y = self.last_click_location
            self.plot_handles['marked_coord'].append(
                self.ax.axvline(x, **self.plot_settings['marked_coord'])
            )
            self.plot_handles['marked_coord'].append(
                self.ax.axhline(y, **self.plot_settings['marked_coord'])
            )

    def remove_artists(self, key: PLOT_KEY) -> None:
        while self.plot_handles[key]:
            self.plot_handles[key].pop().remove()

    # Image
    def image_sum(self) -> np.ndarray:
        return 100 * utils.normalise(
            np.nansum(self.get_observation().data, axis=0)
        ) ** self.plot_settings['_'].setdefault('image_gamma', 1)

    def image_single(self) -> np.ndarray:
        return 100 * utils.normalise(
            self.get_observation().data[
                self.plot_settings['_'].setdefault('image_idx_single', 0)
            ]
        ) ** self.plot_settings['_'].setdefault('image_gamma', 1)

    def image_rgb(self) -> np.ndarray:
        r = self.get_observation().data[
            self.plot_settings['_'].setdefault('image_idx_r', 0)
        ]
        g = self.get_observation().data[
            self.plot_settings['_'].setdefault('image_idx_g', 0)
        ]
        b = self.get_observation().data[
            self.plot_settings['_'].setdefault('image_idx_b', 0)
        ]
        return utils.normalise(np.stack((r, g, b), axis=2)) ** self.plot_settings[
            '_'
        ].setdefault('image_gamma', 1)

    # Keybindings
    def bind_keyboard(self) -> None:
        for fn, events in self.shortcuts.items():
            handler = lambda e, f=fn: self.process_keypress(e, f)
            for event in events:
                self.root.bind(event, handler)

    def process_keypress(self, event, fn) -> None:
        if event.time != self.event_time_to_ignore:
            fn()

    def ignore_keypress(self, event) -> None:
        self.event_time_to_ignore = event.time

    # API
    def run_all_ui_callbacks(self, update_plot: bool = True):
        all_callbacks: set[Callable[[], None]] = set()
        # Use a set so we don't call same callback multiple times
        for k, callbacks in self.ui_callbacks.items():
            all_callbacks.update(callbacks)
        for fn in all_callbacks:
            fn()
        if update_plot:
            self.update_plot()

    def set_value(
        self, key: SETTER_KEY, value: float, update_plot: bool = True
    ) -> None:
        for fn in self.setter_callbacks[key]:
            fn(value)
        for fn in self.ui_callbacks[key]:
            fn()
        if update_plot:
            self.update_plot()

    def set_step(self, step: float) -> None:
        if not step > 0:
            raise ValueError('step must be greater than zero')
        self.step_size = step

    # Buttons
    def increase_step(self) -> None:
        self.set_value('step', self.step_size * 10, update_plot=False)

    def decrease_step(self) -> None:
        self.set_value('step', self.step_size / 10, update_plot=False)

    def move_up(self) -> None:
        self.set_value('y0', self.get_observation().get_y0() + self.step_size)

    def move_up_right(self) -> None:
        self.set_value(
            'y0', self.get_observation().get_y0() + self.step_size, update_plot=False
        )
        self.set_value('x0', self.get_observation().get_x0() + self.step_size)

    def move_right(self) -> None:
        self.set_value('x0', self.get_observation().get_x0() + self.step_size)

    def move_down_right(self) -> None:
        self.set_value(
            'y0', self.get_observation().get_y0() - self.step_size, update_plot=False
        )
        self.set_value('x0', self.get_observation().get_x0() + self.step_size)

    def move_down(self) -> None:
        self.set_value('y0', self.get_observation().get_y0() - self.step_size)

    def move_down_left(self) -> None:
        self.set_value(
            'y0', self.get_observation().get_y0() - self.step_size, update_plot=False
        )
        self.set_value('x0', self.get_observation().get_x0() - self.step_size)

    def move_left(self) -> None:
        self.set_value('x0', self.get_observation().get_x0() - self.step_size)

    def move_up_left(self) -> None:
        self.set_value(
            'y0', self.get_observation().get_y0() + self.step_size, update_plot=False
        )
        self.set_value('x0', self.get_observation().get_x0() - self.step_size)

    def rotate_left(self) -> None:
        self.set_value(
            'rotation', self.get_observation().get_rotation() - self.step_size
        )

    def rotate_right(self) -> None:
        self.set_value(
            'rotation', self.get_observation().get_rotation() + self.step_size
        )

    def increase_radius(self) -> None:
        self.set_value('r0', self.get_observation().get_r0() + self.step_size)

    def decrease_radius(self) -> None:
        try:
            self.set_value('r0', self.get_observation().get_r0() - self.step_size)
        except ValueError:
            # hide value error message when trying to go r0<0
            pass

    # File IO
    def save_button(self) -> None:
        SaveObservation(self)


class Popup:
    def get_int(
        self,
        string_variable: tk.StringVar | str,
        name: str,
        positive: bool = True,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        if isinstance(string_variable, tk.StringVar):
            s = string_variable.get()
        else:
            s = string_variable
        try:
            value = int(s)
        except ValueError:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'Could not convert {name} {s!r} to int',
            )
            raise

        if not np.isfinite(value):
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must be finite',
            )
            raise ValueError

        if positive and not value > 0:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must be greater than zero',
            )
            raise ValueError

        if minimum is not None and value < minimum:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must not be less than {minimum}',
            )
            raise ValueError

        if maximum is not None and value > maximum:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must not be greater than {maximum}',
            )
            raise ValueError

        return value

    def get_float(
        self,
        string_variable: tk.StringVar | str,
        name: str,
        positive: bool = True,
        finite: bool = True,
    ) -> float:
        if isinstance(string_variable, tk.StringVar):
            s = string_variable.get()
        else:
            s = string_variable
        try:
            value = float(s)
        except ValueError:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'Could not convert {name} {s!r} to float',
            )
            raise

        if finite and not math.isfinite(value):
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must be finite',
            )
            raise ValueError

        if positive and not value > 0:
            tkinter.messagebox.showwarning(
                title=f'Error parsing {name}',
                message=f'{name.capitalize()} must be greater than zero',
            )
            raise ValueError

        return value


# File IO popups
class OpenObservation(Popup):
    def __init__(self, gui: GUI, first_run: bool) -> None:
        self.gui = gui
        self.first_run = first_run
        try:
            self.gui.root
        except AttributeError:
            self.first_run = True
        self.make_widget()
        self.make_menu()

        if self.first_run:
            self.window.mainloop()

    def make_widget(self) -> None:
        if self.first_run:
            self.window = tk.Tk()
            self.window.title('PlanetMapper')
            self.gui.configure_style(self.window)
            geometry = self.gui.DEFAULT_GEOMETRY
        else:
            self.window = tk.Toplevel(self.gui.root)
            self.window.title('Observation settings')
            self.window.grab_set()
            self.window.transient(self.gui.root)
            geometry = self.gui.root.geometry()

        x, y = (int(s) for s in geometry.split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(
                sz='600x400',
                x=x + 50,
                y=y + 50,
            )
        )

        frame = ttk.Frame(self.window)
        frame.pack(side='bottom', fill='x')
        frame = ttk.Frame(frame)
        frame.pack(padx=10, pady=10)
        for idx, (text, command) in enumerate(
            [
                ('OK', self.click_ok),
                ('Cancel', self.click_cancel),
                ('Apply', self.click_apply),
            ]
        ):
            ttk.Button(frame, text=text, width=7, command=command).grid(
                row=0,
                column=idx,
                padx=2,
            )
        if not self.first_run:
            self.window.bind('<Escape>', self.close_window)

        window_frame = ttk.Frame(self.window)
        window_frame.pack(expand=True, fill='both')

        self.menu_frame = ttk.Frame(window_frame)
        self.menu_frame.pack(side='top', padx=10, pady=10, fill='x')

        self.heading_frame = ttk.Frame(self.menu_frame)
        self.heading_frame.pack(fill='x')

        self.grid_frame = ttk.Frame(self.menu_frame)
        self.grid_frame.pack(fill='x')

        self.window.protocol('WM_DELETE_WINDOW', self.close_window)

    def make_menu(self):
        kwargs = {}
        self.stringvars: dict[str, tk.StringVar] = {}
        observation = self.gui._observation
        if observation is not None:
            kwargs['path'] = observation.path
            kwargs['target'] = observation.target
            kwargs['utc'] = observation.utc
            kwargs['observer'] = observation.observer

        self.stringvars['path'] = tk.StringVar(value=str(kwargs.get('path', '')))
        self.stringvars['target'] = tk.StringVar(value=str(kwargs.get('target', '')))
        self.stringvars['utc'] = tk.StringVar(value=str(kwargs.get('utc', '')))
        self.stringvars['observer'] = tk.StringVar(
            value=str(kwargs.get('observer', 'EARTH'))
        )

        self.stringvars['path'].trace_add('write', self.path_changed)


        heading = '\n'.join(
            ['Select a FITS or image (e.g. PNG, JPEG) file to navigate and map']
        )

        ttk.Label(self.heading_frame, text=heading + '\n').pack()

        self.grid: list[tuple[tk.Widget, ...]] = [
            (
                ttk.Label(self.grid_frame, text='Path: '),
                ttk.Entry(
                    self.grid_frame,
                    textvariable=self.stringvars['path'],
                    # state='disabled',
                ),
                ttk.Button(self.grid_frame, text='Open', command=self.get_path),
            ),
            (
                ttk.Label(self.grid_frame, text='Target: '),
                ttk.Entry(self.grid_frame, textvariable=self.stringvars['target']),
            ),
            (
                ttk.Label(self.grid_frame, text='Date (UTC): '),
                ttk.Entry(self.grid_frame, textvariable=self.stringvars['utc']),
            ),
            (
                ttk.Label(self.grid_frame, text='Observer: '),
                ttk.Entry(self.grid_frame, textvariable=self.stringvars['observer']),
            ),
        ]
        self.add_to_menu_grid(self.grid)
        self.grid_frame.grid_columnconfigure(1, weight=1)

        value = '\n'.join(self.gui.kernels)
        label = '\n'.join(
            [
                'List of paths of SPICE kernels to load, with each path listed on a new line.',
                'User "~" and glob (e.g. "*", "**") patterns will be automatically expanded:',
            ]
        )
        ttk.Label(self.menu_frame, text='\n' + label).pack(fill='x')
        self.kernel_txt = tkinter.scrolledtext.ScrolledText(self.menu_frame)
        self.kernel_txt.pack(fill='both')
        self.kernel_txt.insert('1.0', value)

    def path_changed(self, *_) -> None:
        path = self.stringvars['path'].get()
        kwargs = {}
        if any(path.endswith(ext) for ext in Observation.FITS_FILE_EXTENSIONS):
            try:
                with fits.open(path) as hdul:
                    header = hdul[0].header  # type: ignore
                Observation._add_kw_from_header(kwargs, header)
            except FileNotFoundError:
                pass
        for k, v in kwargs.items():
            try:
                if v:
                    self.stringvars[k].set(str(v))
            except KeyError:
                pass

    def get_path(self):
        path = tkinter.filedialog.askopenfilename(
            title='Choose observation',
            parent=self.window,
        )
        if path:
            self.stringvars['path'].set(str(path))

    def click_ok(self) -> None:
        if self.apply_changes():
            self.close_window()

    def click_apply(self) -> None:
        self.apply_changes()

    def apply_changes(self) -> bool:
        kwargs = {k: v.get() for k, v in self.stringvars.items()}
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v.strip()) == 0:
                tkinter.messagebox.showwarning(
                    title=f'Error parsing {k}', message=f'{k!r} must not be empty'
                )
                return False

        string = self.kernel_txt.get('1.0', 'end')
        kernels = [k.strip() for k in string.splitlines()]
        base.load_kernels(*kernels, clear_before=True)

        kernel_help = 'Check for typos and make sure you have listed all the required SPICE kernels'

        sb = base.SpiceBase(load_kernels=False)
        try:
            target = sb.standardise_body_name(kwargs['target'])
        except:
            tkinter.messagebox.showwarning(
                title=f'Error parsing target',
                message='Target name {!r} not recognised\n{}'.format(
                    kwargs['target'], kernel_help
                ),
            )
            return False

        try:
            observer = sb.standardise_body_name(kwargs['observer'])
        except:
            tkinter.messagebox.showwarning(
                title=f'Error parsing observer',
                message='Observer name {!r} not recognised\n{}'.format(
                    kwargs['observer'], kernel_help
                ),
            )
            return False

        if target == observer:
            tkinter.messagebox.showwarning(
                title=f'Target and observer identical',
                message='Target and observer must correspond to different bodies',
            )
            return False

        try:
            kwargs['utc'] = float(kwargs['utc'])  #  type: ignore
        except ValueError:
            try:
                spice.utc2et(kwargs['utc'])
            except:
                tkinter.messagebox.showwarning(
                    title=f'Error parsing utc',
                    message='Could not parse {!r}\n{}'.format(
                        kwargs['utc'], kernel_help
                    ),
                )
                return False
        try:
            observation = Observation(**kwargs, load_kernels=False)
        except Exception as e:
            tkinter.messagebox.showwarning(
                title=f'Error processing inputs',
                message=f'Error: {e}',
            )
            return False
        self.gui.set_observation(observation)
        self.gui.kernels = kernels
        return True

    def click_cancel(self) -> None:
        self.close_window()

    def close_window(self, *_) -> None:
        self.window.destroy()
        base.load_kernels(*self.gui.kernels, clear_before=True)

    def add_to_menu_grid(
        self, grid: list[tuple[tk.Widget, ...]], frame: ttk.Frame | None = None
    ) -> None:
        if frame is None:
            frame = self.grid_frame
        ncols = max(len(row) for row in grid)
        for grid_row in grid:
            row = frame.grid_size()[1]
            label = grid_row[0]
            widgets = grid_row[1:]
            label.grid(row=row, column=0, sticky='w', pady=5)
            for idx, widget in enumerate(widgets):
                if idx == len(widgets) - 1:
                    colspan = ncols - len(widgets)
                else:
                    colspan = 1
                widget.grid(row=row, column=1 + idx, sticky='ew', columnspan=colspan)


class SaveObservation(Popup):
    def __init__(self, gui: GUI) -> None:
        self.gui = gui

        self.make_widget()
        self.make_menu()

    def make_widget(self) -> None:
        self.window = tk.Toplevel(self.gui.root)
        self.window.title('Save observation')
        self.window.grab_set()
        self.window.transient(self.gui.root)

        x, y = (int(s) for s in self.gui.root.geometry().split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(
                sz='600x375',
                x=x + 50,
                y=y + 50,
            )
        )

        frame = ttk.Frame(self.window)
        frame.pack(side='bottom', fill='x')
        frame = ttk.Frame(frame)
        frame.pack(padx=10, pady=10)

        self.save_button = ttk.Button(
            frame, text='Save', width=10, command=self.click_save
        )
        self.save_button.grid(row=0, column=0, padx=2)
        ttk.Button(frame, text='Cancel', width=10, command=self.click_cancel).grid(
            row=0, column=1, padx=2
        )

        self.window.bind('<Escape>', self.close_window)

        window_frame = ttk.Frame(self.window)
        window_frame.pack(expand=True, fill='both')

        self.menu_frame = ttk.Frame(window_frame)
        self.menu_frame.pack(side='top', padx=10, pady=10, fill='x')

        self.heading_frame = ttk.Frame(self.menu_frame)
        self.heading_frame.pack(fill='x')

        self.grid_frame = ttk.Frame(self.menu_frame)
        self.grid_frame.pack(fill='x')

        self.saving_progress_window: SavingProgress | None = None

    def add_to_menu_grid(
        self, grid: list[tuple[tk.Widget, ...]], frame: ttk.Frame | None = None
    ) -> None:
        if frame is None:
            frame = self.grid_frame
        ncols = max(len(row) for row in grid)
        for grid_row in grid:
            row = frame.grid_size()[1]
            label = grid_row[0]
            widgets = grid_row[1:]
            label.grid(row=row, column=0, sticky='w', pady=5)
            for idx, widget in enumerate(widgets):
                if idx == len(widgets) - 1:
                    colspan = ncols - len(widgets)
                else:
                    colspan = 1
                widget.grid(row=row, column=1 + idx, sticky='ew', columnspan=colspan)

    def make_menu(self):
        path = self.gui.get_observation().path
        if path is not None:
            root, _ = os.path.splitext(path)
            if root.endswith('.fits'):
                # deal with e.g. .fits.gz files
                root, _ = os.path.splitext(root)
            path_nav = os.path.abspath(root + '_nav.fits')
            path_map = os.path.abspath(root + '_map.fits')
        else:
            path_nav = os.path.abspath(
                self.gui.get_observation().make_filename(suffix='_nav')
            )
            path_map = os.path.abspath(
                self.gui.get_observation().make_filename(suffix='_map')
            )

        self.save_nav = tk.IntVar(value=1)
        self.save_map = tk.IntVar(value=1)
        self.path_nav = tk.StringVar(value=path_nav)
        self.path_map = tk.StringVar(value=path_map)
        self.degree_interval = tk.StringVar(value=str(1))
        self.map_interpolation = tk.StringVar(value='linear')

        self.keep_open = tk.IntVar(value=1)

        self.save_nav.trace_add('write', self.save_nav_toggle)
        self.save_map.trace_add('write', self.save_map_toggle)

        self.nav_widgets: list[tk.Widget] = []
        self.map_widgets: list[tk.Widget] = []

        self.grid_frame.grid_columnconfigure(1, weight=1)
        label_kw = dict(column=0, sticky='w', pady=5)

        # Navigated
        ttk.Checkbutton(
            self.grid_frame, text='Save navigated observation', variable=self.save_nav
        ).grid(row=0, column=1, columnspan=2, sticky='ew')

        ttk.Label(self.grid_frame, text='Path: ').grid(row=1, **label_kw)
        w = ttk.Entry(self.grid_frame, textvariable=self.path_nav)
        w.grid(row=1, column=1, sticky='ew')
        self.nav_widgets.append(w)
        w = ttk.Button(self.grid_frame, text='...', width=3, command=self.get_path_nav)
        w.grid(row=1, column=2)
        self.nav_widgets.append(w)

        ttk.Label(self.grid_frame, text=' ').grid(row=2, **label_kw)

        # Mapped
        ttk.Checkbutton(
            self.grid_frame, text='Save mapped observation', variable=self.save_map
        ).grid(row=3, column=1, columnspan=2, sticky='ew')

        ttk.Label(self.grid_frame, text='Path: ').grid(row=4, **label_kw)
        w = ttk.Entry(self.grid_frame, textvariable=self.path_map)
        w.grid(row=4, column=1, sticky='ew')
        self.map_widgets.append(w)
        w = ttk.Button(self.grid_frame, text='...', width=3, command=self.get_path_map)
        w.grid(row=4, column=2, sticky='w')
        self.map_widgets.append(w)

        ttk.Label(self.grid_frame, text='Degree interval: ').grid(row=5, **label_kw)
        w = ttk.Entry(self.grid_frame, textvariable=self.degree_interval, width=10)
        w.grid(row=5, column=1, sticky='w')
        self.map_widgets.append(w)

        ttk.Label(self.grid_frame, text='Interpolation: ').grid(row=6, **label_kw)
        w = ttk.Combobox(
            self.grid_frame,
            textvariable=self.map_interpolation,
            width=10,
            values=MAP_INTERPOLATIONS,
            state='readonly',
        )
        w.grid(row=6, column=1, sticky='w')
        self.map_widgets.append(w)

        message = '\n'.join(
            [
                '',
                'Click SAVE below to save the requested files',
                '',
                'For larger files, backplane generation, mapping, and saving can take ~1 minute',
                '',
            ]
        )
        ttk.Label(self.menu_frame, text='\n' + message, justify='center').pack()

        ttk.Checkbutton(
            self.menu_frame,
            text='Keep popup open after saving files',
            variable=self.keep_open,
        ).pack()

    def get_path_nav(self) -> None:
        self.get_path(self.path_nav)

    def get_path_map(self) -> None:
        self.get_path(self.path_map)

    def get_path(self, stringvar: tk.StringVar) -> None:
        path = tkinter.filedialog.asksaveasfilename(
            parent=self.window,
            confirmoverwrite=True,
            initialfile=stringvar.get(),
        )
        if len(path.strip()) > 0:
            stringvar.set(path)

    def save_nav_toggle(self, *_) -> None:
        self.toggle(self.save_nav, self.nav_widgets)

    def save_map_toggle(self, *_) -> None:
        self.toggle(self.save_map, self.map_widgets)

    def toggle(self, intvar: tk.IntVar, widgets: list[tk.Widget]) -> None:
        enabled = bool(intvar.get())
        for widget in widgets:
            if enabled:
                if isinstance(widget, ttk.Combobox):
                    widget['state'] = ['readonly']
                else:
                    widget['state'] = 'normal'
            else:
                widget['state'] = 'disable'
        if any(iv.get() for iv in [self.save_nav, self.save_map]):
            self.save_button['state'] = 'normal'
        else:
            self.save_button['state'] = 'disable'

    def click_save(self) -> None:
        if self.try_run_save():
            self.close_window()

    def click_cancel(self) -> None:
        self.close_window()

    def close_window(self, *_) -> None:
        self.window.destroy()

    def try_run_save(self) -> bool:
        save_nav = bool(self.save_nav.get())
        save_map = bool(self.save_map.get())

        path_map = self.path_map.get().strip()
        path_nav = self.path_nav.get().strip()

        keep_open = bool(self.keep_open.get())

        degree_interval = 1
        interpolation = 'linear'

        if (save_nav and len(path_nav) == 0) or (save_map and len(path_map) == 0):
            tkinter.messagebox.showwarning(
                title=f'Error saving file', message=f'File paths must not be empty'
            )
            return False

        if save_map:
            degree_interval = self.get_float(
                self.degree_interval, name='degree interval', positive=True, finite=True
            )
            interpolation = self.map_interpolation.get()

        # If we get to this point, everything should (hopefully) be working

        saving_process = SavingProgress(
            self,
            save_nav=save_nav,
            path_nav=path_nav,
            save_map=save_map,
            path_map=path_map,
            degree_interval=degree_interval,
            interpolation=interpolation,
            keep_open=keep_open,
        )
        try:
            saving_process.run_save()
        except Exception as e:
            tkinter.messagebox.showwarning(
                title=f'Error saving files',
                message=f'Error: {e}',
            )
            return False
        finally:
            self.gui.get_observation()._remove_progress_hook()

        self.gui.help_hint.configure(
            text='File{s} saved successfully'.format(
                s='s' if save_nav and save_map else ''
            )
        )
        if keep_open:
            return False
        return True


class SavingProgress(Popup):
    def __init__(
        self,
        parent: SaveObservation,
        save_nav: bool,
        path_nav: str,
        save_map: bool,
        path_map: str,
        degree_interval: float,
        interpolation: str,
        keep_open: bool,
    ):
        self.parent = parent
        self.parent.saving_progress_window = self

        self.save_nav = save_nav
        self.path_nav = path_nav
        self.save_map = save_map
        self.path_map = path_map
        self.degree_interval = degree_interval
        self.interpolation = interpolation

        self.keep_open = keep_open

        self.make_window()
        self.make_required_widgets()

    def make_window(self) -> None:
        self.window = tk.Toplevel(self.parent.window)
        self.window.transient(self.parent.window)
        self.window.grab_set()
        self.window.title('Saving files...')

        x, y = (int(s) for s in self.parent.window.geometry().split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(sz='500x175', x=x + 50, y=y + 50)
        )

        self.frame = ttk.Frame(self.window)
        self.frame.pack(expand=True, fill='both')

    def make_required_widgets(self) -> None:
        if self.save_nav:
            self.nav_widgets = self.make_widgets('Saving navigated observation...')
        if self.save_map:
            self.map_widgets = self.make_widgets('Saving mapped observation...')
        if self.keep_open:
            button_frame = ttk.Frame(self.frame)
            button_frame.pack(padx=10, pady=10, fill='x')
            self.close_button = ttk.Button(
                button_frame,
                command=self.click_close,
                text='Close',
                width=10,
            )

    def make_widgets(self, label: str) -> dict[str, tk.Widget]:
        frame = ttk.Frame(self.frame)
        frame.pack(padx=10, pady=10, fill='x')
        widgets = {}

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill='x')

        widgets['label'] = ttk.Label(text_frame, text=label, justify='left')
        widgets['label'].pack(side='left')

        widgets['message'] = ttk.Label(text_frame, text='', justify='right')
        widgets['message'].pack(side='right')

        widgets['bar'] = ttk.Progressbar(frame)
        widgets['bar'].pack(fill='x')

        return widgets

    def run_save(self) -> None:
        save_kwargs = dict(show_progress=False, print_info=True)
        observation = self.parent.gui.get_observation()
        if self.save_nav:
            observation._set_progress_hook(SaveNavProgressHookGUI(**self.nav_widgets))
            observation.save_observation(self.path_nav, **save_kwargs)
            observation._remove_progress_hook()
        if self.save_map:
            n_wavelengths = len(self.parent.gui.get_observation().data)
            observation._set_progress_hook(
                SaveMapProgressHookGUI(n_wavelengths, **self.map_widgets)
            )
            observation.save_mapped_observation(
                self.path_map,
                degree_interval=self.degree_interval,
                interpolation=self.interpolation, # type: ignore
                **save_kwargs,
            )
            observation._remove_progress_hook()
        if self.keep_open:
            self.close_button.pack()
            self.window.title('Saving files complete')

    def click_close(self) -> None:
        self.destroy()
        self.parent.close_window()

    def destroy(self) -> None:
        self.window.destroy()
        self.parent.gui.get_observation()._remove_progress_hook()
        self.parent.saving_progress_window = None


# Progress hooks
class SaveProgressHookGUI(progress.SaveProgressHook):
    def __init__(
        self,
        label: ttk.Label,
        bar: ttk.Progressbar,
        message: ttk.Label,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label = label
        self.bar = bar
        self.message = message

    def update_bar(self, progress_change: float) -> None:
        if self.progress_parts.get(self.default_key, 0) >= 1:
            self.bar['value'] = 100
            self.message.configure(text='Complete')
        else:
            self.bar['value'] = self.overall_progress * 100
            self.message.configure(text=format(self.overall_progress, '.0%'))
        self.bar.update_idletasks()


class SaveNavProgressHookGUI(progress.SaveNavProgressHook, SaveProgressHookGUI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class SaveMapProgressHookGUI(progress.SaveMapProgressHook, SaveProgressHookGUI):
    def __init__(self, n_wavelengths: int, *args, **kwargs) -> None:
        super().__init__(n_wavelengths, *args, **kwargs)


# Artist settings popups
class ArtistSetting(Popup):
    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: PLOT_KEY,
        label: str | None = None,
        hint: str | None = None,
        callbacks: list[Callable[[], None]] | None = None,
        row: int | None = None,
    ):
        self.parent = parent
        self.key: PLOT_KEY = key
        self.gui = gui
        self._enable_callback = True
        if label is None:
            label = key
        self.label = label
        self.callbacks = callbacks

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
            parent, text='Edit', width=4, command=self.button_click
        )
        self.button.grid(column=1, row=row, sticky='e')

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
        self.gui.plot_settings[self.key]['visible'] = enabled
        self.gui.update_plot()

    def button_click(self) -> None:
        self.make_window()

    def make_window(self) -> None:
        self.window = tk.Toplevel(self.gui.root)
        self.window.title(self.label)
        self.window.grab_set()
        self.window.transient(self.gui.root)

        x, y = (int(s) for s in self.gui.root.geometry().split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(
                sz=self.get_window_size(),
                x=x + 50,
                y=y + 50,
            )
        )

        frame = ttk.Frame(self.window)
        frame.pack(side='bottom', fill='x')
        frame = ttk.Frame(frame)
        frame.pack(padx=10, pady=10)
        for idx, (text, command) in enumerate(
            [
                ('OK', self.click_ok),
                ('Cancel', self.click_cancel),
                ('Apply', self.click_apply),
            ]
        ):
            ttk.Button(frame, text=text, width=7, command=command).grid(
                row=0,
                column=idx,
                padx=2,
            )
        self.window.bind('<Escape>', self.close_window)

        window_frame = ttk.Frame(self.window)
        window_frame.pack(expand=True, fill='both')

        self.menu_frame = ttk.Frame(window_frame)
        self.menu_frame.pack(side='top', padx=10, pady=10)
        self.grid_frame = ttk.Frame(self.menu_frame)
        self.grid_frame.pack()
        self.make_menu()

    def make_menu(self) -> None:
        raise NotImplementedError

    def apply_settings(self) -> bool:
        raise NotImplementedError

    def run_callbacks(self) -> None:
        if self.callbacks is None:
            # Update artists in place
            settings = self.gui.plot_settings[self.key]
            if settings:
                plt.setp(self.gui.plot_handles[self.key], **settings)
        else:
            # Replot artists
            for callback in self.callbacks:
                callback()
        self.gui.update_plot()

    def close_window(self, *_) -> None:
        self.window.destroy()

    def click_ok(self) -> None:
        if self.apply_settings():
            self.run_callbacks()
            self.close_window()

    def click_apply(self) -> None:
        if self.apply_settings():
            self.run_callbacks()

    def click_cancel(self) -> None:
        self.close_window()

    def add_to_menu_grid(
        self, grid: list[tuple[tk.Widget, tk.Widget]], frame: ttk.Frame | None = None
    ) -> None:
        if frame is None:
            frame = self.grid_frame
        for label, widget in grid:
            row = frame.grid_size()[1]
            label.grid(row=row, column=0, sticky='w', pady=5)
            widget.grid(row=row, column=1, sticky='w')

    def get_window_size(self) -> str:
        return '350x350'


class PlotImageSetting(ArtistSetting):
    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]
        general_settings = self.gui.plot_settings['_']

        self.cmap = tk.StringVar(value=settings.setdefault('cmap', 'gray'))
        self.image_vmin = tk.StringVar(value=str(settings.setdefault('vmin', 0)))
        self.image_vmax = tk.StringVar(value=str(settings.setdefault('vmax', 100)))

        self.image_mode = tk.StringVar(
            value=general_settings.setdefault('image_mode', 'single')
        )
        self.image_idx_single = tk.StringVar(
            value=str(general_settings.setdefault('image_idx_single', 0))
        )
        self.image_idx_r = tk.StringVar(
            value=str(general_settings.setdefault('image_idx_r', 0))
        )
        self.image_idx_g = tk.StringVar(
            value=str(general_settings.setdefault('image_idx_g', 0))
        )
        self.image_idx_b = tk.StringVar(
            value=str(general_settings.setdefault('image_idx_b', 0))
        )
        self.image_gamma = tk.StringVar(
            value=str(general_settings.setdefault('image_gamma', 1))
        )

        # Image mode selection
        frame_top = ttk.Frame(self.grid_frame)
        frame_top.pack()
        ttk.Label(frame_top, text='Image mode:').pack()
        for mode, (fn, desc) in self.gui.image_modes.items():
            ttk.Radiobutton(
                frame_top,
                text=desc,
                value=mode,
                variable=self.image_mode,
            ).pack(fill='x')
        ttk.Label(frame_top, text='').pack()  # spacer

        # Settings
        frame = ttk.Frame(self.grid_frame)
        frame.pack()
        idx_max = self.gui.get_observation().data.shape[0] - 1

        class IndexInput(ttk.Spinbox):
            def __init__(self, textvariable: tk.StringVar):
                super().__init__(
                    frame,
                    textvariable=textvariable,
                    from_=0,
                    to=idx_max,
                    increment=1,
                    width=10,
                )

        self.grid: list[tuple[tk.Widget, tk.Widget, set[IMAGE_MODE]]] = [
            (
                ttk.Label(frame, text='Wavelength index (single): '),
                IndexInput(self.image_idx_single),
                {'single'},
            ),
            (
                ttk.Label(frame, text='Wavelength index (red): '),
                IndexInput(self.image_idx_r),
                {'rgb'},
            ),
            (
                ttk.Label(frame, text='Wavelength index (green): '),
                IndexInput(self.image_idx_g),
                {'rgb'},
            ),
            (
                ttk.Label(frame, text='Wavelength index (blue): '),
                IndexInput(self.image_idx_b),
                {'rgb'},
            ),
            (
                ttk.Label(frame, text='Matplotlib colormap: '),
                ttk.Combobox(
                    frame,
                    textvariable=self.cmap,
                    values=CMAPS,
                    width=10,
                ),
                {'single', 'sum'},
            ),
            (
                ttk.Label(frame, text='gamma: '),
                ttk.Spinbox(
                    frame,
                    textvariable=self.image_gamma,
                    from_=0,
                    to=100,
                    increment=0.2,
                    width=10,
                ),
                {'single', 'sum', 'rgb'},
            ),
            (
                ttk.Label(frame, text='vmin: '),
                ttk.Spinbox(
                    frame,
                    textvariable=self.image_vmin,
                    from_=0,
                    to=100,
                    increment=5,
                    width=10,
                ),
                {'single', 'sum'},
            ),
            (
                ttk.Label(frame, text='vmax: '),
                ttk.Spinbox(
                    frame,
                    textvariable=self.image_vmax,
                    from_=0,
                    to=100,
                    increment=5,
                    width=10,
                ),
                {'single', 'sum'},
            ),
            # TODO vmin/vmax/gamma
        ]
        self.add_to_menu_grid([(a, b) for a, b, c in self.grid], frame=frame)

        msg = '\n'.join(
            [
                'Images are scaled to vary from 0 to 100,',
                'so set vmin=0 and vmax=100 to show the',
                'entire dynamic range.',
            ]
        )
        ttk.Label(self.grid_frame, text=msg).pack()

        self.image_mode.trace_add('write', self.change_image_mode_radio)
        self.change_image_mode_radio()  # run initial setup

    def change_image_mode_radio(self, *_) -> None:
        mode = self.image_mode.get()
        for l, widget, modes in self.grid:
            if mode in modes:
                widget['state'] = 'normal'
            else:
                widget['state'] = 'disable'

    def get_idx(self, stirng_variable: tk.StringVar, name: str) -> int:
        sz = self.gui.get_observation().data.shape[0]
        return self.get_int(
            stirng_variable, name=name, positive=False, minimum=-sz, maximum=sz - 1
        )

    def apply_settings(self) -> bool:
        settings = {}
        general_settings = {}
        try:
            image_mode = self.image_mode.get()
            general_settings['image_mode'] = image_mode
            if image_mode == 'single':
                general_settings['image_idx_single'] = self.get_idx(
                    self.image_idx_single, 'wavelength index (single)'
                )
            if image_mode == 'rgb':
                general_settings['image_idx_r'] = self.get_idx(
                    self.image_idx_r, 'wavelength index (red)'
                )
                general_settings['image_idx_g'] = self.get_idx(
                    self.image_idx_g, 'wavelength index (green)'
                )
                general_settings['image_idx_b'] = self.get_idx(
                    self.image_idx_b, 'wavelength index (blue)'
                )

            general_settings['image_gamma'] = self.get_float(
                self.image_gamma, 'gamma', positive=False
            )
            if image_mode in {'single', 'sum'}:
                settings['vmin'] = self.get_float(
                    self.image_vmin, 'vmin', positive=False
                )
                settings['vmax'] = self.get_float(
                    self.image_vmax, 'vmax', positive=False
                )
                if settings['vmin'] >= settings['vmax']:
                    tkinter.messagebox.showwarning(
                        title='Error parsing limits',
                        message=f'vmin must be less than vmax',
                    )
                    return False
        except ValueError:
            return False

        if image_mode in {'single', 'sum'}:
            try:
                cmap = self.cmap.get()
                matplotlib.cm.get_cmap(cmap)
            except ValueError:
                tkinter.messagebox.showwarning(
                    title='Error parsing colormap',
                    message=f'Unrecognised matplotlib colormap {self.cmap.get()!r}',
                )
                return False
            settings['cmap'] = cmap

        self.gui.plot_settings[self.key].update(settings)
        self.gui.plot_settings['_'].update(general_settings)
        return True

    def get_window_size(self) -> str:
        return '350x600'


class PlotLineSetting(ArtistSetting):
    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]

        self.linewidth = tk.StringVar(value=str(settings.get('linewidth', '1.0')))
        self.linestyle = tk.StringVar(value=str(settings.get('linestyle', 'solid')))
        self.color = tk.StringVar(value=str(settings.get('color', 'red')))

        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Linewidth: '),
                    ttk.Spinbox(
                        self.grid_frame,
                        textvariable=self.linewidth,
                        from_=0.1,
                        to=10,
                        increment=0.1,
                        width=10,
                    ),
                ),
                (
                    ttk.Label(self.grid_frame, text='Linestyle: '),
                    ttk.Combobox(
                        self.grid_frame,
                        textvariable=self.linestyle,
                        values=LINESTYLES,
                        state='readonly',
                        width=10,
                    ),
                ),
                (
                    ttk.Label(self.grid_frame, text='Colour: '),
                    ColourButton(self.grid_frame, width=10, textvariable=self.color),
                ),
            ]
        )

    def apply_settings(self) -> bool:
        settings = self.gui.plot_settings[self.key]
        try:
            linewidth = self.get_float(self.linewidth, 'linewidth')
        except ValueError:
            return False

        settings['linewidth'] = linewidth
        settings['linestyle'] = self.linestyle.get()
        settings['color'] = self.color.get()
        return True


class PlotGridSetting(PlotLineSetting):
    def make_menu(self) -> None:
        super().make_menu()
        self.grid_interval = tk.StringVar(
            value=str(self.gui.plot_settings['_'].setdefault('grid_interval', 30))
        )

        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Grid interval (°): '),
                    ttk.Combobox(
                        self.grid_frame,
                        textvariable=self.grid_interval,
                        values=GRID_INTERVALS,
                        width=10,
                    ),
                ),
            ]
        )

    def apply_settings(self) -> bool:
        try:
            grid_interval = self.get_float(self.grid_interval, 'grid interval')
        except ValueError:
            return False
        self.gui.plot_settings['_']['grid_interval'] = grid_interval
        return super().apply_settings()


class PlotRingsSetting(PlotLineSetting):
    def make_menu(self) -> None:
        super().make_menu()
        radii_selected = self.gui.get_observation().ring_radii.copy()
        radii_options = data_loader.get_ring_radii().setdefault(
            self.gui.get_observation().target, {}
        )

        ttk.Label(self.menu_frame, text='').pack(fill='x')  # Add a spacer

        self.checkbox_dict: dict[tuple[float, ...], tk.IntVar] = {}
        for name, radii in sorted(radii_options.items(), key=lambda x: x[1]):
            key = tuple(radii)
            if key in self.checkbox_dict:
                # Skip repeated rings (for Neptune where multiple rings have same radii
                # on fact sheet)
                continue
            iv = tk.IntVar()
            self.checkbox_dict[key] = iv
            label = '{n}  ({r})'.format(
                n=name, r=', '.join(format(r, 'g') + 'km' for r in radii)
            )
            ttk.Checkbutton(self.menu_frame, text=label, variable=iv).pack(fill='x')
            iv.set(all(r in radii_selected for r in radii))
        for radii, iv in self.checkbox_dict.items():
            # Radii will be indicated by checkbox, so remove them from the text list
            # Do after creating all checkboxes so that overlapping rings (e.g. Saturn's
            # B & C rings) don't break logic.
            if iv.get():
                radii_selected -= set(radii)

        value = '\n'.join(str(r) for r in sorted(radii_selected))
        label = '\n'.join(
            [
                'Manually list{s} ring radii in km from the'.format(
                    s=' more' if self.checkbox_dict else ''
                ),
                'target\'s centre below. Each radius should',
                'be listed on a new line:',
            ]
        )
        ttk.Label(self.menu_frame, text='\n' + label).pack(fill='x')
        self.txt = tkinter.scrolledtext.ScrolledText(self.menu_frame)
        self.txt.pack(fill='both')
        self.txt.insert('1.0', value)

    def apply_settings(self) -> bool:
        rings: list[float] = []
        try:
            string = self.txt.get('1.0', 'end')
            for value in string.splitlines():
                value = value.strip()
                if value:
                    rings.append(self.get_float(value, 'ring radius'))
        except ValueError:
            return False

        for radii, iv in self.checkbox_dict.items():
            if iv.get():
                rings.extend(radii)

        self.gui.get_observation().ring_radii.clear()
        self.gui.get_observation().ring_radii.update(rings)
        return super().apply_settings()

    def get_window_size(self) -> str:
        return '350x600'


class PlotScatterSetting(ArtistSetting):
    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]

        self.marker = tk.StringVar(value=str(settings.setdefault('marker', 'o')))
        self.size = tk.StringVar(value=str(settings.setdefault('s', '36')))
        self.color = tk.StringVar(value=str(settings.setdefault('color', 'red')))

        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Marker: '),
                    ttk.Combobox(
                        self.grid_frame,
                        textvariable=self.marker,
                        values=MARKERS,
                        width=10,
                    ),
                ),
                (
                    ttk.Label(self.grid_frame, text='Size: '),
                    ttk.Spinbox(
                        self.grid_frame,
                        textvariable=self.size,
                        from_=1,
                        to=100,
                        increment=1,
                        width=10,
                    ),
                ),
                (
                    ttk.Label(self.grid_frame, text='Colour: '),
                    ColourButton(self.grid_frame, width=10, textvariable=self.color),
                ),
            ]
        )

    def apply_settings(self) -> bool:
        settings = self.gui.plot_settings[self.key]

        try:
            marker = self.marker.get()
            matplotlib.markers.MarkerStyle(marker)
        except ValueError:
            tkinter.messagebox.showwarning(
                title='Error parsing marker',
                message=f'Unrecognised matplotlib marker {self.marker.get()!r}',
            )
            return False

        try:
            size = self.get_float(self.size, 'size')
        except ValueError:
            return False

        settings['marker'] = marker
        settings['s'] = size
        settings['color'] = self.color.get()
        return True


class PlotCoordinatesSetting(PlotScatterSetting):
    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: PLOT_KEY,
        coordinate_list: list[tuple[float, float]],
        menu_label: str,
        label: str | None = None,
        hint: str | None = None,
        callbacks: list[Callable[[], None]] | None = None,
        **kwargs,
    ):
        self.coordinate_list = coordinate_list
        self.menu_label = menu_label
        super().__init__(gui, parent, key, label, hint, callbacks, **kwargs)

    def make_menu(self) -> None:
        super().make_menu()
        value = '\n'.join(f'{a}, {b}' for a, b in self.coordinate_list)
        ttk.Label(self.menu_frame, text='\n' + self.menu_label).pack(fill='x')
        self.txt = tkinter.scrolledtext.ScrolledText(self.menu_frame)
        self.txt.pack(fill='both')
        self.txt.insert('1.0', value)

    def apply_settings(self) -> bool:
        values: list[tuple[float, float]] = []
        try:
            string = self.txt.get('1.0', 'end')
            for line in string.splitlines():
                line = line.strip().lstrip('(').rstrip(')')
                if line:
                    coordinates = line.split(',')
                    if len(coordinates) != 2:
                        tkinter.messagebox.showwarning(
                            title=f'Error parsing coordinates',
                            message=f'Line {line!r} must have exactly two values separated by a comma',
                        )
                        return False
                    values.append(
                        tuple(
                            self.get_float(c, 'coordinate', positive=False)
                            for c in coordinates
                        )
                    )
        except ValueError:
            return False
        self.coordinate_list.clear()
        self.coordinate_list[:] = values
        return super().apply_settings()

    def get_window_size(self) -> str:
        return '350x600'


class PlotTextSetting(ArtistSetting):
    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]
        self.color = tk.StringVar(value=str(settings.get('color', 'red')))
        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Colour: '),
                    ColourButton(self.grid_frame, width=10, textvariable=self.color),
                ),
            ]
        )

    def apply_settings(self) -> bool:
        settings = self.gui.plot_settings[self.key]
        settings['color'] = self.color.get()
        return True


class PlotOutlinedTextSetting(ArtistSetting):
    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]
        self.color = tk.StringVar(value=str(settings.setdefault('color', 'red')))
        self.outline_color = tk.StringVar(
            value=str(settings.setdefault('outline_color', 'red'))
        )

        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Colour: '),
                    ColourButton(self.grid_frame, width=10, textvariable=self.color),
                ),
                (
                    ttk.Label(self.grid_frame, text='Outline: '),
                    ColourButton(
                        self.grid_frame, width=10, textvariable=self.outline_color
                    ),
                ),
            ]
        )

    def apply_settings(self) -> bool:
        settings = self.gui.plot_settings[self.key]
        settings['color'] = self.color.get()
        settings['outline_color'] = self.outline_color.get()
        return True


class GenericOtherBodySetting(ArtistSetting):
    def add_other_body_menu_setting(self):
        value = '\n'.join(
            b.target for b in self.gui.get_observation().other_bodies_of_interest
        )
        label = '\n'.join(
            [
                'List other bodies of interest to',
                'mark (e.g. moons). Body names should',
                'be recognisable by SPICE (e.g "Europa"',
                'or "502") with each body listed on a',
                'new line:',
            ]
        )
        ttk.Label(self.menu_frame, text='\n' + label).pack(fill='x')
        self.txt = tkinter.scrolledtext.ScrolledText(self.menu_frame)
        self.txt.pack(fill='both')
        self.txt.insert('1.0', value)

    def apply_other_body_setting(self) -> bool:
        bodies: list[Body] = []
        string = self.txt.get('1.0', 'end')
        for line in string.splitlines():
            line = line.strip()
            if line:
                try:
                    bodies.append(self.gui.get_observation().create_other_body(line))
                except NotFoundError:
                    tkinter.messagebox.showwarning(
                        title=f'Error parsing target name',
                        message=f'Target {line!r} is not recognised by SPICE',
                    )
                    return False
        self.gui.get_observation().other_bodies_of_interest.clear()
        self.gui.get_observation().other_bodies_of_interest[:] = bodies
        return True

    def get_window_size(self) -> str:
        return '350x600'


class PlotOtherBodyScatterSetting(PlotScatterSetting, GenericOtherBodySetting):
    def make_menu(self) -> None:
        super().make_menu()
        self.add_other_body_menu_setting()

    def apply_settings(self) -> bool:
        return self.apply_other_body_setting() and super().apply_settings()


class PlotOtherBodyTextSetting(PlotTextSetting, GenericOtherBodySetting):
    def make_menu(self) -> None:
        super().make_menu()
        self.add_other_body_menu_setting()

    def apply_settings(self) -> bool:
        return self.apply_other_body_setting() and super().apply_settings()


# Generic artist settting elements
class ColourButton(ttk.Button):
    def __init__(
        self,
        parent: tk.Widget,
        *args,
        textvariable: tk.StringVar,
        **kwargs,
    ) -> None:
        self.parent = parent
        self.textvariable = textvariable
        self.textvariable.set(matplotlib.colors.to_hex(textvariable.get()))
        super().__init__(
            parent, *args, command=self.command, text=self.textvariable.get(), **kwargs
        )

    def command(self):
        _, color = tkinter.colorchooser.askcolor(
            initialcolor=self.textvariable.get(),
            parent=self.parent,
            title='Pick colour',
        )
        if color is not None:
            self.textvariable.set(color)
            self.configure(text=color)


class NumericEntry:
    # TODO add validation
    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: SETTER_KEY,
        label: str | None = None,
        row: int | None = None,
        add_callbacks: list[SETTER_KEY] | None = None,
        **kw,
    ):
        self.parent = parent
        self.key: SETTER_KEY = key
        self.gui = gui
        self._enable_callback = True

        if label is None:
            label = key
        self.label = ttk.Label(parent, text=label + ': ')

        self.sv = tk.StringVar()
        self.sv.trace_add('write', self.text_input)
        self.entry = ttk.Entry(parent, width=10, textvariable=self.sv)

        self.gui.ui_callbacks[self.key].add(self.update_text)
        if add_callbacks:
            for k in add_callbacks:
                self.gui.ui_callbacks[k].add(self.update_text)

        self.update_text()

        if row is None:
            row = parent.grid_size()[1]
        self.label.grid(row=row, column=0, sticky='e', **kw)
        self.entry.grid(row=row, column=1, sticky='w', **kw)

        self.disable_keybindings()

    def disable_keybindings(self):
        for bindings in self.gui.shortcuts.values():
            for binding in bindings:
                if binding not in self.gui.shortcuts_to_keep_in_entry:
                    self.entry.bind(binding, self.gui.ignore_keypress)

    def update_text(self) -> None:
        if not self._enable_callback:
            return
        self._enable_callback = False
        value = self.gui.getters[self.key]()
        self.sv.set(self.format_value(value))
        self.entry.configure(foreground='black')
        self._enable_callback = True

    def format_value(self, value: float) -> str:
        return format(value, '.8g')

    def text_input(self, *_) -> None:
        if not self._enable_callback:
            return
        self._enable_callback = False
        value = self.sv.get()
        try:
            self.gui.set_value(self.key, float(value))
            self.entry.configure(foreground='black')
        except (ValueError, ZeroDivisionError):
            self.entry.configure(foreground='red')
        self._enable_callback = True


# Plotting stuff
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


class CustomNavigationToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, *, pack_toolbar: bool = True, gui: GUI) -> None:
        # Default tooltips don't work with tk (on my laptop with dark mode at lease)
        # so disable them here by setting to None, then use our custom tooltips instead.
        # This list also removes the Save and Subplots buttons which we don't want.
        self.toolitems = (
            ('Home', None, 'home', 'home'),
            ('Back', None, 'back', 'back'),
            ('Forward', None, 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', None, 'move', 'pan'),
            ('Zoom', None, 'zoom_to_rect', 'zoom'),
        )
        super().__init__(canvas, window, pack_toolbar=pack_toolbar)
        try:
            self._message_label.configure(foreground='#666666')
        except:
            pass
        try:
            for name, button in self._buttons.items():
                # Get default tooltips from super() and use them
                for text, tooltip_text, image_file, callback in super().toolitems:
                    if text == name:
                        hint = tooltip_text.replace('\n', ', ')
                        gui.add_tooltip(button, hint)
                        break
        except:
            pass
