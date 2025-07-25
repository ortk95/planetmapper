# pylint: disable=attribute-defined-outside-init,protected-access
import functools
import math
import os
import platform
import sys
import tkinter as tk
import tkinter.colorchooser
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
import tkinter.scrolledtext
import traceback
import webbrowser
from abc import ABC, abstractmethod
from collections import defaultdict
from tkinter import ttk
from typing import Any, Callable, Literal, ParamSpec, TypedDict, TypeVar

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.markers
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from . import base, common, data_loader, progress, utils
from .body import BasicBody, Body, NotFoundError
from .body_xy import MapKwargs
from .observation import Observation

Widget = TypeVar('Widget', bound=tk.Widget)
SetterKey = Literal[
    'x0',
    'y0',
    'r0',
    'rotation',
    'step',
    'plate_scale_arcsec',
    'plate_scale_km',
    'wcs_offset_ra',
    'wcs_offset_dec',
    'wcs_offset_rotation',
    'wcs_offset_scale',
]
PlotKey = Literal[
    'image',
    'grid',
    'limb',
    'limb_illuminated',
    'terminator',
    'ring',
    'pole',
    'coordinate_of_interest_lonlat',
    'coordinate_of_interest_radec',
    'other_body_of_interest_marker',
    'other_body_of_interest_label',
    'marked_coord',
    '_',
]
ImageMode = Literal['sum', 'single', 'rgb']

DEFAULT_PLOT_SETTINGS: dict[PlotKey, dict] = {
    'grid': dict(zorder=3.1, color='#333', linewidth=1, linestyle='dotted'),
    'terminator': dict(zorder=3.2, color='w', linewidth=1, linestyle='dashed'),
    'limb': dict(zorder=3.3, color='w', linewidth=0.5, linestyle='solid'),
    'limb_illuminated': dict(zorder=3.31, color='w', linewidth=1, linestyle='solid'),
    'ring': dict(zorder=3.4, color='w', linewidth=0.5, linestyle='solid'),
    'pole': dict(zorder=3.5, color='k', outline_color='w'),
    'coordinate_of_interest_lonlat': dict(zorder=3.6, marker='x', color='k', s=36),
    'coordinate_of_interest_radec': dict(zorder=3.7, marker='+', color='k', s=36),
    'other_body_of_interest_marker': dict(zorder=3.8, marker='+', color='w', s=36),
    'other_body_of_interest_label': dict(zorder=3.81, color='grey'),
    'marked_coord': dict(zorder=4, color='cyan', linewidth=0.5, linestyle='solid'),
    'image': dict(zorder=0.9, cmap='inferno'),
    '_': dict(
        grid_interval=30,
        grid_lat_limit=90,
        grid_planetocentric=False,
        image_mode='single',
        image_idx_single=0,
        image_idx_r=0,
        image_idx_g=1,
        image_idx_b=2,
        image_gamma=1,
        image_vmin=0,
        image_vmax=100,
        image_limit_type='absolute',
    ),
}


LINESTYLES = ('solid', 'dashed', 'dotted', 'dashdot')
MARKERS = ('x', '+', 'o', '.', '*', 'v', '^', '<', '>', ',', 'D', 'd', '|', '_')
GRID_INTERVALS = ('10', '30', '45', '90')
CMAPS = ('gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis')
LIMIT_TYPES = ('absolute', 'percentile')

MAP_INTERPOLATIONS = ('nearest', 'linear', 'quadratic', 'cubic')
MAP_PROJECTIONS = ('rectangular', 'orthographic', 'azimuthal', 'azimuthal equal area')

DEFAULT_HINT = ''


# Deal with X11 font bug by replacing high codepoint chars with ASCII equivalents.
# This seems to prevent the use of fonts which cause the X_OpenFont error which
# XQuartz was producing when trying to run planetmapper over SSH on mac. This is a bit
# of a hack and produces an uglier UI, but is better than always crashing.
# TODO remove this when the bug is fixed in XQuartz
# https://github.com/ortk95/planetmapper/issues/145
try:
    USE_X11_FONT_BUGFIX = bool(os.environ['PLANETMAPPER_USE_X11_FONT_BUGFIX'])
except KeyError:
    USE_X11_FONT_BUGFIX = False  # pyright: ignore[reportConstantRedefinition]
X11_FONT_BUGRIX_TRANSLATIONS = str.maketrans(
    {
        '↖': None,
        '↑': '^',
        '↗': None,
        '←': '<',
        '→': '>',
        '↙': None,
        '↓': 'v',
        '↘': None,
        '↺': '<',
        '↻': '>',
    }
)


_X11_ERROR_HELP_URL = (
    'https://planetmapper.readthedocs.io/en/latest/common_issues.html#ssh-errors'
)
_X11_ERROR_HELP_TEXT = (
    'Check you have X11 forwarding set up correctly - see the help page for more info:\n'
    + _X11_ERROR_HELP_URL
)

T = TypeVar('T')
P = ParamSpec('P')


def _add_help_note_to_x11_errors(fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def decorated(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except tkinter.TclError as e:
            if 'no $DISPLAY' in str(e):
                note = _X11_ERROR_HELP_TEXT
                if note not in e.args[0]:
                    e.args = (e.args[0] + '\n\n' + note,)
            raise e

    return decorated


def _run_gui_from_cli(*args: str | None) -> None:
    """Called with `planetmapper` from the command line"""
    if USE_X11_FONT_BUGFIX:
        print('*** Using X11 font bugfix ***')
    gui = GUI()
    if args and args[0] is not None:
        try:
            gui.set_observation(Observation(args[0]))
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            print(f'Error loading observation: {e}')
            sys.exit(1)
    gui.run()


def run_gui(path: str | os.PathLike | None = None) -> None:
    """
    Launch the PlanetMapper Graphical User Interface (GUI) to fit observations.

    This is the Python equivalent of running `planetmapper` from the command line. See
    :ref:`the user interface documentation <gui examples>` for more details about how to
    use the GUI.

    See also :func:`Observation.run_gui`.

    Args:
        path: Optionally specify a FITS file to open in the GUI. If this is provided,
            the GUI will open with the observation from the FITS file loaded and ready
            to fit. If not provided, you can select a file to open from the GUI.
            Passed to the `path` argument of :class:`planetmapper.Observation`.
    """
    gui = GUI()
    if path is not None:
        gui.set_observation(Observation(path))
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

    MINIMUM_SIZE = (400, 650)
    DEFAULT_GEOMETRY = '800x650+15+15'
    CONTROLS_WIDTH = 260

    def __init__(self, allow_open: bool = True) -> None:
        self.allow_open = allow_open

        self._popups: list[Popup] = []

        self._observation: Observation | None = None
        self._observation_wavelengths: np.ndarray | None = None
        self._observation_wavelengths_fmt = 'f'
        self._observation_wavelengths_unit = ''
        self._observation_full_path: str = ''
        self._observation_filename: str = ''

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
            self.copy_formatted_coord_values: ['<Control-Shift-C>'],
            self.display_header: ['<Control-h>'],
        }
        self.shortcuts_to_keep_in_entry = ['<Control-s>', '<Control-o>']

        self.setter_callbacks: defaultdict[SetterKey, list[Callable[[float], Any]]] = (
            defaultdict(
                list,
                {
                    'x0': [lambda f: self.get_observation().set_x0(f)],
                    'y0': [lambda f: self.get_observation().set_y0(f)],
                    'r0': [lambda f: self.get_observation().set_r0(f)],
                    'rotation': [lambda f: self.get_observation().set_rotation(f)],
                    'step': [self.set_step],
                    'plate_scale_arcsec': [
                        lambda f: self.get_observation().set_plate_scale_arcsec(f)
                    ],
                    'plate_scale_km': [
                        lambda f: self.get_observation().set_plate_scale_km(f)
                    ],
                    'wcs_offset_ra': [lambda f: self._set_wcs_offsets(dra_arcsec=f)],
                    'wcs_offset_dec': [lambda f: self._set_wcs_offsets(ddec_arcsec=f)],
                    'wcs_offset_rotation': [
                        lambda f: self._set_wcs_offsets(drotation=f)
                    ],
                    'wcs_offset_scale': [
                        lambda f: self._set_wcs_offsets(d_scale_arcsec=f)
                    ],
                },
            )
        )
        self.ui_callbacks: defaultdict[SetterKey | None, set[Callable[[], Any]]] = (
            defaultdict(set)
        )

        self.getters: dict[SetterKey, Callable[[], float]] = {
            'x0': lambda: self.get_observation().get_x0(),
            'y0': lambda: self.get_observation().get_y0(),
            'r0': lambda: self.get_observation().get_r0(),
            'rotation': lambda: self.get_observation().get_rotation(),
            'step': lambda: self.step_size,
            'plate_scale_arcsec': lambda: self.get_observation().get_plate_scale_arcsec(),
            'plate_scale_km': lambda: self.get_observation().get_plate_scale_km(),
            'wcs_offset_ra': lambda: self._get_wcs_offsets()[0],
            'wcs_offset_dec': lambda: self._get_wcs_offsets()[1],
            'wcs_offset_rotation': lambda: self._get_wcs_offsets()[3],
            'wcs_offset_scale': lambda: self._get_wcs_offsets()[2],
        }
        self.plot_handles: defaultdict[PlotKey, list[Artist]] = defaultdict(list)
        self.plot_settings: defaultdict[PlotKey, dict] = defaultdict(dict)
        for k, v in DEFAULT_PLOT_SETTINGS.items():
            self.plot_settings[k] = v.copy()

        # {section_title: [(callback, text, hint, required_key), ...]}
        self.disc_finding_routines: dict[
            str, list[tuple[Callable[[], Any], str, str, str | None]]
        ] = {
            'Reset disc': [
                (
                    lambda: self.get_observation().reset_disc_params(),
                    'Reset all disc parameters',
                    'Reset the disc parameters to their initial values',
                    None,
                ),
                (
                    lambda: self.get_observation().centre_disc(),
                    'Centre disc in image',
                    'Centre the target\'s planetary disc and make it fill ~90% of the observation',
                    None,
                ),
                (
                    lambda: self.get_observation().rotate_north_to_top(),
                    'Rotate north to top',
                    'Rotate the disc so that the north pole of the target is at the top of the image',
                    None,
                ),
            ],
            'Use FITS header metadata': [
                (
                    lambda: self.get_observation().disc_from_header(),
                    'Use PlanetMapper metadata',
                    'Set disc parameters using information in the observation\'s FITS header generated by any previous runs of PlanetMapper',
                    'header',
                ),
            ],
            'Use WCS data from FITS header': [
                (
                    lambda: self.get_observation().disc_from_wcs(
                        suppress_warnings=True, validate=False, use_header_offsets=False
                    ),
                    'Use WCS position, rotation & scale',
                    'Set all disc parameters using approximate WCS information in the observation\'s FITS header',
                    'wcs',
                ),
                (
                    lambda: self.get_observation().position_from_wcs(
                        suppress_warnings=True, validate=False, use_header_offsets=False
                    ),
                    'Use WCS position',
                    'Set disc position using approximate WCS information in the observation\'s FITS header',
                    'wcs',
                ),
                (
                    lambda: self.get_observation().rotation_from_wcs(
                        suppress_warnings=True, validate=False, use_header_offsets=False
                    ),
                    'Use WCS rotation',
                    'Set disc rotation using approximate WCS information in the observation\'s FITS header',
                    'wcs',
                ),
                (
                    lambda: self.get_observation().plate_scale_from_wcs(
                        suppress_warnings=True, validate=False, use_header_offsets=False
                    ),
                    'Use WCS plate scale',
                    'Set plate scale using approximate WCS information in the observation\'s FITS header',
                    'wcs',
                ),
            ],
            'Fit observation': [
                (
                    lambda: self.get_observation().fit_disc_position(),
                    'Fit disc position',
                    'Set x0 and y0 so that the planet\'s disc is fit to the brightest part of the data (this may take a few seconds)',
                    None,
                ),
                (
                    lambda: self.get_observation().fit_disc_radius(),
                    'Fit disc radius',
                    'Set r0 by calculating the radius around (x0, y0) where the brightness decrease is the fastest (this may take a few seconds)',
                    None,
                ),
            ],
        }
        self._observation_available_disc_finding_routines: set[str] = set()
        self._disc_finding_enableable_buttons: dict[str, list[ttk.Button]] = {}

        self.kernels: list[str] = [
            os.path.join(base.get_kernel_path(), pattern)
            for pattern in base._KERNEL_DATA['kernel_patterns']
        ]

        self.delayed_actions: dict[str, tuple[Callable[[], Any], str]] = {}

        self.event_time_to_ignore = None
        self.gui_built = False

        self._cached_wcs_offset_info: (
            tuple[tuple[float, float, float, float], tuple[float, float, float, float]]
            | None
        ) = None
        self._wcs_entries_to_enable: list[NumericEntry | EnableableLabel] = []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @_add_help_note_to_x11_errors
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

        self.image_modes: dict[ImageMode, tuple[Callable[[], np.ndarray], str]] = {
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

        self.plot_settings['_'].update(
            image_idx_single=0,
            image_idx_r=0,
            image_idx_g=1,
            image_idx_b=2,
        )

        try:
            self._observation_wavelengths = (
                self.get_observation().get_wavelengths_from_header()
            )
            # Get an appropriate number of decimal places to represent wavelength steps
            decimals = -np.log10(np.nanmean(np.diff(self._observation_wavelengths)))
            if np.isfinite(decimals):
                decimals = int(max(1, decimals + 0.5)) + 1
                self._observation_wavelengths_fmt = f'.{decimals}f'
            else:
                self._observation_wavelengths_fmt = 'f'
            self._observation_wavelengths_unit = str(
                self.get_observation().header.get('CUNIT3', '')
            ).strip()
            if self._observation_wavelengths_unit:
                self._observation_wavelengths_unit = (
                    ' ' + self._observation_wavelengths_unit
                )
        except utils.GetWavelengthsError:
            self._observation_wavelengths = None

        path = observation.path
        if path is None:
            self._observation_full_path = ''
            self._observation_filename = ''
        else:
            basename = os.path.basename(path)
            self._observation_filename = basename
            if basename == path:
                path = os.path.abspath(path)
            self._observation_full_path = path

        self.update_observation_available_disc_finding_routines()

        self.click_locations = []
        self.clear_click_location()

        self._cached_wcs_offset_info = None

        if self.gui_built:
            self.run_all_ui_callbacks()
            self.rebuild_plot()
            self.root.title(self.get_observation().get_description(multiline=False))
            self.reset_help_hint()
            self.enable_disc_finding_buttons()

    def update_observation_available_disc_finding_routines(self):
        routines = self._observation_available_disc_finding_routines
        observation = self.get_observation()
        routines.clear()
        try:
            observation.disc_from_wcs(suppress_warnings=True)
            routines.add('wcs')
        except ValueError:
            pass
        try:
            observation.disc_from_header()
            routines.add('header')
        except ValueError:
            pass
        observation.reset_disc_params()

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
        self.root.protocol('WM_DELETE_WINDOW', self.quit)

        self.configure_style(self.root)
        self.root.title(self.get_observation().get_description(multiline=False))

        # On some systems (e.g. over SSH/X11), building the GUI can be a bit slow,
        # especially creating the matplotlib plot. Therefore, create the initial bare
        # version of the GUI, then display it with update to give the user some feedback
        # and impression of progress while the rest of the GUI is being built.
        loading_frame = ttk.Frame(self.root)
        loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        loading_label = ttk.Label(
            loading_frame,
            text='Loading...',
            justify='center',
            foreground='grey70',
            font=('TkDefaultFont', self._default_font_size + 20),
        )
        loading_label.place(relx=0.5, rely=0.5, anchor='center')
        self.root.update_idletasks()

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
        self.root.update_idletasks()

        # Figure can sometimes be initialised with a slightly incorrect shape, so force
        # it to fit the plot frame nicely here once the frame has its final shape.
        self.fig.set_size_inches(
            self.canvas_frame.winfo_width() / self.fig.dpi,
            self.canvas_frame.winfo_height() / self.fig.dpi,
        )
        self.update_plot()

        loading_frame.destroy()
        self.gui_built = True

    def quit(self) -> None:
        plt.close(self.fig)
        self.root.destroy()
        self.gui_built = False

    def configure_style(self, root: tk.Tk | None) -> None:
        if root is None:
            root = self.root
        self.style = ttk.Style(root)
        self.style.theme_use('default')
        for element in ['TEntry', 'TCombobox', 'TSpinbox', 'TButton', 'TLabel']:
            self.style.configure(
                element,
                foreground='black',
                insertcolor='black',
                fieldbackground='white',
                selectbackground='#bdf',
                selectforeground='black',
            )

        self._default_font = tkinter.font.nametofont('TkDefaultFont').actual()
        self._default_font_size = self._default_font['size']

        self.style.map('TScale', troughcolor=[('disabled', '#d9d9d9')])
        self.style.configure(
            'Small.TCheckbutton',
            font=('TkDefaultFont', int(self._default_font_size * 0.85)),
        )

    def build_controls(self) -> None:
        self.notebook = ttk.Notebook(self.controls_frame)
        self.notebook.pack(fill='both', expand=True)
        self.build_main_controls_tab()
        self.build_disc_finding_controls_tab()
        self.build_plot_settings_controls_tab()
        self.build_coords_tab()
        self.build_help_tab()

    def build_main_controls_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Controls')

        buttons: list[
            tuple[
                str,
                str,
                Callable[[], None],
                int | tuple[int, int],
                int | tuple[int, int],
            ]
        ] = [
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
            (
                'View FITS header',
                'View the FITS header for the observation',
                self.display_header,
                (0, 2) if self.allow_open else 1,
                1,
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
            wide_buttons=True,
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
            entry_tooltip='Set the rotation of the disc in degrees',
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
            entry_tooltip='Set the equatorial radius of the disc, r0, in pixels',
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
        buttons: list[
            tuple[
                str,
                str,
                Callable[[], None],
                int | tuple[int, int],
                int | tuple[int, int],
            ]
        ],
        button_tooltip_base: str,
        entry_tooltip: str,
        numeric_entries: list[SetterKey | tuple[SetterKey, str]],
        ipadx=30,
        ipady=1,
        add_callbacks: list[SetterKey] | None = None,
        wide_buttons: bool = False,
        **kw,
    ) -> None:
        label_frame = ttk.LabelFrame(frame, text=label)
        label_frame.pack(fill='x', pady=3, ipadx=1, ipady=1)

        if buttons:
            button_frame = ttk.Frame(label_frame)
            button_frame.pack(
                fill='x' if wide_buttons else 'none',
                expand=wide_buttons,
                padx=5 if wide_buttons else 0,
            )
            for text, hint, fn, column, row in buttons:
                if isinstance(column, tuple):
                    column, columnspan = column
                else:
                    columnspan = 1
                if isinstance(row, tuple):
                    row, rowspan = row
                else:
                    rowspan = 1
                text = self.maybe_replace_string_with_x11_bugfix(text)
                self.add_tooltip(
                    ttk.Button(button_frame, text=text, command=fn, width=1),
                    button_tooltip_base.format(hint=hint),
                    fn,
                ).grid(
                    column=column,
                    row=row,
                    columnspan=columnspan,
                    rowspan=rowspan,
                    ipadx=ipadx,
                    ipady=ipady,
                    padx=2,
                    pady=2,
                    sticky='we',
                )
                if wide_buttons:
                    button_frame.grid_columnconfigure(column, weight=1)

        if numeric_entries:
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

        # Image
        frame = ttk.LabelFrame(menu, text='Observation')
        frame.pack(fill='x', pady=5)
        frame.grid_columnconfigure(0, weight=1)
        self.image_setting = PlotImageSetting(
            self,
            frame,
            'image',
            label='Observed image',
            hint='the image of your observation',
            callbacks=[self.replot_image],
        )
        self.ui_callbacks[None].add(self.image_setting.update_tool_ui_state)

        # Plot features
        frame = ttk.LabelFrame(menu, text='Plotted features')
        frame.pack(fill='x', pady=5)
        frame.grid_columnconfigure(0, weight=1)
        PlotLineSetting(self, frame, 'limb', label='Limb', hint='the target\'s limb')
        PlotLineSetting(
            self,
            frame,
            'limb_illuminated',
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
            'ring',
            label='Rings',
            hint='rings around the target (click Edit to define ring radii)',
            callbacks=[self.replot_rings],
        )
        PlotOutlinedTextSetting(
            self,
            frame,
            'pole',
            label='Poles',
            hint='the target\'s poles',
        )
        PlotCoordinatesSetting(
            self,
            frame,
            'coordinate_of_interest_lonlat',
            label='Lon/Lat POI',
            hint='points of interest on the surface of the target (click Edit to define POI)',
            callbacks=[self.replot_coordinates_lonlat],
            coordinate_list=self.get_observation().coordinates_of_interest_lonlat,
            menu_label=(
                'List of Lon/Lat points of interest. '
                'Coordinates should be written as comma '
                'separated "lon, lat" values, with each '
                'coordinate pair on a new line:'
            ),
        )
        PlotCoordinatesSetting(
            self,
            frame,
            'coordinate_of_interest_radec',
            label='RA/Dec POI',
            hint='points of interest in the sky (click Edit to define POI)',
            callbacks=[self.replot_coordinates_radec],
            coordinate_list=self.get_observation().coordinates_of_interest_radec,
            menu_label=(
                'List of RA/Dec points of interest. '
                'Coordinates should be written as comma '
                'separated "ra, dec" values, with each '
                'coordinate pair on a new line:'
            ),
        )
        PlotOtherBodyScatterSetting(
            self,
            frame,
            'other_body_of_interest_marker',
            label='Other bodies (e.g. moons)',
            hint='other bodies of interest (click Edit to specify other bodies to show, e.g. moons)',
            callbacks=[self.replot_other_bodies],
        )
        PlotOtherBodyTextSetting(
            self,
            frame,
            'other_body_of_interest_label',
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
        self.notebook.add(frame, text='Disc')
        for label, routines in self.disc_finding_routines.items():
            label_frame = ttk.LabelFrame(frame, text=label)
            label_frame.pack(fill='x', pady=10)
            for fn, name, description, required_key in routines:
                button = ttk.Button(
                    label_frame,
                    text=name,
                    command=self.make_disc_finding_fn(fn),
                )
                self.add_tooltip(button, description).pack(fill='x', pady=2, padx=5)
                if required_key is not None:
                    self._disc_finding_enableable_buttons.setdefault(
                        required_key, []
                    ).append(button)

            if label == 'Use WCS data from FITS header':
                self.build_wcs_offset_section(label_frame)

        self.enable_disc_finding_buttons()

    def build_help_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        frame.pack()
        self.notebook.add(frame, text='Help')

        wraplength = self.CONTROLS_WIDTH - 10

        label_frame = ttk.LabelFrame(frame, text='About')
        label_frame.pack(fill='x', pady=10)
        label = ttk.Label(
            label_frame,
            text=f'PlanetMapper {common.__version__}',
            justify='center',
            font=('TkDefaultFont', int(self._default_font_size * 1.6)),
            wraplength=wraplength,
        )
        label.pack(pady=5)

        label_frame = ttk.LabelFrame(frame, text='Documentation')
        label_frame.pack(fill='x', pady=10)
        label = ttk.Label(
            label_frame,
            text='For documentation, tutorials & support visit planetmapper.readthedocs.io',
            justify='center',
            wraplength=wraplength,
        )
        label.pack(pady=2)
        self.add_tooltip(
            ttk.Button(
                label_frame,
                text='Open documentation',
                command=lambda: webbrowser.open('https://planetmapper.readthedocs.io'),
            ),
            'Open PlanetMapper documentation in your web browser (https://planetmapper.readthedocs.io)',
        ).pack(padx=5, pady=(2, 5), fill='x')

        label_frame = ttk.LabelFrame(frame, text='Paper')
        label_frame.pack(fill='x', pady=10)
        label = ttk.Label(
            label_frame,
            text='King et al., (2023). JOSS, 8(90), 5728, https://doi.org/10.21105/joss.05728',
            justify='center',
            wraplength=wraplength,
        )
        label.pack(pady=2)

        self.add_tooltip(
            ttk.Button(
                label_frame,
                text='Open paper',
                command=lambda: webbrowser.open(common.CITATION_DOI),
            ),
            f'Open the PlanetMapper paper in your web browser ({common.CITATION_DOI})',
        ).pack(padx=5, pady=2, fill='x')
        self.add_tooltip(
            ttk.Button(
                label_frame,
                text='Copy citation string',
                command=lambda: self.copy_to_clipboard(common.CITATION_STRING),
            ),
            'Copy full citation string to clipboard',
        ).pack(padx=5, pady=2, fill='x')
        self.add_tooltip(
            ttk.Button(
                label_frame,
                text='Copy citation BibTeX',
                command=lambda: self.copy_to_clipboard(common.CITATION_BIBTEX),
            ),
            'Copy citation BibTeX entry to clipboard',
        ).pack(padx=5, pady=(2, 5), fill='x')

        label_frame = ttk.LabelFrame(frame, text='Credits')
        label_frame.pack(fill='x', pady=10)
        label = ttk.Label(
            label_frame,
            text=(
                'PlanetMapper was developed and is maintained by Oliver King '
                'at the University of Leicester, UK.'
            ),
            justify='center',
            wraplength=wraplength,
        )
        label.pack(pady=5)
        messages = [
            (
                'PlanetMapper was developed with support from a European Research '
                'Council Consolidator Grant (under the European Union\'s Horizon 2020 '
                'research and innovation programme, grant agreement No 723890).'
            ),
            'PlanetMapper is licensed under the MIT License.',
        ]
        label = ttk.Label(
            label_frame,
            text='\n\n'.join(messages),
            justify='center',
            font=('TkDefaultFont', int(self._default_font_size * 0.8)),
            wraplength=wraplength,
        )
        label.pack(pady=5)

    def _get_wcs_offsets(self) -> tuple[float, float, float, float]:
        if 'wcs' not in self._observation_available_disc_finding_routines:
            return (np.nan, np.nan, np.nan, np.nan)
        disc_params = self.get_observation().get_disc_params()
        if (
            self._cached_wcs_offset_info is not None
            and self._cached_wcs_offset_info[0] == disc_params
        ):
            return self._cached_wcs_offset_info[1]
        (
            dra_arcsec,
            ddec_arcsec,
            dr,
            drotation,
        ) = self.get_observation()._get_wcs_offsets_for_arcsec(
            suppress_warnings=True, use_header_offsets=False
        )

        r0_wcs = self.get_observation().get_r0() - dr
        plate_scale_wcs = self.get_observation().target_diameter_arcsec / (2 * r0_wcs)
        plate_scale_disc = self.get_observation().get_plate_scale_arcsec()
        d_scale_arcsec = plate_scale_disc - plate_scale_wcs

        self._cached_wcs_offset_info = (
            disc_params,
            (dra_arcsec, ddec_arcsec, d_scale_arcsec, drotation),
        )
        return self._cached_wcs_offset_info[1]

    def _set_wcs_offsets(
        self,
        *,
        dra_arcsec: float | None = None,
        ddec_arcsec: float | None = None,
        d_scale_arcsec: float | None = None,
        drotation: float | None = None,
    ) -> None:
        if 'wcs' not in self._observation_available_disc_finding_routines:
            return
        observation = self.get_observation()
        x0_wcs, y0_wcs, r0_wcs, rotation_wcs = observation._get_disc_params_from_wcs(
            suppress_warnings=True, use_header_offsets=False
        )
        if dra_arcsec is not None or ddec_arcsec is not None:
            dra_arcsec = (
                self._get_wcs_offsets()[0] if dra_arcsec is None else dra_arcsec
            )
            ddec_arcsec = (
                self._get_wcs_offsets()[1] if ddec_arcsec is None else ddec_arcsec
            )
            ra0, dec0 = observation.xy2radec(x0_wcs, y0_wcs)
            x0, y0 = observation.radec2xy(
                ra0 + dra_arcsec / 3600, dec0 + ddec_arcsec / 3600
            )
            observation.set_disc_params(x0=x0, y0=y0)
        if d_scale_arcsec is not None:
            plate_scale_wcs = observation.target_diameter_arcsec / (2 * r0_wcs)
            observation.set_plate_scale_arcsec(plate_scale_wcs + d_scale_arcsec)
        if drotation is not None:
            observation.set_disc_params(rotation=rotation_wcs + drotation)

    def build_wcs_offset_section(self, label_frame: ttk.LabelFrame) -> None:
        container_frame = ttk.Frame(label_frame)
        container_frame.pack(fill='x')
        self.add_tooltip(
            container_frame,
            'Differences between disc parameters and WCS data '
            '(useful for navigating multiple observations with systematic pointing errors)',
        )

        label = EnableableLabel(
            container_frame, text='Offsets between disc and WCS data:'
        )
        label.pack(pady=(5, 0))
        self._wcs_entries_to_enable.append(label)

        frame = ttk.Frame(container_frame)
        frame.pack(pady=2)

        entries: list[tuple[SetterKey, str, float | None, tuple[SetterKey, ...]]] = [
            (
                'wcs_offset_ra',
                'RA (arcsec)',
                1e-5,
                (
                    'wcs_offset_rotation',
                    'wcs_offset_scale',
                ),
            ),
            (
                'wcs_offset_dec',
                'Dec (arcsec)',
                1e-5,
                (
                    'wcs_offset_rotation',
                    'wcs_offset_scale',
                ),
            ),
            (
                'wcs_offset_rotation',
                'Rotation (°)',
                1e-5,
                (
                    'wcs_offset_ra',
                    'wcs_offset_dec',
                    'wcs_offset_scale',
                ),
            ),
            (
                'wcs_offset_scale',
                'Scale (arcsec/pixel)',
                1e-9,
                (
                    'wcs_offset_ra',
                    'wcs_offset_dec',
                    'wcs_offset_rotation',
                ),
            ),
        ]

        for key, label, zero_threshold, default_callbacks in entries:
            ne = NumericEntry(
                self,
                frame,
                key,
                label=label,
                value_fmt='+.6g',
                add_callbacks=[
                    'x0',
                    'y0',
                    'r0',
                    'rotation',
                    'plate_scale_arcsec',
                    'plate_scale_km',
                ],
                default_callbacks_to_add=default_callbacks,
                zero_threshold=zero_threshold,
            )
            self._wcs_entries_to_enable.append(ne)

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

    def enable_disc_finding_buttons(self) -> None:
        for key, buttons in self._disc_finding_enableable_buttons.items():
            state = (
                'normal'
                if key in self._observation_available_disc_finding_routines
                else 'disable'
            )
            for button in buttons:
                button.configure(state=state)

        enable_wcs_entries = 'wcs' in self._observation_available_disc_finding_routines
        for entry in self._wcs_entries_to_enable:
            entry.set_enabled(enable_wcs_entries)

    def build_help_hint(self) -> None:
        frame = ttk.Frame(self.hint_frame)
        frame.pack(fill='x', padx=5, pady=1)
        self.help_hint = ttk.Label(frame, text=DEFAULT_HINT)
        self.help_hint.pack(side='left')
        self.help_hint.bind('<Enter>', lambda e: self.reset_help_hint(hover=True))
        self.help_hint.bind('<Leave>', lambda e: self.reset_help_hint())
        self.reset_help_hint()

    def set_help_hint(self, msg: str, *, color: str = 'black'):
        self.help_hint.configure(text=msg, foreground=color)

    def reset_help_hint(self, *, hover: bool = False):
        msg = self._observation_full_path if hover else self._observation_filename
        self.set_help_hint(msg, color='gray50')

    def add_tooltip(
        self, widget: Widget, msg: str, shortcut_fn: Callable | None = None, **kw
    ) -> Widget:
        if shortcut_fn is not None:
            keys = self.shortcuts.get(shortcut_fn, None)
            if keys is not None:
                key = keys[0]
                key = key.replace('<less>', '<').upper()
                if key[0] == '<' and key[-1] == '>' and len(key) > 2:
                    key = key[1:-1]
                msg = f'{msg} (keyboard shortcut: {key})'

        widget.bind('<Enter>', lambda e: self.set_help_hint(msg, **kw))
        widget.bind('<Leave>', lambda e: self.reset_help_hint())
        return widget

    # Coords
    def build_coords_tab(self):
        top_level_frame = ttk.Frame(self.notebook)
        top_level_frame.pack()
        self.notebook.add(top_level_frame, text='Coords')

        frame = ttk.Frame(top_level_frame)
        frame.pack(padx=5, fill='x')
        self.coords_tab_labels: dict[str, ttk.Label] = {}
        # {group_name: [(key, label, tooltip)]}
        self.coords_labels: dict[str, list[tuple[str, str, str | None]]] = {
            'Pixel coordinates': [
                ('x', 'X', None),
                ('y', 'Y', None),
            ],
            'Celestial coordinates': [
                ('ra', 'Right ascension', None),
                ('dec', 'Declination', None),
            ],
            'Planetographic coordinates': [
                ('lon', 'Longitude', None),
                ('lat', 'Latitude', None),
            ],
            'Planetocentric coordinates': [
                ('lon_centric', 'Longitude', None),
                ('lat_centric', 'Latitude', None),
            ],
            'Illumination angles': [
                ('phase', 'Phase', None),
                ('incidence', 'Incidence', None),
                ('emission', 'Emission', None),
                ('azimuth', 'Azimuth', None),
            ],
            'Distances': [
                (
                    'limb_distance',
                    'Height above limb',
                    'Height above the target\'s limb, negative values are on the disc',
                ),
                (
                    'ring_radius',
                    'Ring plane radius',
                    'Distance from the target\'s centre in the target\'s equatorial (i.e. ring) plane',
                ),
            ],
        }
        for name, part_labels in self.coords_labels.items():
            label_frame = ttk.LabelFrame(frame, text=name)
            label_frame.pack(fill='x', pady=4)
            for col in range(2):
                label_frame.grid_columnconfigure(col, weight=1, uniform='a')
            for row, (key, label, tooltip) in enumerate(part_labels):
                label = label + ':'
                l1 = ttk.Label(label_frame, text=label)
                l1.grid(row=row, column=0, sticky='e', pady=1, padx=2)

                l2 = ttk.Label(label_frame, text='')
                l2.grid(row=row, column=1, sticky='w', pady=1, padx=2)

                self.coords_tab_labels[key] = l2

                if tooltip is not None:
                    self.add_tooltip(l1, tooltip)
                    self.add_tooltip(l2, tooltip)

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=5)

        self.coords_copy_formatted_button = self.add_tooltip(
            ttk.Button(
                button_frame,
                text='Copy formatted values',
                command=self.copy_formatted_coord_values,
                state='disable',
            ),
            'Copy formatted (i.e. human readable) coordinate values',
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

        messages = [
            'Click on the plot to get coordinates.',
            'Right click on the plot to clear.',
            'Note that most of these values change when you adjust the disc position.',
        ]
        ttk.Label(
            top_level_frame,
            text='\n'.join(messages),
            justify='center',
            foreground='gray50',
            wraplength=self.CONTROLS_WIDTH - 10,
        ).pack(padx=5, pady=2)

    def get_click_coords(self) -> dict[str, float]:
        if self.last_click_location is None:
            return {}
        out: dict[str, float] = {}
        observation = self.get_observation()
        x, y = self.last_click_location
        ra, dec = observation.xy2radec(x, y)
        out['x'] = x
        out['y'] = y
        out['ra'], out['dec'] = observation.xy2radec(x, y)

        _, _, out['limb_distance'] = observation.limb_coordinates_from_radec(ra, dec)
        ring_radius, _, _ = observation.ring_plane_coordinates(ra, dec)
        if math.isfinite(ring_radius):
            out['ring_radius'] = ring_radius

        try:
            targvec = observation._xy2targvec(x, y)
            out['lon'], out['lat'] = observation.targvec2lonlat(targvec)
            out['lon_centric'], out['lat_centric'] = (
                observation._targvec2lonlat_centric(targvec)
            )
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

        # format distance with thin space for thousands separator
        distance_formatter = lambda x: f'{x:_.0f} km'.replace('_', '\u2009')
        out['limb_distance'] = distance_formatter(coords['limb_distance'])
        try:
            out['ring_radius'] = distance_formatter(coords['ring_radius'])
        except KeyError:
            pass

        try:
            # Use targvec for a bit more speed here
            lon, lat = coords['lon'], coords['lat']
            ew = observation.positive_longitude_direction
            ns = 'N' if lat >= 0 else 'S'
            out['lon'] = f'{lon:{fmt}}°{ew}'
            out['lat'] = f'{abs(lat):{fmt}}°{ns}'

            lon_centric, lat_centric = coords['lon_centric'], coords['lat_centric']
            ew = 'E'
            ns = 'N' if lat >= 0 else 'S'
            out['lon_centric'] = f'{lon_centric:{fmt}}°{ew}'
            out['lat_centric'] = f'{abs(lat_centric):{fmt}}°{ns}'

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
            for row, (key, label, tooltip) in enumerate(part_labels):
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
            lon, lat = coords['lon'], coords['lat']
            parts.extend(
                [
                    f'"lonlat": [{lon:{fmt}}, {lat:{fmt}}]',
                    f'"lonlat_centric": [{coords["lon_centric"]:{fmt}}, {coords["lat_centric"]:{fmt}}]',
                    f'"phase": {coords["phase"]:{fmt}}',
                    f'"incidence": {coords["incidence"]:{fmt}}',
                    f'"emission": {coords["emission"]:{fmt}}',
                    f'"azimuth": {coords["azimuth"]:{fmt}}',
                ]
            )
        except KeyError:
            pass  # Not on disc
        parts.append(f'"limb_distance": {coords["limb_distance"]:{fmt}}')
        try:
            parts.append(f'"ring_radius": {coords["ring_radius"]:{fmt}}')
        except KeyError:
            pass
        return '{' + ', '.join(parts) + '}'

    def figure_click_callback(self, event: MouseEvent) -> None:
        if not event.inaxes or event.dblclick:
            return

        try:
            # Disable when panning/zooming
            if self.toolbar.mode._navigate_mode is not None:
                return
        # pylint: disable-next=bare-except
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

    def copy_to_clipboard(self, s: str | None) -> None:
        if s is None:
            s = ''
        self.root.clipboard_clear()
        self.root.clipboard_append(s)

    # Plotting
    def update_plot(self, print_coords: bool = False) -> None:
        self.get_observation().update_transform()
        self.canvas.draw()
        self.update_coords(print_coords=print_coords)

    def update_only_image(self) -> None:
        self.replot_image()
        self.canvas.draw()

    def update_plot_transforms(self) -> None:
        # Use func to convert radec -> angular, then matplotlib transform to do
        # angular -> xy. This two step approach should help reduce distortion at high
        # declinations or for large diameter targets.
        # https://github.com/ortk95/planetmapper/issues/456
        self.plot_coordinate_func = self.get_observation().radec2angular
        self.transform = (
            self.get_observation().matplotlib_angular2xy_transform() + self.ax.transData
        )

    def build_plot(self) -> None:
        # Use Figure rather than plt.figure to avoid segmentation fault when running
        # from tkinter GUI (issue #258)
        self.fig = Figure()
        self.ax = self.fig.add_axes([0.06, 0.03, 0.93, 0.96])
        self.update_plot_transforms()

        bg_color = '#eeeeee'
        toolbar_frame = tk.Frame(self.plot_frame, background=bg_color)
        toolbar_frame.pack(side='bottom', fill='x')
        tk.Label(toolbar_frame, text='\N{NO-BREAK SPACE}', background=bg_color).pack(
            side='left'
        )
        self.canvas_frame = tk.Frame(self.plot_frame, background='white')
        self.canvas_frame.pack(side='top', fill='both', expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.toolbar = CustomNavigationToolbar(
            self.canvas,
            toolbar_frame,
            pack_toolbar=False,
            gui=self,
            bg_color=bg_color,
        )
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.fig.canvas.callbacks.connect(
            'button_press_event', self.figure_click_callback
        )

        self.add_tooltip(
            self.canvas.get_tk_widget(),
            'Customise plot in the "Settings" tab and click on the plot to get values in the "Coords" tab',
        )

        self.replot_all()
        self.format_plot()

    def rebuild_plot(self) -> None:
        self.update_plot_transforms()
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
        vmin = self.plot_settings['_'].setdefault('image_vmin', 0)
        vmax = self.plot_settings['_'].setdefault('image_vmax', 1)
        limit_type = self.plot_settings['_'].setdefault('image_limit_type', 'absolute')

        with utils.ignore_warnings('All-NaN slice encountered'):
            image = self.image_modes[mode][0]()
            if limit_type == 'percentile':
                vmin = np.nanpercentile(image, vmin)
                vmax = np.nanpercentile(image, vmax)

        self.plot_handles['image'].append(
            self.ax.imshow(
                image,
                origin='lower',
                vmin=vmin,
                vmax=vmax,
                **self.plot_settings['image'],
            )
        )

    def replot_limb(self):
        self.remove_artists('limb')
        self.remove_artists('limb_illuminated')
        self.plot_handles['limb'].extend(
            self.ax.plot(
                *self.plot_coordinate_func(*self.get_observation().limb_radec()),
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
        self.plot_handles['limb_illuminated'].extend(
            self.ax.plot(
                *self.plot_coordinate_func(ra_day, dec_day),
                transform=self.transform,
                **self.plot_settings['limb_illuminated'],
            )
        )

    def replot_terminator(self):
        self.remove_artists('terminator')
        self.plot_handles['terminator'].extend(
            self.ax.plot(
                *self.plot_coordinate_func(*self.get_observation().terminator_radec()),
                transform=self.transform,
                **self.plot_settings['terminator'],
            )
        )

    def replot_poles(self):
        self.remove_artists('pole')
        for lon, lat, s in self.get_observation().get_poles_to_plot():
            ra, dec = self.get_observation().lonlat2radec(lon, lat)
            self.plot_handles['pole'].append(
                self.ax.add_artist(
                    OutlinedText(
                        *self.plot_coordinate_func(ra, dec),
                        s,
                        transform=self.transform,
                        ha='center',
                        va='center',
                        weight='bold',
                        size='small',
                        clip_on=True,
                        **self.plot_settings['pole'],
                    )
                )
            )

    def replot_grid(self) -> None:
        self.remove_artists('grid')
        interval = self.plot_settings['_'].setdefault('grid_interval', 30)
        lat_limit = self.plot_settings['_'].setdefault('grid_lat_limit', 90)
        planetocentric = self.plot_settings['_'].setdefault(
            'grid_planetocentric', False
        )
        for ra, dec in self.get_observation().visible_lonlat_grid_radec(
            interval, planetocentric=planetocentric, lat_limit=lat_limit
        ):
            self.plot_handles['grid'].extend(
                self.ax.plot(
                    *self.plot_coordinate_func(ra, dec),
                    transform=self.transform,
                    **self.plot_settings['grid'],
                )
            )

    def replot_coordinates_lonlat(self) -> None:
        self.remove_artists('coordinate_of_interest_lonlat')
        for lon, lat in self.get_observation().coordinates_of_interest_lonlat:
            if self.get_observation().test_if_lonlat_visible(lon, lat):
                ra, dec = self.get_observation().lonlat2radec(lon, lat)
                self.plot_handles['coordinate_of_interest_lonlat'].append(
                    self.ax.scatter(
                        *self.plot_coordinate_func(ra, dec),
                        transform=self.transform,
                        **self.plot_settings['coordinate_of_interest_lonlat'],
                    )
                )

    def replot_coordinates_radec(self) -> None:
        self.remove_artists('coordinate_of_interest_radec')
        for ra, dec in self.get_observation().coordinates_of_interest_radec:
            self.plot_handles['coordinate_of_interest_radec'].append(
                self.ax.scatter(
                    *self.plot_coordinate_func(ra, dec),
                    transform=self.transform,
                    **self.plot_settings['coordinate_of_interest_radec'],
                )
            )

    def replot_rings(self) -> None:
        self.remove_artists('ring')
        for radius in self.get_observation().ring_radii:
            ra, dec = self.get_observation().ring_radec(radius)
            self.plot_handles['ring'].extend(
                self.ax.plot(
                    *self.plot_coordinate_func(ra, dec),
                    transform=self.transform,
                    **self.plot_settings['ring'],
                )
            )

    def replot_other_bodies(self) -> None:
        self.remove_artists('other_body_of_interest_label')
        self.remove_artists('other_body_of_interest_marker')
        for body in self.get_observation().other_bodies_of_interest:
            ra = body.target_ra
            dec = body.target_dec
            label = body.target
            hidden = not self.get_observation().test_if_other_body_visible(body)
            if hidden:
                label = f'({label})'
            self.plot_handles['other_body_of_interest_label'].append(
                self.ax.text(
                    *self.plot_coordinate_func(ra, dec),
                    label + '\n',
                    size='small',
                    ha='center',
                    va='center',
                    transform=self.transform,
                    clip_on=True,
                    **self.plot_settings['other_body_of_interest_label'],
                )
            )
            self.plot_handles['other_body_of_interest_marker'].append(
                self.ax.scatter(
                    *self.plot_coordinate_func(ra, dec),
                    transform=self.transform,
                    alpha=0.5 if hidden else 1,
                    **self.plot_settings['other_body_of_interest_marker'],
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

    def remove_artists(self, key: PlotKey) -> None:
        while self.plot_handles[key]:
            self.plot_handles[key].pop().remove()

    # Delayed actions
    def add_delayed_action(self, name: str, ms: int, func: Callable[[], Any]) -> None:
        self.cancel_delayed_action(name)
        self.delayed_actions[name] = (
            func,
            self.root.after(ms, lambda: self.run_delayed_action(name)),
        )

    def cancel_delayed_action(self, name: str) -> None:
        action = self.delayed_actions.pop(name, None)
        if action:
            self.root.after_cancel(action[1])

    def run_delayed_action(self, name: str) -> None:
        func, _ = self.delayed_actions.pop(name)
        func()

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
                if platform.system() == 'Darwin' and 'Control' in event:
                    # On MacOS, bind Command as well as Control
                    self.root.bind(event.replace('Control', 'Command'), handler)

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

    def set_value(self, key: SetterKey, value: float, update_plot: bool = True) -> None:
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

    def maybe_replace_string_with_x11_bugfix(self, s: str) -> str:
        # X11 font bug https://github.com/ortk95/planetmapper/issues/145
        if USE_X11_FONT_BUGFIX:
            s = s.translate(X11_FONT_BUGRIX_TRANSLATIONS)
        return s

    def display_header(self) -> None:
        HeaderDisplay(self)

    def add_popup(self, popup: 'Popup') -> None:
        self._popups.append(popup)

    def remove_popup(self, popup: 'Popup') -> None:
        self._popups.remove(popup)

    def get_popups(self) -> list['Popup']:
        return self._popups

    def close_all_popups(self, *, keep_open: list['Popup'] | None = None) -> None:
        if keep_open is None:
            keep_open = []
        # use list() as popups will be removed from the list during iteration
        for popup in list(self.get_popups()):
            if popup in keep_open:
                continue
            popup.close_window()


class PopupAlreadyOpenError(Exception):
    pass


class Popup:
    def __init__(
        self,
        gui: GUI,
        *,
        bind_escape: bool = True,
        create_window_immediately: bool = True,
        minimum_size: tuple[int, int] | None = (200, 200),
    ) -> None:
        self.gui = gui
        self.bind_escape = bind_escape
        self.minimum_size = minimum_size
        if create_window_immediately:
            self.create_window()

    def get_popup_id(self) -> str:
        """
        ID used to determine if popup already exists when opening.

        Override in subclasses if multiple instances of the same class can exist at the
        same time.
        """
        return self.__class__.__name__

    def create_window(self) -> None:
        if self.defer_to_any_already_open_popup():
            raise PopupAlreadyOpenError
        self.gui.add_popup(self)

        try:
            self.window = tk.Toplevel(self.gui.root)
        except AttributeError:
            # GUI hasn't been created yet, so create a new window
            self.window = tk.Tk()
            self.gui.configure_style(self.window)

        self.window.protocol('WM_DELETE_WINDOW', self.close_window)
        if self.bind_escape:
            self.window.bind('<Escape>', self.close_window)

        if self.minimum_size is not None:
            self.window.minsize(*self.minimum_size)

    def close_window(self, *_) -> None:
        self.window.destroy()
        self.gui.remove_popup(self)

    def maybe_get_already_open_popup(self) -> 'None | Popup':
        for other in self.gui.get_popups():
            if other.get_popup_id() == self.get_popup_id():
                return other
        return None

    def defer_to_any_already_open_popup(self) -> bool:
        already_open = self.maybe_get_already_open_popup()
        if already_open:
            already_open.give_focus()
            return True
        return False

    def give_focus(self) -> None:
        """Bring popup to front and give focus"""
        self.window.lift()
        self.window.focus()

    # utility methods
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
        try:
            super().__init__(gui)
        except PopupAlreadyOpenError:
            return
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
            self.window.title('PlanetMapper')
            geometry = self.gui.DEFAULT_GEOMETRY
        else:
            self.window.title('Observation settings')
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

        window_frame = ttk.Frame(self.window)
        window_frame.pack(expand=True, fill='both')

        self.menu_frame = ttk.Frame(window_frame)
        self.menu_frame.pack(side='top', padx=10, pady=10, fill='x')

        self.heading_frame = ttk.Frame(self.menu_frame)
        self.heading_frame.pack(fill='x')

        self.grid_frame = ttk.Frame(self.menu_frame)
        self.grid_frame.pack(fill='x')

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
                with fits.open(path) as hdul:  # type: ignore
                    # pylint: disable-next=no-member
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
            initialdir=self.get_open_initialdir(),
        )
        if path:
            self.stringvars['path'].set(str(path))

    def get_open_initialdir(self) -> str:
        path = self.stringvars['path'].get()
        path = os.path.expandvars(os.path.expanduser(path))
        if len(path.strip()) == 0:
            return os.getcwd()
        for _ in range(32):  # Limit to 32 iterations to avoid infinite loop
            if os.path.isdir(path):
                return path
            dirname = os.path.dirname(path)
            if dirname == path:
                break
            path = dirname
        return os.getcwd()

    def click_ok(self) -> None:
        if self.apply_changes():
            self.close_window()

    def click_apply(self) -> None:
        self.apply_changes()

    def apply_changes(self) -> bool:
        ObservationKwargs = TypedDict(
            'ObservationKwargs',
            {'path': str, 'target': str, 'utc': str | float, 'observer': str},
        )
        observation_kwargs: ObservationKwargs = {
            k: v.get() for k, v in self.stringvars.items()
        }  #  type: ignore
        for k, v in observation_kwargs.items():
            if isinstance(v, str) and len(v.strip()) == 0:
                tkinter.messagebox.showwarning(
                    title=f'Error parsing {k}', message=f'{k!r} must not be empty'
                )
                return False

        string = self.kernel_txt.get('1.0', 'end')
        kernels = [k.strip() for k in string.splitlines()]
        base.load_kernels(*kernels, clear_before=True)

        sb = base.SpiceBase(auto_load_kernels=False)
        try:
            target = sb.standardise_body_name(observation_kwargs['target'])
            spice.bods2c(target)
        # pylint: disable-next=broad-except
        except Exception as e:
            self.show_spice_warning(title='Error parsing target', exception=e)
            return False

        try:
            observer = sb.standardise_body_name(observation_kwargs['observer'])
            spice.bods2c(observer)
            # pylint: disable-next=broad-except
        except Exception as e:
            self.show_spice_warning(title='Error parsing observer', exception=e)
            return False

        if target == observer:
            tkinter.messagebox.showwarning(
                title='Target and observer identical',
                message='\n'.join(
                    [
                        'Target and observer must correspond to different bodies',
                        f'Target: {target!r}',
                        f'Observer: {observer!r}',
                    ]
                ),
            )
            return False

        try:
            observation_kwargs['utc'] = float(
                observation_kwargs['utc']
            )  #  type: ignore
        except ValueError:
            try:
                spice.utc2et(observation_kwargs['utc'])  #  type: ignore
            # pylint: disable-next=broad-except
            except Exception as e:
                self.show_spice_warning(title='Error parsing date', exception=e)
                return False
        try:
            observation = Observation(**observation_kwargs, auto_load_kernels=False)
        # pylint: disable-next=broad-except
        except Exception as e:
            traceback.print_exc()
            tkinter.messagebox.showwarning(
                title='Error processing inputs',
                message=f'Error: {e}' + '\n\nSee terminal for more details',
            )
            return False
        self.gui.set_observation(observation)
        self.gui.kernels = kernels
        self.gui.close_all_popups(keep_open=[self])
        return True

    def show_spice_warning(self, *, title: str, exception: Exception) -> None:
        message = base._SPICE_ERROR_HELP_TEXT
        # MacOS doesn't show the title, so add it to the start of the message instead
        if platform.system() == 'Darwin' and title != '':
            message = f'{title}\n\n{message}'
            title = ''
        tkinter.messagebox.showwarning(
            master=self.window,
            parent=self.window,
            title=title,
            message=message,
            detail=f'{type(exception).__name__}\n{exception}',
        )

    def click_cancel(self) -> None:
        self.close_window()

    def close_window(self, *_) -> None:
        super().close_window()
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
        try:
            super().__init__(gui)
        except PopupAlreadyOpenError:
            return

        self.make_widget()
        self.make_menu()
        self.save_nav_toggle()
        self.save_map_toggle()

    def make_widget(self) -> None:
        self.window.title('Save observation')
        self.window.grab_set()
        self.window.transient(self.gui.root)
        # Use grab_set and transient for saving popups to help ensure that the state
        # does not change while the user is in the progress of saving data

        x, y = (int(s) for s in self.gui.root.geometry().split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(
                sz='600x575',
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
        self.menu_frame.pack(side='top', padx=10, pady=10, fill='both', expand=True)

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

        self.map_projection = tk.StringVar(value='rectangular')
        self.map_degree_interval = tk.StringVar(value=str(1))
        self.map_lon = tk.StringVar(value=str(0))
        self.map_lat = tk.StringVar(value=str(0))
        self.map_output_size = tk.StringVar(value=str(100))
        self.map_interpolation = tk.StringVar(value='linear')

        self.keep_open = tk.IntVar(value=0)

        self.save_nav.trace_add('write', self.save_nav_toggle)
        self.save_map.trace_add('write', self.save_map_toggle)
        self.map_projection.trace_add('write', self.save_map_toggle)

        self.nav_widgets: list[tk.Widget] = []
        self.map_widgets: list[tk.Widget] = []
        self.map_rect_widgets: list[tk.Widget] = []
        self.map_ortho_widgets: list[tk.Widget] = []

        self.grid_frame.grid_columnconfigure(1, weight=1)

        LabelKwargs = TypedDict('LabelKwargs', {'sticky': str, 'pady': int})
        label_kwargs = LabelKwargs(sticky='w', pady=5)

        # Navigated
        ttk.Checkbutton(
            self.grid_frame, text='Save navigated observation', variable=self.save_nav
        ).grid(row=0, column=1, columnspan=2, sticky='ew')

        ttk.Label(self.grid_frame, text='Path: ').grid(row=1, column=0, **label_kwargs)
        w = ttk.Entry(self.grid_frame, textvariable=self.path_nav)
        w.grid(row=1, column=1, sticky='ew')
        self.nav_widgets.append(w)
        w = ttk.Button(self.grid_frame, text='...', width=3, command=self.get_path_nav)
        w.grid(row=1, column=2)
        self.nav_widgets.append(w)

        ttk.Label(self.grid_frame, text=' ').grid(row=2, column=0, **label_kwargs)

        # Mapped
        ttk.Checkbutton(
            self.grid_frame, text='Save mapped observation', variable=self.save_map
        ).grid(row=3, column=1, columnspan=2, sticky='ew')

        ttk.Label(self.grid_frame, text='Path: ').grid(row=4, column=0, **label_kwargs)
        w = ttk.Entry(self.grid_frame, textvariable=self.path_map)
        w.grid(row=4, column=1, sticky='ew')
        self.map_widgets.append(w)
        w = ttk.Button(self.grid_frame, text='...', width=3, command=self.get_path_map)
        w.grid(row=4, column=2, sticky='w')
        self.map_widgets.append(w)

        self.map_option_grid = ttk.Frame(self.grid_frame)
        self.map_option_grid.grid(row=5, column=0, columnspan=3, sticky='nsew')

        for col in [1, 3, 5]:
            self.map_option_grid.grid_columnconfigure(col, weight=1)

        label_kwargs = LabelKwargs(sticky='w', pady=2)

        ttk.Label(self.map_option_grid, text='Interpolation: ').grid(
            row=0, column=0, **label_kwargs
        )
        w = ttk.Combobox(
            self.map_option_grid,
            textvariable=self.map_interpolation,
            width=15,
            values=MAP_INTERPOLATIONS,
            state='readonly',
        )
        w.grid(row=0, column=1, columnspan=5, sticky='w')
        self.map_widgets.append(w)

        ttk.Label(self.map_option_grid, text='Projection: ').grid(
            row=1, column=0, **label_kwargs
        )
        w = ttk.Combobox(
            self.map_option_grid,
            textvariable=self.map_projection,
            width=15,
            values=MAP_PROJECTIONS,
            state='readonly',
        )
        w.grid(row=1, column=1, columnspan=5, sticky='w')
        self.map_widgets.append(w)

        # Projection options
        width = 10
        ttk.Label(self.map_option_grid, text='Degree interval: ').grid(
            row=2, column=0, **label_kwargs
        )
        w = ttk.Entry(
            self.map_option_grid, textvariable=self.map_degree_interval, width=width
        )
        w.grid(row=2, column=1, sticky='w')
        self.map_rect_widgets.append(w)

        ttk.Label(self.map_option_grid, text='Output size: ').grid(
            row=3, column=0, **label_kwargs
        )
        w = ttk.Entry(
            self.map_option_grid, textvariable=self.map_output_size, width=width
        )
        w.grid(row=3, column=1, sticky='w')
        self.map_ortho_widgets.append(w)

        ttk.Label(self.map_option_grid, text='Longitude: ').grid(
            row=3, column=2, **label_kwargs
        )
        w = ttk.Entry(self.map_option_grid, textvariable=self.map_lon, width=width)
        w.grid(row=3, column=3, sticky='w')
        self.map_ortho_widgets.append(w)

        ttk.Label(self.map_option_grid, text='Latitude: ').grid(
            row=3, column=4, **label_kwargs
        )
        w = ttk.Entry(self.map_option_grid, textvariable=self.map_lat, width=width)
        w.grid(row=3, column=5, sticky='w')
        self.map_ortho_widgets.append(w)

        # Backplanes to include
        ttk.Label(self.menu_frame, text='\n').pack(fill='x')  # Spacer
        self.backplanes_to_save_dict: dict[str, tk.IntVar] = {}
        self.backplanes_label_frame = ttk.Frame(self.menu_frame)
        self.backplanes_label_frame.pack(fill='x', pady=2)
        ttk.Label(
            self.backplanes_label_frame,
            text='Backplanes to include: ',
            justify='left',
        ).pack(side='left', fill='y')

        self.backplanes_all_button = ttk.Button(
            self.backplanes_label_frame,
            text='Select all',
            command=lambda: [
                var.set(1) for var in self.backplanes_to_save_dict.values()
            ],
            padding=0,
        )
        self.backplanes_none_button = ttk.Button(
            self.backplanes_label_frame,
            text='Select none',
            command=lambda: [
                var.set(0) for var in self.backplanes_to_save_dict.values()
            ],
            padding=0,
        )
        self.backplanes_all_button.pack(side='left', fill='y')
        self.backplanes_none_button.pack(side='left', fill='y')

        self.backplane_grid = ttk.Frame(self.menu_frame)
        self.backplane_grid.pack(fill='x')
        ncols = 4
        for col in range(ncols):
            self.backplane_grid.grid_columnconfigure(col, weight=1)
        for i, backplane in enumerate(self.gui.get_observation().backplanes.keys()):
            row, col = divmod(i, ncols)
            var = tk.IntVar(value=1)
            self.backplanes_to_save_dict[backplane] = var
            cb = ttk.Checkbutton(
                self.backplane_grid,
                text=backplane,
                style='Small.TCheckbutton',
                variable=var,
            )
            cb.grid(row=row, column=col, sticky='ew', padx=5, pady=0)

        # Footer
        menu_footer_frame = ttk.Frame(self.menu_frame)
        menu_footer_frame.pack(side='bottom', fill='x')

        message = '\n'.join(
            [
                'Click SAVE below to save the requested files',
                'For larger files, backplane generation, mapping, and saving can take ~1 minute',
                '',
            ]
        )
        ttk.Label(menu_footer_frame, text=message, justify='center').pack()
        ttk.Checkbutton(
            menu_footer_frame,
            text='Keep this popup open after saving files',
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
        self.toggle(bool(self.save_nav.get()), self.nav_widgets)

    def save_map_toggle(self, *_) -> None:
        map_enabled = bool(self.save_map.get())
        projection_type = self.map_projection.get()
        self.toggle(map_enabled, self.map_widgets)
        self.toggle(
            map_enabled and projection_type in {'rectangular'}, self.map_rect_widgets
        )
        self.toggle(
            map_enabled
            and projection_type
            in {'orthographic', 'azimuthal', 'azimuthal equal area'},
            self.map_ortho_widgets,
        )

    def toggle(self, enabled: bool, widgets: list[tk.Widget]) -> None:
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
        self.try_run_save()

    def click_cancel(self) -> None:
        self.close_window()

    def try_run_save(self) -> None:
        save_nav = bool(self.save_nav.get())
        save_map = bool(self.save_map.get())
        map_kw: MapKwargs = {}

        path_map = self.path_map.get().strip()
        path_nav = self.path_nav.get().strip()

        keep_open = bool(self.keep_open.get())

        interpolation = 'linear'

        if (save_nav and len(path_nav) == 0) or (save_map and len(path_map) == 0):
            tkinter.messagebox.showwarning(
                title='Error saving file',
                message='File paths must not be empty',
            )
            return

        try:
            if save_map:
                interpolation = self.map_interpolation.get()
                map_kw['projection'] = self.map_projection.get()
                if map_kw['projection'] in {'rectangular'}:
                    map_kw['degree_interval'] = self.get_float(
                        self.map_degree_interval,
                        name='degree interval',
                        positive=True,
                        finite=True,
                    )
                if map_kw['projection'] in {'orthographic', 'azimuthal'}:
                    map_kw['size'] = self.get_int(
                        self.map_output_size,
                        name='output size',
                        positive=True,
                    )
                    map_kw['lon'] = self.get_float(
                        self.map_lon,
                        name='longitude',
                        positive=False,
                        finite=True,
                    )
                    map_kw['lat'] = self.get_float(
                        self.map_lat,
                        name='latitude',
                        positive=False,
                        finite=True,
                    )
        except ValueError:
            return

        backplanes_to_save = {
            k for k, v in self.backplanes_to_save_dict.items() if v.get()
        }

        # If we get to this point, everything should (hopefully) be working

        saving_process = SavingProgress(
            self,
            save_nav=save_nav,
            path_nav=path_nav,
            save_map=save_map,
            path_map=path_map,
            interpolation=interpolation,
            map_kw=map_kw,
            keep_open=keep_open,
            backplanes_to_save=backplanes_to_save,
        )
        try:
            saving_process.run_save()
        # pylint: disable-next=broad-except
        except Exception as e:
            traceback.print_exc()
            tkinter.messagebox.showwarning(
                title='Error saving files',
                message=f'Error: {e}' + '\n\nSee terminal for more details',
            )
            return
        finally:
            self.gui.get_observation()._remove_progress_hook()

        if saving_process.is_save_success:
            self.gui.set_help_hint(
                'File{s} saved successfully'.format(
                    s='s' if save_nav and save_map else ''
                ),
                color='green',
            )


class SavingProgress(Popup):
    def __init__(
        self,
        parent: SaveObservation,
        save_nav: bool,
        path_nav: str,
        save_map: bool,
        path_map: str,
        interpolation: str,
        map_kw: MapKwargs,
        keep_open: bool,
        backplanes_to_save: set[str],
    ):
        self.parent = parent
        self.parent.saving_progress_window = self

        self.save_nav = save_nav
        self.path_nav = path_nav
        self.save_map = save_map
        self.path_map = path_map
        self.interpolation = interpolation
        self.map_kw = map_kw
        self.backplanes_to_save = backplanes_to_save

        self.keep_open = keep_open

        self.should_cancel_save = False
        self.is_running_save = False
        self.is_save_success = False

        try:
            super().__init__(self.parent.gui, bind_escape=False)
        except PopupAlreadyOpenError:
            return

        self.make_window()
        self.make_required_widgets()

    def make_window(self) -> None:
        self.window.title('Saving files...')
        self.window.grab_set()
        self.window.transient(self.parent.window)

        x, y = (int(s) for s in self.parent.window.geometry().split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(sz='500x175', x=x + 50, y=y + 50)
        )

        self.frame = ttk.Frame(self.window)
        self.frame.pack(expand=True, fill='both')

        self.window.bind('<Escape>', self.press_escape)

    def make_required_widgets(self) -> None:
        if self.save_nav:
            self.nav_widgets = self.make_widgets('Saving navigated observation...')
        if self.save_map:
            self.map_widgets = self.make_widgets('Saving mapped observation...')
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(padx=10, pady=10, fill='x')
        self.close_button = ttk.Button(
            button_frame,
            command=self.click_close,
            text='Close',
            width=10,
        )
        self.cancel_button = ttk.Button(
            button_frame,
            command=self.click_cancel,
            text='Cancel saving',
            width=10,
        )
        self.cancel_button.pack()

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
        SaveKwargs = TypedDict(
            'SaveKwargs',
            {'show_progress': bool, 'print_info': bool, 'backplanes_to_save': set[str]},
        )
        save_kwargs = SaveKwargs(
            show_progress=False,
            print_info=True,
            backplanes_to_save=self.backplanes_to_save,
        )
        observation = self.parent.gui.get_observation()
        self.is_running_save = True
        save_nav_done = False
        save_map_done = False
        try:
            if self.save_nav:
                observation._set_progress_hook(
                    SaveNavProgressHookGUI(**self.nav_widgets, parent=self)
                )
                observation.save_observation(self.path_nav, **save_kwargs)
                observation._remove_progress_hook()
                save_nav_done = True
            if self.save_map:
                n_wavelengths = len(self.parent.gui.get_observation().data)
                observation._set_progress_hook(
                    SaveMapProgressHookGUI(
                        n_wavelengths, **self.map_widgets, parent=self
                    )
                )
                observation.save_mapped_observation(
                    self.path_map,
                    interpolation=self.interpolation,  # type: ignore
                    **self.map_kw,
                    **save_kwargs,
                )
                observation._remove_progress_hook()
                save_map_done = True
            self.is_save_success = True
            self.window.title('Saving files complete')
        except CancelSave:
            print('Cancelled save')
            self.window.title('Cancelled saving files')
            if self.save_nav and not save_nav_done:
                self.nav_widgets['message'].configure(
                    text='Cancelled', foreground='red3'  # type: ignore
                )
            if self.save_map and not save_map_done:
                self.map_widgets['message'].configure(
                    text='Cancelled', foreground='red3'  # type: ignore
                )
        finally:
            self.is_running_save = False
            self.cancel_button.pack_forget()
            self.close_button.pack()

    def set_abort_trap(self) -> None:
        """Set the abort trap to cancel the save process."""
        self.should_cancel_save = True

    def press_escape(self, *_) -> None:
        if self.is_running_save:
            return  # only use ESC to close window, not cancel the save
        self.click_close()

    def click_cancel(self) -> None:
        self.set_abort_trap()

    def click_close(self) -> None:
        self.close_window()
        if not self.keep_open and self.is_save_success:
            self.parent.close_window()

    def close_window(self, *_) -> None:
        self.set_abort_trap()
        if self.is_running_save:
            return  # clicking close while saving should only cancel the save
        super().close_window()
        self.parent.gui.get_observation()._remove_progress_hook()
        self.parent.saving_progress_window = None


# Progress hooks
class CancelSave(Exception):
    pass


class SaveProgressHookGUI(progress._SaveProgressHook):
    def __init__(
        self,
        label: ttk.Label,
        bar: ttk.Progressbar,
        message: ttk.Label,
        *args,
        parent: SavingProgress,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label = label
        self.bar = bar
        self.message = message
        self.parent = parent

    def update_bar(self, progress_change: float) -> None:
        if self.progress_parts.get(self.default_key, 0) >= 1:
            self.bar['value'] = 100
            self.message.configure(text='Complete')
        else:
            self.bar['value'] = self.overall_progress * 100
            self.message.configure(text=format(self.overall_progress, '.0%'))
        # self.bar.update_idletasks()
        self.bar.update()

        if self.parent.should_cancel_save:
            raise CancelSave


class SaveNavProgressHookGUI(progress._SaveNavProgressHook, SaveProgressHookGUI):
    pass


class SaveMapProgressHookGUI(progress._SaveMapProgressHook, SaveProgressHookGUI):
    pass


# Header
class HeaderDisplay(Popup):
    def __init__(self, gui: GUI) -> None:
        try:
            super().__init__(gui)
        except PopupAlreadyOpenError:
            return
        self.make_widget()

    def make_widget(self) -> None:
        self.window.title('FITS Header')
        geometry = self.gui.root.geometry()

        x, y = (int(s) for s in geometry.split('+')[1:])
        self.window.geometry(
            '{sz}+{x:.0f}+{y:.0f}'.format(
                sz='650x800',
                x=x + 50,
                y=y + 50,
            )
        )

        self.window_frame = ttk.Frame(self.window)
        self.window_frame.pack(expand=True, fill='both')

        self.content_frame = ttk.Frame(self.window_frame)
        self.content_frame.pack(expand=True, fill='both')

        self.add_header_widget()

    def add_header_widget(self) -> None:
        self.header_txt = tkinter.scrolledtext.ScrolledText(self.content_frame)
        self.header_txt.pack(expand=True, fill='both')
        self.header_txt.insert('1.0', self.get_header_string())
        self.header_txt.configure(state='disabled')

    def get_header_string(self) -> str:
        header = self.gui.get_observation().header
        return header.tostring(sep='\n')

    def click_close(self) -> None:
        self.close_window()


# Artist settings popups
class ArtistSetting(Popup, ABC):
    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: PlotKey,
        label: str | None = None,
        hint: str | None = None,
        callbacks: list[Callable[[], None]] | None = None,
        row: int | None = None,
    ) -> None:
        self.parent = parent
        self.key: PlotKey = key
        self.gui = gui
        self._enable_callback = True
        if label is None:
            label = key
        self.label = label
        self.callbacks = callbacks
        if row is None:
            row = parent.grid_size()[1]

        super().__init__(gui, create_window_immediately=False)

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
        self.make_popup()

    def make_popup(self) -> None:
        try:
            self.create_window()
        except PopupAlreadyOpenError:
            return
        self.make_widget()
        self.make_menu()

    @abstractmethod
    def make_menu(self) -> None: ...

    @abstractmethod
    def apply_settings(self) -> bool: ...

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

    def get_popup_id(self) -> str:
        return f'{self.__class__.__name__}:{self.key}'

    def make_widget(self) -> None:
        self.window.title(self.label)

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

        window_frame = ttk.Frame(self.window)
        window_frame.pack(expand=True, fill='both')

        self.menu_frame = ttk.Frame(window_frame)
        self.menu_frame.pack(side='top', padx=10, pady=10)
        self.grid_frame = ttk.Frame(self.menu_frame)
        self.grid_frame.pack()

    def add_to_menu_grid(
        self, grid: list[tuple[tk.Widget, tk.Widget]], frame: ttk.Frame | None = None
    ) -> None:
        if frame is None:
            frame = self.grid_frame
        for label, widget in grid:
            row = frame.grid_size()[1]
            label.grid(row=row, column=0, sticky='w', pady=5)
            widget.grid(row=row, column=1, sticky='w')

    def click_ok(self) -> None:
        if self.apply_settings():
            self.run_callbacks()
            self.close_window()

    def click_apply(self) -> None:
        if self.apply_settings():
            self.run_callbacks()

    def click_cancel(self) -> None:
        self.close_window()

    def get_window_size(self) -> str:
        return '350x350'

    @property
    def label_wraplength(self) -> int:
        return 320


class PlotImageSetting(ArtistSetting):
    REPLOT_DELAY_MS = 100
    SLIDER_DELAY_MS = 10

    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: PlotKey,
        label: str | None = None,
        hint: str | None = None,
        callbacks: list[Callable[[], None]] | None = None,
        row: int | None = None,
    ):
        super().__init__(gui, parent, key, label, hint, callbacks, row)

        row = parent.grid_size()[1]

        self.tools_frame = ttk.Frame(parent)
        self.tools_frame.grid(
            row=row, column=0, columnspan=2, padx=5, pady=5, sticky='nsew'
        )

        general_settings = self.gui.plot_settings['_']
        self.single_wavelength_enabled = tk.IntVar(
            value=int(general_settings.setdefault('image_mode', 'single') == 'single')
        )
        self.wavelength_variable = tk.IntVar(
            value=int(general_settings.setdefault('image_idx_single', 0))
        )

        self.single_wavelength_checkbutton = ttk.Checkbutton(
            self.tools_frame,
            text='Single wavelength',
            variable=self.single_wavelength_enabled,
        )
        self.single_wavelength_checkbutton.pack()

        self.wavelength_slider = ttk.Scale(
            self.tools_frame,
            variable=self.wavelength_variable,
            from_=0,
            to=self.gui.get_observation().data.shape[0],
        )
        self.wavelength_slider.pack(fill='x')

        self.single_wavelength_label = ttk.Label(
            self.tools_frame,
            justify='center',
        )
        self.single_wavelength_label.pack()

        self.enabled.trace_add('write', self.update_tool_ui_state)
        self.single_wavelength_enabled.trace_add(
            'write', self.on_single_wavelength_checkbutton_change
        )
        self.wavelength_variable.trace_add('write', self.on_wavelength_slider_change)

        self.gui.add_tooltip(
            self.single_wavelength_checkbutton,
            'Toggle displaying single wavelength or average of all wavelengths (click Edit for more options)',
        )
        self.gui.add_tooltip(
            self.wavelength_slider,
            'Change displayed wavelength (click Edit for more options)',
        )

        self.in_tool_updating_state = False
        self.update_tool_ui_state()

    def on_single_wavelength_checkbutton_change(self, *_) -> None:
        if self.in_tool_updating_state or self.gui.get_observation().data.shape[0] < 2:
            return
        is_single = bool(self.single_wavelength_enabled.get())
        self.gui.plot_settings['_']['image_mode'] = 'single' if is_single else 'sum'
        self.schedule_replot(skip_full_delay=True)

    def on_wavelength_slider_change(self, *_) -> None:
        if self.in_tool_updating_state or self.gui.get_observation().data.shape[0] < 2:
            return

        wavelength_idx = self.wavelength_variable.get()
        if not self.single_wavelength_enabled.get():
            # tick the checkbox if user changes the slider value
            if wavelength_idx != self.gui.plot_settings['_']['image_idx_single']:
                self.single_wavelength_enabled.set(1)

        self.gui.plot_settings['_']['image_idx_single'] = wavelength_idx

        self.gui.add_delayed_action(
            'set_slider_label',
            self.SLIDER_DELAY_MS,
            self.set_slider_label,
        )
        self.schedule_replot(set_slider=False)

    def schedule_replot(
        self, skip_full_delay: bool = False, set_slider: bool = True
    ) -> None:
        self.gui.add_delayed_action(
            'do_replot',
            1 if skip_full_delay else self.REPLOT_DELAY_MS,
            self.do_replot if set_slider else self.do_replot_no_slider,
        )

    def do_replot_no_slider(self):
        self.update_tool_ui_state(set_slider=False)
        self.gui.update_only_image()

    def do_replot(self):
        self.update_tool_ui_state()
        self.gui.update_only_image()

    def update_tool_ui_state(self, *_, set_slider: bool = True) -> None:
        self.in_tool_updating_state = True
        try:
            general_settings = self.gui.plot_settings['_']
            n_wavelengths = self.gui.get_observation().data.shape[0]

            enable = self.enabled.get() and n_wavelengths > 1
            is_single_wavelength = bool(
                general_settings.setdefault('image_mode', 'single') == 'single'
            )
            wavelength_idx = int(general_settings.setdefault('image_idx_single', 0))

            self.single_wavelength_enabled.set(int(is_single_wavelength))
            if set_slider:
                self.wavelength_variable.set(wavelength_idx)

            self.single_wavelength_checkbutton.configure(
                state='normal' if enable else 'disable',
            )
            if set_slider:
                self.wavelength_slider.configure(
                    to=n_wavelengths - 1,
                    state='normal' if enable else 'disable',
                )
            self.set_slider_label()
        finally:
            self.in_tool_updating_state = False

    def set_slider_label(self):
        general_settings = self.gui.plot_settings['_']
        image_mode = general_settings.setdefault('image_mode', 'single')
        if image_mode == 'single':
            wavelength_idx = int(general_settings.setdefault('image_idx_single', 0))
            if self.gui._observation_wavelengths is not None:
                value_s = f' ({self.gui._observation_wavelengths[wavelength_idx]:{self.gui._observation_wavelengths_fmt}}{self.gui._observation_wavelengths_unit})'
            else:
                value_s = ''
            self.single_wavelength_label.configure(
                text=f'{wavelength_idx}{value_s}',
                foreground='black',
            )
        else:
            self.single_wavelength_label.configure(
                text=(
                    'Showing sum of all wavelengths'
                    if image_mode == 'sum'
                    else 'Showing RGB composite'
                ),
                foreground='gray50',
            )

    def make_menu(self) -> None:
        settings = self.gui.plot_settings[self.key]
        general_settings = self.gui.plot_settings['_']

        self.cmap = tk.StringVar(value=settings.setdefault('cmap', 'gray'))

        self.image_vmin = tk.StringVar(
            value=str(general_settings.setdefault('image_vmin', 0))
        )
        self.image_vmax = tk.StringVar(
            value=str(general_settings.setdefault('image_vmax', 100))
        )
        self.image_limit_type = tk.StringVar(
            value=str(general_settings.setdefault('image_limit_type', 'absolute'))
        )

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

        self.grid: list[
            tuple[tk.Widget, tk.Widget, set[ImageMode | Literal['_readonly']]]
        ] = [
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
            (
                ttk.Label(frame, text='vmin/vmax type: '),
                ttk.Combobox(
                    frame,
                    textvariable=self.image_limit_type,
                    values=LIMIT_TYPES,
                    width=10,
                    state='readonly',
                ),
                {'single', 'sum', '_readonly'},
            ),
        ]
        self.add_to_menu_grid([(a, b) for a, b, c in self.grid], frame=frame)

        messages = [
            (
                '\nImages are scaled to vary from 0 to 100, so set vmin=0 and vmax=100 '
                'to show the entire dynamic range.'
            ),
            (
                'Set vmin/vmax type to "percentile" to calculate the limits as '
                'percentiles of the data in the image. This can be useful if your data '
                'has extreme outliers (e.g. try vmin=1, vmax=99 & type=percentile).'
            ),
        ]
        ttk.Label(
            self.grid_frame,
            text='\n\n'.join(messages),
            wraplength=self.label_wraplength,
            foreground='gray40',
            justify='center',
        ).pack()

        self.image_mode.trace_add('write', self.change_image_mode_radio)
        self.change_image_mode_radio()  # run initial setup

    def change_image_mode_radio(self, *_) -> None:
        mode = self.image_mode.get()
        for l, widget, modes in self.grid:
            if mode in modes:
                widget['state'] = 'readonly' if '_readonly' in modes else 'normal'
            else:
                widget['state'] = 'disable'

    def get_idx(self, stirng_variable: tk.StringVar, name: str) -> int:
        sz = self.gui.get_observation().data.shape[0]
        value = self.get_int(
            stirng_variable, name=name, positive=False, minimum=-sz, maximum=sz - 1
        )
        if value < 0:
            value = sz - value
        return value

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
                general_settings['image_limit_type'] = self.image_limit_type.get()
                general_settings['image_vmin'] = self.get_float(
                    self.image_vmin, 'vmin', positive=False
                )
                general_settings['image_vmax'] = self.get_float(
                    self.image_vmax, 'vmax', positive=False
                )
                if general_settings['image_vmin'] >= general_settings['image_vmax']:
                    tkinter.messagebox.showwarning(
                        title='Error parsing limits',
                        message='vmin must be less than vmax',
                    )
                    return False
        except ValueError:
            return False

        if image_mode in {'single', 'sum'}:
            try:
                cmap = self.cmap.get()
                plt.get_cmap(cmap)  #  type: ignore
            except ValueError:
                tkinter.messagebox.showwarning(
                    title='Error parsing colormap',
                    message=f'Unrecognised matplotlib colormap {self.cmap.get()!r}',
                )
                return False
            settings['cmap'] = cmap

        self.gui.plot_settings[self.key].update(settings)
        self.gui.plot_settings['_'].update(general_settings)
        self.update_tool_ui_state()
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
        self.grid_lat_limit = tk.StringVar(
            value=str(self.gui.plot_settings['_'].setdefault('grid_lat_limit', 90))
        )
        self.grid_type = tk.StringVar(
            value=(
                'planetocentric'
                if self.gui.plot_settings['_'].setdefault('grid_planetocentric', False)
                else 'planetographic'
            )
        )

        self.radio_frame = ttk.Frame(self.grid_frame)
        self.add_to_menu_grid(
            [
                (
                    ttk.Label(self.grid_frame, text='Grid interval (°): '),
                    ttk.Combobox(
                        self.grid_frame,
                        textvariable=self.grid_interval,
                        values=GRID_INTERVALS,
                        width=11,
                    ),
                ),
                (
                    ttk.Label(self.grid_frame, text='Latitude limit (°): '),
                    ttk.Spinbox(
                        self.grid_frame,
                        textvariable=self.grid_lat_limit,
                        from_=0,
                        to=90,
                        increment=5,
                        width=10,
                    ),
                ),
                (ttk.Label(self.grid_frame, text='Grid type: '), self.radio_frame),
            ]
        )
        for v in ('planetographic', 'planetocentric'):
            ttk.Radiobutton(
                self.radio_frame,
                text=v.capitalize(),
                value=v,
                variable=self.grid_type,
            ).pack(fill='x')

    def apply_settings(self) -> bool:
        try:
            grid_interval = self.get_float(self.grid_interval, 'grid interval')
            grid_lat_limit = self.get_float(self.grid_lat_limit, 'latitude limit')
            grid_planetocentric = self.grid_type.get() == 'planetocentric'
        except ValueError:
            return False
        self.gui.plot_settings['_']['grid_interval'] = grid_interval
        self.gui.plot_settings['_']['grid_lat_limit'] = grid_lat_limit
        self.gui.plot_settings['_']['grid_planetocentric'] = grid_planetocentric
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
        label = (
            'Manually list{s} ring radii in km from the target\'s centre below. '
            'Each radius should be listed on a new line:'
        ).format(s=' more' if self.checkbox_dict else '')
        ttk.Label(
            self.menu_frame, text='\n' + label, wraplength=self.label_wraplength
        ).pack(fill='x')
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
        key: PlotKey,
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
        ttk.Label(
            self.menu_frame,
            text='\n' + self.menu_label,
            wraplength=self.label_wraplength,
        ).pack(fill='x')
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
                            title='Error parsing coordinates',
                            message=f'Line {line!r} must have exactly two values separated by a comma',
                        )
                        return False
                    values.append(
                        tuple(
                            self.get_float(c, 'coordinate', positive=False)
                            for c in coordinates
                        )  #  type: ignore
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
        label = (
            'List other bodies of interest to '
            'mark (e.g. moons). Body names should '
            'be recognisable by SPICE (e.g "Europa" '
            'or "502") with each body listed on a '
            'new line:'
        )
        ttk.Label(
            self.menu_frame, text='\n' + label, wraplength=self.label_wraplength
        ).pack(fill='x')
        self.txt = tkinter.scrolledtext.ScrolledText(self.menu_frame)
        self.txt.pack(fill='both')

        self.button_frame = ttk.Frame(self.menu_frame)
        self.button_frame.pack(fill='x', pady=3)
        self.add_satellites_button = ttk.Button(
            self.button_frame,
            text='Add all visible satellites in target system',
            command=self.add_satellites,
        )
        self.add_satellites_button.pack()

        self.populate_text_box()

    def populate_text_box(
        self, bodies: list[Body | BasicBody] | None = None, append: bool = False
    ) -> None:
        if bodies is None:
            bodies = self.gui.get_observation().other_bodies_of_interest
        bodies = sorted((b for b in bodies), key=lambda b: b.target_body_id)

        if append:
            lines = [l for l in self.txt.get('1.0', 'end').splitlines() if l.strip()]
        else:
            lines = []

        for b in bodies:
            line = b.target
            if line not in lines:
                lines.append(line)
        value = '\n'.join(lines)
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', value)

    def add_satellites(self) -> None:
        bodies = self.gui.get_observation()._get_all_satellite_bodies(
            skip_insufficient_data=True, only_visible=True
        )
        self.populate_text_box(bodies, append=True)

    def apply_other_body_setting(self) -> bool:
        bodies: list[Body | BasicBody] = []
        string = self.txt.get('1.0', 'end')
        lines = [l for line in string.splitlines() if (l := line.strip())]
        for line in set(lines):
            try:
                bodies.append(self.gui.get_observation().create_other_body(line))
            except NotFoundError:
                tkinter.messagebox.showwarning(
                    title='Error parsing target name',
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
    def __init__(
        self,
        gui: GUI,
        parent: tk.Widget,
        key: SetterKey,
        label: str | None = None,
        row: int | None = None,
        add_callbacks: list[SetterKey] | None = None,
        entry_width: int = 10,
        value_fmt: str = '.8g',
        default_callbacks_to_add: tuple[SetterKey, ...] = (
            'wcs_offset_ra',
            'wcs_offset_dec',
            'wcs_offset_rotation',
            'wcs_offset_scale',
        ),
        zero_threshold: float | None = None,
        **kw,
    ):
        self.parent = parent
        self.key: SetterKey = key
        self.gui = gui
        self.entry_width = entry_width
        self.value_fmt = value_fmt
        self.zero_threshold = zero_threshold

        self._enable_callback = True

        if label is None:
            label = key
        self.label = ttk.Label(parent, text=label + ': ')

        self.sv = tk.StringVar()
        self.sv.trace_add('write', self.text_input)
        self.entry = ttk.Entry(parent, width=self.entry_width, textvariable=self.sv)

        self.gui.ui_callbacks[self.key].add(self.update_text)
        if add_callbacks:
            for k in add_callbacks:
                self.gui.ui_callbacks[k].add(self.update_text)
        for k in default_callbacks_to_add:
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
        if self.zero_threshold is not None and abs(value) < self.zero_threshold:
            return '0'
        return format(value, self.value_fmt)

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

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the entry."""
        state = 'normal' if enabled else 'disabled'
        self._enable_callback = enabled
        self.label.configure(state=state)
        self.entry.configure(state=state)
        if enabled:
            self.update_text()
        else:
            self.sv.set('')


class EnableableLabel(ttk.Label):
    def set_enabled(self, enabled: bool) -> None:
        state = 'normal' if enabled else 'disabled'
        self.configure(state=state)


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
    # Custom navigation toolbar for matplotlib plots that improves the cosmetics
    # slightly. This uses our own GUI tooltips rather than the matplotlib ones, and
    # applies some custom colours for consistency with the rest of the GUI. This is
    # especially useful when using dark mode, as the default navigation toolbar will be
    # dark while the rest of the GUI is always in light mode.
    # See issue #320 for more details: https://github.com/ortk95/planetmapper/issues/320

    def __init__(
        self, canvas, window, *, pack_toolbar: bool = True, gui: GUI, bg_color: str
    ) -> None:
        # Default tooltips don't work with tk (on my laptop with dark mode at least)
        # so disable them here by setting to None, then use our custom tooltips instead.
        # This list also removes the Subplots button which we don't want.
        self.toolitems = (
            ('Home', None, 'home', 'home'),
            ('Back', None, 'back', 'back'),
            ('Forward', None, 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', None, 'move', 'pan'),
            ('Zoom', None, 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            ('Save', None, 'filesave', 'save_figure'),
        )
        super().__init__(canvas, window, pack_toolbar=pack_toolbar)

        # The following lines are all cosmetic styling, so aren't crucial. Therefore,
        # wrap everything in try/except to avoid breaking the GUI if matplotlib changes
        # the internals of the toolbar.

        try:
            self.configure(background=bg_color)
        # pylint: disable-next=bare-except
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
                button.configure(highlightbackground=bg_color)
                if isinstance(button, tk.Checkbutton):
                    # In dark mode, text can default to white, making the checkbutton
                    # image difficult to read, especially with the custom backgrounds
                    # we use here. Therefore, set the foreground to black and re-set
                    # the image for the button to ensure it is updated properly.
                    button.configure(foreground='black')
                    self._set_image_for_button(button)
        # pylint: disable-next=bare-except
        except:
            pass

        try:
            for child in self.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(background=bg_color)
            self._message_label.configure(foreground='#888888')
        # pylint: disable-next=bare-except
        except:
            pass
