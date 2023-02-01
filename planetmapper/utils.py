"""
Various general helpful utilities.
"""
import os
import pathlib
import traceback
import warnings
from datetime import datetime
from typing import Literal

import matplotlib.ticker
import numpy as np
from matplotlib.axes import Axes


def format_radec_axes(
    ax: Axes,
    dec: float,
    dms_ticks: bool = True,
    add_axis_labels: bool = True,
    aspect_adjustable: Literal['box', 'datalim'] = 'datalim',
) -> None:
    """
    Format an axis to display RA/Dec coordinates nicely.

    Args:
        ax: Matplotlib axis to format.
        dec: Declination in degrees of centre of axis.
        dms_ticks: Toggle between showing ticks as degrees, minutes and seconds
            (e.g. 12°34′56″) or decimal degrees (e.g. 12.582).
        add_axis_labels: Add axis labels.
        aspect_adjustable: Set `adjustable` parameter when setting the aspect ratio.
            Passed to :func:`matplotlib.axes.Axes.set_aspect`.
    """
    if add_axis_labels:
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
    ax.set_aspect(1 / np.cos(np.deg2rad(dec)), adjustable=aspect_adjustable)
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    if dms_ticks:
        ax.yaxis.set_major_locator(DMSLocator())
        ax.yaxis.set_major_formatter(DMSFormatter())
        ax.xaxis.set_major_locator(DMSLocator())
        ax.xaxis.set_major_formatter(DMSFormatter())


class DMSFormatter(matplotlib.ticker.FuncFormatter):
    """
    Matplotlib tick formatter to display angular values as degrees, minutes and seconds
    e.g. `12°34′56″`. Designed to work with :class:`DMSLocator`. ::

        ax = plt.cga()

        ax.yaxis.set_major_locator(planetmapper.utils.DMSLocator())
        ax.yaxis.set_major_formatter(planetmapper.utils.DMSFormatter())

        ax.xaxis.set_major_locator(planetmapper.utils.DMSLocator())
        ax.xaxis.set_major_formatter(planetmapper.utils.DMSFormatter())
    """

    def __init__(self) -> None:
        super().__init__(self._format)
        self.skip_parts = set()
        self.fmt_s = '02.0f'

    def _format(self, dd, pos):
        d, m, s = decimal_degrees_to_dms(dd)
        out = []
        if 'd' not in self.skip_parts or m == 0 and s == 0:
            out.append(f'{d}°')
        if 'm' not in self.skip_parts or s == 0:
            out.append(f'{m:02.0f}′')
        if 's' not in self.skip_parts:
            out.append(f'{s:{self.fmt_s}}″')
        return ''.join(out)

    def set_locs(self, locs) -> None:
        """:meta private:"""
        vmin, vmax = sorted(self.axis.get_view_interval())
        dms_min = decimal_degrees_to_dms(vmin)
        dms_max = decimal_degrees_to_dms(vmax)
        vrange = abs(vmax - vmin)

        self.skip_parts.clear()
        ofs = ''
        if dms_min[:2] == dms_max[:2]:
            d, m, s = dms_min
            self.skip_parts.add('d')
            self.skip_parts.add('m')
            if d != 0 or m != 0:
                ofs = f'{d:+.0f}°{m:02.0f}′'
        elif dms_min[0] == dms_max[0]:
            d, m, s = dms_min
            self.skip_parts.add('d')
            if d != 0:
                ofs = f'{d:+.0f}°'

        if vrange > 10 / 60:
            self.skip_parts.add('s')
        if vrange > 10:
            self.skip_parts.add('m')

        if vrange < 10 / 3600:
            self.skip_parts.add('m')
        if vrange < 10 / 60:
            self.skip_parts.add('d')

        if vrange < 0.01 / 3600:
            self.fmt_s = '.3g'
        elif vrange < 0.1 / 3600:
            self.fmt_s = '.3f'
        elif vrange < 1 / 3600:
            self.fmt_s = '.2f'
        elif vrange < 10 / 3600:
            self.fmt_s = '.1f'
        else:
            self.fmt_s = '02.0f'

        if self.skip_parts == {'d', 'm', 's'}:
            self.skip_parts = {'d', 'm', 's'}
        self.set_offset_string(ofs)
        return super().set_locs(locs)


class DMSLocator(matplotlib.ticker.Locator):
    """
    Matplotlib tick locator to display angular values as degrees, minutes and seconds.
    Designed to work with :class:`DMSFormatter`. ::

        ax = plt.cga()

        ax.yaxis.set_major_locator(planetmapper.utils.DMSLocator())
        ax.yaxis.set_major_formatter(planetmapper.utils.DMSFormatter())

        ax.xaxis.set_major_locator(planetmapper.utils.DMSLocator())
        ax.xaxis.set_major_formatter(planetmapper.utils.DMSFormatter())
    """

    def __init__(self) -> None:
        super().__init__()
        steps = [1, 2, 5, 10]
        self.locator = matplotlib.ticker.MaxNLocator(steps=steps, nbins=8)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
        """:meta private:"""
        vrange = abs(vmax - vmin)
        if vrange < 1 / 60:
            multiplier = 3600
        elif vrange < 1:
            multiplier = 60
        else:
            multiplier = 1
        ticks = self.locator.tick_values(vmin * multiplier, vmax * multiplier)
        return ticks / multiplier


def decimal_degrees_to_dms(decimal_degrees: float) -> tuple[int, int, float]:
    """
    Get degrees, minutes, seconds from decimal degrees.

    `decimal_degrees_to_dms(-11.111)` returns `(-11.0, 6.0, 39.6)`.

    Args:
        decimal_degrees: Decimal degrees.

    Returns:
        `(degrees, minutes, seconds)` tuple
    """
    dd = abs(decimal_degrees)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    if decimal_degrees < 0:
        if degrees:
            degrees = -degrees
        elif minutes:
            minutes = -minutes
        else:
            seconds = -seconds
    return int(degrees), int(minutes), seconds


def decimal_degrees_to_dms_str(decimal_degrees: float, seconds_fmt: str = '') -> str:
    """
    Create nicely formated DMS string from decimal degrees value (e.g. `'12°34′56″'`).

    Uses :func:`decimal_degrees_to_dms` to perform the conversion.

    Args:
        decimal_degrees: Decimal degrees.
        seconds_fmt: Optionally specify a format string for the seconds part of the
            returned value. For example, `seconds_fmt='.3f'` will fix three decimal
            places for the fractional part of the seconds value.

    Returns:
        String representting the degress, minutes, seconds of the angle.
    """
    d, m, s = decimal_degrees_to_dms(decimal_degrees)
    return f'{d}°{m}′{s:{seconds_fmt}}″'


class filter_fits_comment_warning(warnings.catch_warnings):
    """
    Context manager to hide FITS `Card is too long, comment will be truncated` warnings.
    """

    def __enter__(self):
        out = super().__enter__()
        warnings.filterwarnings(
            'ignore',
            message='Card is too long, comment will be truncated.',
            module='astropy.io.fits.card',
        )
        return out


def cprint(*msg, fg=None, bg=None, style=None, skip_print=False, sep=' ', **kwargs):
    """
    Prints coloured and formatted text.

    Parameters
    ----------
    msg
        Message to print.

    fg, bg : {'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w'}
        Foreground and background colours (see code for options).

    style : {'b', 'f', 'i', 'u', 'x', 'y', 'r', 'h', 's'}
        Formatting style to apply to text. Can be multiple values, e.g. 'bi'
        for bold and italic style.

    """
    colcode = {
        'k': 0,  # black
        'r': 1,  # red
        'g': 2,  # green
        'y': 3,  # yellow
        'b': 4,  # blue
        'm': 5,  # magenta
        'c': 6,  # cyan
        'w': 7,  # white
    }

    fmtcode = {
        'b': 1,  # bold
        'f': 2,  # faint
        'i': 3,  # italic
        'u': 4,  # underline
        'x': 5,  # blinking
        'y': 6,  # fast blinking
        'r': 7,  # reverse
        'h': 8,  # hide
        's': 9,  # strikethrough
    }

    # Properties
    props = []
    if isinstance(style, str):
        props = [fmtcode[s] for s in style]
    if isinstance(fg, str):
        props.append(30 + colcode[fg])
    if isinstance(bg, str):
        props.append(40 + colcode[bg])

    # Display
    msg = sep.join(
        str(x) for x in msg
    )  # Reproduce print() behaviour for easy translation
    props = ';'.join(str(x) for x in props)

    if props:
        msg = '\x1b[%sm%s\x1b[0m' % (props, msg)

    if not skip_print:
        print(msg, **kwargs)

    return msg


def print_progress(annotation=None, c1='g', c2='k', style='b'):
    """
    Print progress summary of current code.

    Prints summary of code location and execution time for use in optimising and monitoring code.
    Uses traceback to identify the call stack and prints tree-like diagram of stacks where this
    function was called. The call stack is relative to the first time this function is called as it
    uses properties of print_progress to communicate between calls.

    Printed output contains:
        - Seconds elapsed since last call of print_progress.
        - Current time.
        - Traceback for location print_progress is called from, relative to first location
          print_progress was called. This includes file names, function names and line numbers.
        - Optional annotation provided to explain what current line is.

    Arguments
    ---------
    annotation : str
        Optionally provide annotation about current line of code.

    c1, c2, style : str
        Formatting options to pass to cprint()
    """
    now = datetime.now()  # Ignore duration of current code

    # Get timings
    if print_progress.last_dtm is None:
        title = ' seconds'
    else:
        td = now - print_progress.last_dtm
        title = f'{td.total_seconds():8.2f}'
    title += ' @ '
    title += now.strftime('%H:%M:%S')
    title += ' '

    # Get stack
    stack = traceback.extract_stack()[:-1]
    if print_progress.first_stack is None:
        print_progress.first_stack = stack
    first_stack = print_progress.first_stack
    split_idx = len([None for a, b in zip(stack[:-1], first_stack) if a == b])
    stack = stack[split_idx:]
    stack_parts = [(s[2], s[1], os.path.split(s[0])[1]) for s in stack]
    stack_text = [
        f'{a} (line {b} in <ipython>)'
        if c.startswith('<ipython-input-')
        else f'{a} (line {b} in {c})'
        for a, b, c in stack_parts
    ]
    last_stack = print_progress.last_stack
    if last_stack is not None:
        last_idx = len([None for a, b in zip(stack[:-1], last_stack) if a == b])
        stack_text = stack_text[last_idx:]
    else:
        last_idx = 0
    print_progress.last_stack = stack

    # Do actual printing
    for st in stack_text:
        msg = cprint(
            ' ' + '│  ' * last_idx + st + ' ',
            fg=c1,
            bg=c2,
            style=style,
            skip_print=True,
        )
        if st == stack_text[-1]:
            tt = title
            if annotation:
                msg += cprint(
                    f'{annotation} ', fg=c1, bg=c2, style=style + 'i', skip_print=True
                )
        else:
            tt = ' ' * len(title)
        tt = cprint(tt, fg=c2, bg=c1, style=style, skip_print=True)
        print(tt + msg, flush=True)
        last_idx += 1
    print_progress.last_dtm = datetime.now()  # Ignore duration of this code


print_progress.last_dtm = None
print_progress.last_stack = None
print_progress.first_stack = None


def normalise(values, top=1, bottom=0, single_value=None):
    """
    Normalise iterable.
    """
    assert top > bottom
    values = np.array(values)
    if single_value is not None and len(set(values)) == 1:
        return np.full(values.shape, single_value)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    # Put into 0 to 1 range
    if vmax != vmin:
        values = (values - vmin) / (vmax - vmin)
    else:
        values = values - vmin
    return values * (top - bottom) + bottom


def check_path(path: str):
    """
    Checks if file path's directory tree exists, and creates it if necessary.

    Assumes path is to a file if `os.path.split(path)[1]` contains '.',
    otherwise assumes path is to a directory.
    """
    if os.path.isdir(path):
        return
    if '.' in os.path.split(path)[1]:
        path = os.path.split(path)[0]
        if os.path.isdir(path):
            return
    if path == '':
        return
    print('Creating directory path "{}"'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
