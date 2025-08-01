"""
Various general helpful utilities.
"""

import os
import pathlib
import warnings
from typing import Literal, Sequence

import matplotlib.ticker
import numpy as np
from astropy.io import fits
from matplotlib.axes import Axes


def format_radec_axes(
    ax: Axes,
    dec: float,
    dms_ticks: bool = True,
    add_axis_labels: bool = True,
    aspect_adjustable: Literal['box', 'datalim'] | None = 'datalim',
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
            Passed to :func:`matplotlib.axes.Axes.set_aspect`. Set to None to skip
            setting the aspect ratio (generally this is only recommended if you're
            setting the aspect ratio yourself).
    """
    if add_axis_labels:
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
    if aspect_adjustable is not None:
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

    # pylint: disable-next=unused-argument
    def _format(self, dd, pos):
        d, m, s = decimal_degrees_to_dms(dd)
        out = []
        if 'd' not in self.skip_parts or (m == 0 and s == 0):
            out.append(f'{d}°')
        if 'm' not in self.skip_parts or ('d' in self.skip_parts and s == 0):
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
            self.skip_parts = set()
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


def decimal_degrees_to_dms_str(decimal_degrees: float, seconds_fmt: str = 'g') -> str:
    """
    Create nicely formatted DMS string from decimal degrees value (e.g. `'12°34′56″'`).

    Uses :func:`decimal_degrees_to_dms` to perform the conversion.

    Args:
        decimal_degrees: Decimal degrees.
        seconds_fmt: Optionally specify a format string for the seconds part of the
            returned value. For example, `seconds_fmt='.3f'` will fix three decimal
            places for the fractional part of the seconds value. Note that the integral
            part of the seconds value will always be zero-padded to two digits, so
            `seconds_fmt='.3f'` will return seconds as e.g. `01.234`.

    Returns:
        String representing the degrees, minutes, seconds of the angle.
    """
    d, m, s = decimal_degrees_to_dms(decimal_degrees)
    s_str = f'{s:{seconds_fmt}}'
    if len(s_str.split('.')[0]) < 2:
        s_str = '0' + s_str  # Zero pad integer part of seconds to e.g. 01.234
    return f'{d}°{m:02d}′{s_str}″'


class ignore_warnings(warnings.catch_warnings):
    """
    Context manager to ignore general warnings using warnings.filterwarnings.
    """

    def __init__(self, *warning_strings: str, **kwargs):
        super().__init__(**kwargs)
        self.warning_strings = warning_strings

    def __enter__(self):
        out = super().__enter__()
        for ws in self.warning_strings:
            warnings.filterwarnings('ignore', ws)
        return out


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


def normalise(
    values: np.ndarray | Sequence[float],
    top: float = 1.0,
    bottom: float = 0.0,
    single_value: float | None = None,
) -> np.ndarray:
    """
    Normalise iterable.

    Args:
        values: Iterable of values to normalise.
        top: Top of normalised range.
        bottom: Bottom of normalised range.
        single_value: If all values are the same, return this value.

    Returns:
        Normalised values.
    """
    assert top > bottom
    values = np.array(values)
    if single_value is not None and len(set(values)) == 1:
        return np.full(values.shape, single_value)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    # Put into 0 to 1 range
    if vmax != vmin:
        values = (values - vmin) / (vmax - vmin)  #  type: ignore
    else:
        values = values - vmin
    return values * (top - bottom) + bottom  #  type: ignore


def check_path(path: str) -> None:
    """
    Checks if file path's directory tree exists, and creates it if necessary.

    Assumes path is to a file if `os.path.split(path)[1]` contains '.',
    otherwise assumes path is to a directory.
    """
    path = os.path.expandvars(os.path.expanduser(path))
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


# Wavelengths


class GetWavelengthsError(ValueError):
    """
    Error raised when wavelength information cannot be extracted from a FITS header.
    """


def generate_wavelengths_from_header(
    header: fits.Header | dict,
    *,
    check_ctype: bool = True,
    axis: int = 3,
) -> np.ndarray:
    """
    Generate wavelength array from keyword values in a FITS Header.

    This uses the NAXIS3, CRVAL3, CDELT3 (or CD3_3) and CRPIX3 keywords to generate the
    wavelength array described by the Header. The axis to generate wavelengths for can
    be customised using the `axis` parameter.

    By default, this function will raise an exception if the CTYPE of the axis is not
    'WAVE'. This can be disabled by setting `check_ctype` to False.

    See the
    `JWST documentation <https://jwst-docs.stsci.edu/jwst-calibration-status/miri-calibration-status/miri-mrs-calibration-status>`_
    for an an example of how the wavelength array can be generated from the FITS Header.

    Args:
        header: FITS Header object (or dictionary).
        check_ctype: Check that the CTYPE of the axis is 'WAVE'.
        axis: Axis to generate wavelengths for, using FITS (1-based) counting. This
            defaults to 3.

    Returns:
        Wavelength array.

    Raises:
        GetWavelengthsError: If the wavelength array cannot be generated from the
            FITS Header.
    """
    try:
        if check_ctype and header[f'CTYPE{axis}'] != 'WAVE':
            raise GetWavelengthsError(
                f'Header item CTYPE{axis} = {header[f"CTYPE{axis}"]!r} (not \'WAVE\')'
            )

        naxis3 = int(header[f'NAXIS{axis}'])  # type: ignore
        crval3 = float(header[f'CRVAL{axis}'])  # type: ignore
        try:
            cdelt3 = float(header[f'CDELT{axis}'])  #  type: ignore
        except KeyError:
            cdelt3 = float(header[f'CD{axis}_{axis}'])  # type: ignore
        crpix3 = float(header.get(f'CRPIX{axis}', 1))  #  type: ignore
    except (KeyError, ValueError, TypeError) as e:
        raise GetWavelengthsError(
            'Could not generate wavelength array from FITS Header'
        ) from e

    # https://jwst-docs.stsci.edu/jwst-calibration-status/miri-calibration-status/miri-mrs-calibration-status
    wavl = (np.arange(naxis3) + crpix3 - 1) * cdelt3 + crval3
    return wavl
