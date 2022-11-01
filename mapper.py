#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    
## Coordinate systems # TODO add units
- `xy` - image pixel coordinates [only used in subclass?] # TODO
- `radec` - observer frame RA/Dec coordinates
- `obsvec` - observer frame (e.g. J2000) rectangular vector
- `obsvec_norm` - normalised observer frame rectangular vector
    i.e. magnitude is meaningless and only direction matters for `obsvec_norm`
- `rayvec` - target frame rectangular vector from observer to point
- `targvec` - target frame rectangular vector
- `lonlat` - planetary coordinates on target

By default, angles should be degrees unless specified with `_radians`. Note that
angles in spice are radians, so care should be taken converting to/from spice
values.
"""
import datetime
import glob
import math
import os
import sys
from typing import Callable, Iterable, TypeVar, ParamSpec, NamedTuple, cast, Any
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.patches
from matplotlib.transforms import Transform
from matplotlib.axes import Axes
import numpy as np
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError
import utils
from functools import wraps
import PIL.Image
from astropy.io import fits
import warnings

__version__ = '0.1.1'
__author__ = 'Oliver King'
__url__ = 'https://github.com/ortk95/planetmapper'

KERNEL_PATH = '~/spice/naif/generic_kernels/'

T = TypeVar('T')
P = ParamSpec('P')
Numeric = TypeVar('Numeric', bound=float | np.ndarray)


class Backplane(NamedTuple):
    name: str
    description: str
    fn: Callable[[], np.ndarray]


def main(*args):
    utils.print_progress()
    o = Observation('data/europa.fits.gz')
    print(o)
    utils.print_progress('__init__')
    # o.plot_backplane('radial_velocity')
    # utils.print_progress('plot')
    o.add_header_metadata()
    lines = o.header.tostring(sep='\n', endcard=False).strip().splitlines()
    print(*lines[-30:], sep='\n')

    o.save('data/test_out.fits.gz')
    utils.print_progress('saved')


class SpiceTool:
    """
    Basic class containing methods to interface with spice and basic tools.
    """

    DEFAULT_DTM_FORMAT_STRING = '%Y-%m-%dT%H:%M:%S.%f'
    _KERNELS_LOADED = False

    def __init__(self, optimize_speed: bool = True) -> None:
        super().__init__()
        self._optimize_speed = optimize_speed

    @staticmethod
    def standardise_body_name(name: str) -> str:
        name = spice.bodc2s(spice.bods2c(name))
        return name

    @staticmethod
    def et2dtm(et: float) -> datetime.datetime:
        s = spice.et2utc(et, 'ISOC', 6) + '+0000'
        # manually add '+0000' to string to make it timezone aware
        # i.e. this lets python know it is UTC
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')

    @classmethod
    def load_spice_kernels(
        cls, kernel_path: str = KERNEL_PATH, manual_kernels: None | list[str] = None, only_if_needed:bool=True,
    ) -> None:
        # TODO do this better - don't necessarily need to keep running it every time
        if only_if_needed and cls._KERNELS_LOADED:
            return

        if manual_kernels:
            kernels = manual_kernels
        else:
            kernel_path = os.path.expanduser(kernel_path)
            pcks = sorted(glob.glob(kernel_path + 'pck/*.tpc'))
            spks1 = sorted(glob.glob(kernel_path + 'spk/planets/de*.bsp'))
            spks2 = sorted(glob.glob(kernel_path + 'spk/satellites/*.bsp'))
            fks = sorted(glob.glob(kernel_path + 'fk/planets/*.tf'))
            lsks = sorted(glob.glob(kernel_path + 'lsk/naif*.tls'))
            jwst = sorted(glob.glob(kernel_path + '../../jwst/*.bsp'))
            kernels = [pcks[-1], spks1[-1], *spks2, lsks[-1], *jwst]
        for kernel in kernels:
            spice.furnsh(kernel)
        cls._KERNELS_LOADED = True

    @staticmethod
    def close_loop(arr: np.ndarray) -> np.ndarray:
        return np.append(arr, [arr[0]], axis=0)

    @staticmethod
    def _radian_pair2degrees(
        radians0: Numeric, radians1: Numeric
    ) -> tuple[Numeric, Numeric]:
        return np.rad2deg(radians0), np.rad2deg(radians1)  # type: ignore

    @staticmethod
    def _degree_pair2radians(
        degrees0: Numeric, degrees1: Numeric
    ) -> tuple[Numeric, Numeric]:
        return np.deg2rad(degrees0), np.deg2rad(degrees1)  # type: ignore

    @staticmethod
    def unit_vector(v: np.ndarray) -> np.ndarray:
        # Fastest method
        return v / (sum(v * v)) ** 0.5

    def _encode_str(self, s: str) -> bytes | str:
        if self._optimize_speed:
            return s.encode('UTF-8')
        else:
            return s


class Body(SpiceTool):
    """
    Class representing spice data about an observation of an astronomical body.
    """

    def __init__(
        self,
        target: str,
        utc: str | datetime.datetime,
        observer: str = 'EARTH',
        *,
        observer_frame: str = 'J2000',
        illumination_source: str = 'SUN',
        aberration_correction: str = 'CN+S',
        subpoint_method: str = 'INTERCEPT/ELLIPSOID',
        surface_method: str = 'ELLIPSOID',
        load_kernels: bool = True,
        kernel_path: str = KERNEL_PATH,
        manual_kernels: None | list[str] = None,
        **kw,
    ) -> None:
        super().__init__(**kw)

        # Process inputs
        self.target = self.standardise_body_name(target)
        if isinstance(utc, datetime.datetime):
            # convert input datetime to UTC, then to a string compatible with spice
            utc = utc.astimezone(datetime.timezone.utc)
            utc = utc.strftime(self.DEFAULT_DTM_FORMAT_STRING)
        self.utc = utc
        self.observer = self.standardise_body_name(observer)
        self.observer_frame = observer_frame
        self.illumination_source = illumination_source
        self.aberration_correction = aberration_correction
        self.subpoint_method = subpoint_method
        self.surface_method = surface_method

        # Encode strings which are regularly passed to spice (for speed)
        self._target_encoded = self._encode_str(self.target)
        self._observer_encoded = self._encode_str(self.observer)
        self._observer_frame_encoded = self._encode_str(self.observer_frame)
        self._illumination_source_encoded = self._encode_str(self.illumination_source)
        self._aberration_correction_encoded = self._encode_str(aberration_correction)
        self._subpoint_method_encoded = self._encode_str(subpoint_method)
        self._surface_method_encoded = self._encode_str(surface_method)

        # Load kernels
        if load_kernels:
            self.load_spice_kernels(
                kernel_path=kernel_path, manual_kernels=manual_kernels
            )

        # Get target properties and state
        self.et = spice.utc2et(self.utc)
        self.dtm = self.et2dtm(self.et)
        self.target_frame = 'IAU_' + self.target
        self._target_frame_encoded = self._encode_str(self.target_frame)
        self.target_body_id: int = spice.bodn2c(self.target)

        self.radii = spice.bodvar(self.target_body_id, 'RADII', 3)
        self.r_eq = self.radii[0]
        self.r_polar = self.radii[2]
        self.flattening = (self.r_eq - self.r_polar) / self.r_eq

        starg, lt = spice.spkezr(
            self._target_encoded,  # type: ignore
            self.et,
            self._observer_frame_encoded,  # type: ignore
            self._aberration_correction_encoded,  # type: ignore
            self._observer_encoded,  # type: ignore
        )
        self._target_obsvec = cast(np.ndarray, starg)[:3]
        self.target_light_time = cast(float, lt)
        # cast() calls are only here to make type checking play nicely with spice.spkezr
        self.target_distance = self.target_light_time * spice.clight()
        self._target_ra_radians, self._target_dec_radians = self._obsvec2radec_radians(
            self._target_obsvec
        )
        self.target_ra, self.target_dec = self._radian_pair2degrees(
            self._target_ra_radians, self._target_dec_radians
        )

        # Find sub observer point
        self._subpoint_targvec, self._subpoint_et, self._subpoint_rayvec = spice.subpnt(
            self._subpoint_method_encoded,  # type: ignore
            self._target_encoded,  # type: ignore
            self.et,
            self._target_frame_encoded,  # type: ignore
            self._aberration_correction_encoded,  # type: ignore
            self._observer_encoded,  # type: ignore
        )
        self.subpoint_distance = np.linalg.norm(self._subpoint_rayvec)
        self.subpoint_lon, self.subpoint_lat = self.targvec2lonlat(
            self._subpoint_targvec
        )
        self._subpoint_obsvec = self._rayvec2obsvec(
            self._subpoint_rayvec, self._subpoint_et
        )
        self.subpoint_ra, self.subpoint_dec = self._radian_pair2degrees(
            *self._obsvec2radec_radians(self._subpoint_obsvec)
        )

    def __repr__(self) -> str:
        return f'Body({self.target!r}, {self.utc!r})'

    # Coordinate transformations target -> observer direction
    def _lonlat2targvec_radians(self, lon: float, lat: float) -> np.ndarray:
        """
        Transform lon/lat coordinates on body to rectangular vector in target frame.
        """
        return spice.pgrrec(
            self._target_encoded,  # type: ignore
            lon,
            lat,
            0,
            self.r_eq,
            self.flattening,
        )

    def _targvec2obsvec(self, targvec: np.ndarray) -> np.ndarray:
        """
        Transform rectangular vector in target frame to rectangular vector in observer
        frame.
        """
        # Get the target vector from the subpoint to the point of interest
        targvec_offset = targvec - self._subpoint_targvec  # type: ignore
        # ^ ignoring type warning due to numpy bug (TODO remove type: ingore in future)
        # https://github.com/numpy/numpy/issues/22437

        # Calculate the difference in LOS distance between observer<->subpoint and
        # observer<->point of interest
        dist_offset = (
            np.linalg.norm(self._subpoint_rayvec + targvec_offset)
            - self.subpoint_distance
        )

        # Use the calculated difference in distance relative to the subpoint to
        # calculate the time corresponding to when the ray left the surface at the point
        # of interest
        targvec_et = self._subpoint_et - dist_offset / spice.clight()

        # Create the transform matrix converting between the target vector at the time
        # the ray left the point of interest -> the observer vector at the time the ray
        # hit the detector
        transform_matrix = spice.pxfrm2(
            self._target_frame_encoded,  # type: ignore
            self._observer_frame_encoded,  # type: ignore
            targvec_et,
            self.et,
        )

        # Use the transform matrix to perform the actual transformation
        return self._subpoint_obsvec + np.matmul(transform_matrix, targvec_offset)

    def _rayvec2obsvec(self, rayvec: np.ndarray, et: float) -> np.ndarray:
        """
        Transform rectangular vector from point to observer in target frame to
        rectangular vector of point in observer frame.
        """
        px = spice.pxfrm2(
            self._target_frame_encoded,  # type: ignore
            self._observer_frame_encoded,  # type: ignore
            et,
            self.et,
        )
        return np.matmul(px, rayvec)

    def _obsvec2radec_radians(self, obsvec: np.ndarray) -> tuple[float, float]:
        """
        Transform rectangular vector in observer frame to observer ra/dec coordinates.
        """
        dst, ra, dec = spice.recrad(obsvec)
        return ra, dec

    # Coordinate transformations observer -> target direction
    def _radec2obsvec_norm_radians(self, ra: float, dec: float) -> np.ndarray:
        return spice.radrec(1, ra, dec)

    def _obsvec_norm2targvec(self, obsvec_norm: np.ndarray) -> np.ndarray:
        """TODO add note about raising NotFoundError"""
        spoint, trgepc, srfvec = spice.sincpt(
            self._surface_method_encoded,
            self._target_encoded,
            self.et,
            self._target_frame_encoded,
            self._aberration_correction_encoded,
            self._observer_encoded,
            self._observer_frame_encoded,
            obsvec_norm,
        )
        return spoint

    def _targvec2lonlat_radians(self, targvec: np.ndarray) -> tuple[float, float]:
        lon, lat, alt = spice.recpgr(
            self._target_encoded,  # type: ignore
            targvec,
            self.r_eq,
            self.flattening,
        )
        return lon, lat

    # Useful transformations (built from combinations of above transformations)
    def _lonlat2radec_radians(self, lon: float, lat: float) -> tuple[float, float]:
        return self._obsvec2radec_radians(
            self._targvec2obsvec(
                self._lonlat2targvec_radians(lon, lat),
            )
        )

    def lonlat2radec(self, lon: float, lat: float) -> tuple[float, float]:
        return self._radian_pair2degrees(
            *self._lonlat2radec_radians(*self._degree_pair2radians(lon, lat))
        )

    def _radec2lonlat_radians(
        self, ra: float, dec: float, not_found_nan: bool = True
    ) -> tuple[float, float]:
        try:
            ra, dec = self._targvec2lonlat_radians(
                self._obsvec_norm2targvec(
                    self._radec2obsvec_norm_radians(ra, dec),
                )
            )
        except NotFoundError:
            if not_found_nan:
                ra = np.nan
                dec = np.nan
            else:
                raise
        return ra, dec

    def radec2lonlat(self, ra: float, dec: float, **kw) -> tuple[float, float]:
        return self._radian_pair2degrees(
            *self._radec2lonlat_radians(*self._degree_pair2radians(ra, dec), **kw)
        )

    def lonlat2targvec(self, lon: float, lat: float) -> np.ndarray:
        return self._lonlat2targvec_radians(*self._degree_pair2radians(lon, lat))

    def targvec2lonlat(self, targvec: np.ndarray) -> tuple[float, float]:
        return self._radian_pair2degrees(*self._targvec2lonlat_radians(targvec))

    def _targvec_arr2radec_arrs_radians(
        self,
        targvec_arr: np.ndarray | list[np.ndarray],
        condition_func: None | Callable[[np.ndarray], bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if condition_func is not None:
            ra_dec = [
                self._obsvec2radec_radians(self._targvec2obsvec(t))
                if condition_func(t)
                else (np.nan, np.nan)
                for t in targvec_arr
            ]
        else:
            ra_dec = [
                self._obsvec2radec_radians(self._targvec2obsvec(t)) for t in targvec_arr
            ]
        ra = np.array([r for r, d in ra_dec])
        dec = np.array([d for r, d in ra_dec])
        return ra, dec

    def _targvec_arr2radec_arrs(
        self,
        targvec_arr: np.ndarray | list[np.ndarray],
        condition_func: None | Callable[[np.ndarray], bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._radian_pair2degrees(
            *self._targvec_arr2radec_arrs_radians(targvec_arr, condition_func)
        )

    # General
    def _illumf_from_targvec_radians(
        self, targvec: np.ndarray
    ) -> tuple[float, float, float, bool, bool]:
        trgepc, srfvec, phase, incdnc, emissn, visibl, lit = spice.illumf(
            self._surface_method_encoded,  # type: ignore
            self._target_encoded,  # type: ignore
            self._illumination_source_encoded,  # type: ignore
            self.et,
            self._target_frame_encoded,  # type: ignore
            self._aberration_correction_encoded,  # type: ignore
            self._observer_encoded,  # type: ignore
            targvec,
        )
        return phase, incdnc, emissn, visibl, lit

    # Limb
    def _limb_targvec(
        self,
        npts: int = 100,
        close_loop: bool = True,
        method: str = 'TANGENT/ELLIPSOID',
        corloc: str = 'ELLIPSOID LIMB',
    ) -> np.ndarray:
        refvec = [0, 0, 1]
        rolstp = 2 * np.pi / npts
        _, points, epochs, tangts = spice.limbpt(
            method,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            corloc,
            self.observer,
            refvec,
            rolstp,
            npts,
            4,
            self.target_distance,
            npts,
        )
        if close_loop:
            points = self.close_loop(points)
        return points

    def limb_radec(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._targvec_arr2radec_arrs(self._limb_targvec(**kw))

    def limb_radec_by_illumination(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        targvec_arr = self._limb_targvec(**kw)
        ra_day, dec_day = self._targvec_arr2radec_arrs(targvec_arr)
        ra_night = ra_day.copy()
        dec_night = dec_day.copy()
        for idx, targvec in enumerate(targvec_arr):
            if self._test_if_targvec_illuminated(targvec):
                ra_night[idx] = np.nan
                dec_night[idx] = np.nan
            else:
                ra_day[idx] = np.nan
                dec_day[idx] = np.nan
        return ra_day, dec_day, ra_night, dec_night

    # Visibility
    def _test_if_targvec_visible(self, targvec: np.ndarray) -> bool:
        phase, incdnc, emissn, visibl, lit = self._illumf_from_targvec_radians(targvec)
        return visibl

    def test_if_lonlat_visible(self, lon: float, lat: float) -> bool:
        return self._test_if_targvec_visible(self.lonlat2targvec(lon, lat))

    # Illumination
    def _illumination_angles_from_targvec_radians(
        self, targvec: np.ndarray
    ) -> tuple[float, float, float]:
        phase, incdnc, emissn, visibl, lit = self._illumf_from_targvec_radians(targvec)
        return phase, incdnc, emissn

    def illumination_angles_from_lonlat(
        self, lon: float, lat: float
    ) -> tuple[float, float, float]:
        phase, incdnc, emissn = self._illumination_angles_from_targvec_radians(
            self.lonlat2targvec(lon, lat)
        )
        return np.deg2rad(phase), np.deg2rad(incdnc), np.deg2rad(emissn)

    def terminator_radec(
        self,
        npts: int = 100,
        only_visible: bool = True,
        close_loop: bool = True,
        method: str = 'UMBRAL/TANGENT/ELLIPSOID',
        corloc: str = 'ELLIPSOID TERMINATOR',
    ) -> tuple[np.ndarray, np.ndarray]:
        refvec = [0, 0, 1]
        rolstp = 2 * np.pi / npts
        _, targvec_arr, epochs, trmvcs = spice.termpt(
            method,
            self.illumination_source,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            corloc,
            self.observer,
            refvec,
            rolstp,
            npts,
            4,
            self.target_distance,
            npts,
        )
        if close_loop:
            targvec_arr = self.close_loop(targvec_arr)
        ra, dec = self._targvec_arr2radec_arrs(
            targvec_arr,
            condition_func=self._test_if_targvec_visible if only_visible else None,
        )
        return ra, dec

    def _test_if_targvec_illuminated(self, targvec: np.ndarray) -> bool:
        phase, incdnc, emissn, visibl, lit = self._illumf_from_targvec_radians(targvec)
        return lit

    def test_if_lonlat_illuminated(self, lon: float, lat: float) -> bool:
        return self._test_if_targvec_illuminated(self.lonlat2targvec(lon, lat))

    # Lonlat grid
    def visible_lon_grid_radec(
        self, lons: list[float] | np.ndarray, npts: int = 50
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lats = np.linspace(-90, 90, npts)
        out = []
        for lon in lons:
            targvecs = [self.lonlat2targvec(lon, lat) for lat in lats]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lat_grid_radec(
        self, lats: list[float] | np.ndarray, npts: int = 100
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lons = np.linspace(0, 360, npts)
        out = []
        for lat in lats:
            targvecs = [self.lonlat2targvec(lon, lat) for lon in lons]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_latlon_grid_radec(
        self, interval: float = 30, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lon_radec = self.visible_lon_grid_radec(np.arange(0, 360, interval), **kw)
        lat_radec = self.visible_lat_grid_radec(np.arange(-90, 90, interval), **kw)
        return lon_radec + lat_radec

    # State
    def _state_from_targvec(
        self, targvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        state, lt = spice.spkcpt(
            trgpos=targvec,
            trgctr=self._target_encoded,  # type: ignore
            trgref=self._target_frame_encoded,  # type: ignore
            et=self.et,
            outref=self._observer_frame_encoded,  # type: ignore
            refloc='OBSERVER',
            abcorr=self._aberration_correction_encoded,  # type: ignore
            obsrvr=self._observer_encoded,  # type: ignore
        )
        position = state[:3]
        velocity = state[3:]
        return position, velocity, lt

    def _radial_velocity_from_state(
        self, position: np.ndarray, velocity: np.ndarray, _lt: float | None = None
    ) -> float:
        # TODO note how lt argument is meaningless but there for convenience when
        # chaining with _state_from_targvec
        # dot the velocity with the normalised position vector to get radial component
        return velocity.dot(self.unit_vector(position))

    def _radial_velocity_from_targvec(self, targvec: np.ndarray) -> float:
        return self._radial_velocity_from_state(*self._state_from_targvec(targvec))

    def radial_velocity_from_lonlat(self, lon: float, lat: float) -> float:
        return self._radial_velocity_from_targvec(self.lonlat2targvec(lon, lat))

    # Description
    def get_description(self, newline: bool = True) -> str:
        return '{t} ({tid}){nl}from {o}{nl}at {d}'.format(
            t=self.target,
            tid=self.target_body_id,
            nl=('\n' if newline else ' '),
            o=self.observer,
            d=self.dtm.strftime('%Y-%m-%d %H:%M %Z'),
        )

    # Plotting
    def get_poles_to_plot(self) -> list[tuple[float, float, str]]:
        """
        Get list of poles for a plot.

        If at least one pole is visible, return the visible poles.
        If no poles are visible, return both poles in brackets.
        """
        poles: list[tuple[float, float, str]] = []
        pole_options = ((0, 90, 'N'), (0, -90, 'S'))
        for lon, lat, s in pole_options:
            if self.test_if_lonlat_visible(lon, lat):
                poles.append((lon, lat, s))

        if len(poles) == 0:
            # if no poles are visible, show both
            for lon, lat, s in pole_options:
                poles.append((lon, lat, f'({s})'))

        return poles

    def _plot_wireframe(
        self, transform: None | Transform, ax: Axes | None = None
    ) -> Axes:
        """Plot generic wireframe representation of the observation"""
        if ax is None:
            fig, ax = plt.subplots()

        if transform is None:
            transform = ax.transData
        else:
            transform = transform + ax.transData

        for ra, dec in self.visible_latlon_grid_radec(30):
            ax.plot(ra, dec, color='silver', linestyle=':', transform=transform)

        ax.plot(*self.limb_radec(), color='k', linewidth=0.5, transform=transform)
        ax.plot(
            *self.terminator_radec(),
            color='k',
            linestyle='--',
            transform=transform,
        )

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination()
        ax.plot(ra_day, dec_day, color='k', transform=transform)

        for lon, lat, s in self.get_poles_to_plot():
            ra, dec = self.lonlat2radec(lon, lat)
            ax.text(
                ra,
                dec,
                s,
                ha='center',
                va='center',
                weight='bold',
                color='grey',
                path_effects=[
                    path_effects.Stroke(linewidth=3, foreground='w'),
                    path_effects.Normal(),
                ],
                transform=transform,
                clip_on=True,
            )
        ax.set_title(self.get_description(newline=True))
        return ax

    def plot_wireframe_radec(self, ax: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot basic wireframe representation of the observation
        """
        ax = self._plot_wireframe(transform=None, ax=ax)

        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_aspect(1 / np.cos(self._target_dec_radians), adjustable='datalim')
        ax.invert_xaxis()

        if show:
            plt.show()
        return ax


class BodyXY(Body):
    def __init__(
        self,
        target: str,
        utc: str | datetime.datetime,
        nx: int = 0,
        ny: int = 0,
        *,
        sz: int | None = None,
        **kw,
    ) -> None:
        if sz is not None:
            if nx != 0 or ny != 0:
                raise ValueError('`sz` cannot be used if `nx` and/or `ny` are nonzero')
            nx = sz
            ny = sz

        super().__init__(target, utc, **kw)

        self._cache: dict[str, Any] = {}

        self._nx: int = nx
        self._ny: int = ny

        self._x0: float = 0
        self._y0: float = 0
        self._r0: float = 10
        self._rotation_radians: float = 0
        self.set_disc_method('default')
        self._default_disc_method = 'manual'

        if self._nx > 0 and self._ny > 0:
            # centre disc if dimensions provided
            self._x0 = self._nx / 2 - 0.5
            self._y0 = self._ny / 2 - 0.5
            self._r0 = 0.9 * (min(self._x0, self._y0))

        self._matplotlib_transform: matplotlib.transforms.Affine2D | None = None
        self._matplotlib_transform_radians: matplotlib.transforms.Affine2D | None = None

        self.backplanes: dict[str, Backplane] = {}
        self._register_default_backplanes()

    def __repr__(self) -> str:
        return f'BodyXY({self.target!r}, {self.utc!r}, {self._nx!r}, {self._ny!r})'

    # Cache management
    @staticmethod
    def cache_result(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def decorated(self, *args, **kwargs):
            k = fn.__name__
            if k not in self._cache:
                self._cache[k] = fn(self, *args, **kwargs)  #  type: ignore
            return self._cache[k]

        return decorated  # type: ignore

    def clear_cache(self):
        self._cache.clear()

    # Coordinate transformations
    @cache_result
    def _get_xy2radec_matrix_radians(self) -> np.ndarray:
        r_km = self.r_eq
        r_radians = np.arcsin(r_km / self.target_distance)

        s = r_radians / self.get_r0()

        theta = self._get_rotation_radians()

        stretch_matrix = np.array(
            [[-1 / np.abs(np.cos(self._target_dec_radians)), 0], [0, 1]]
        )
        rotation_matrix = self.rotation_matrix_radians(theta)
        transform_matrix_2x2 = s * np.matmul(rotation_matrix, stretch_matrix)

        v0 = np.array([self.get_x0(), self.get_y0()])
        a0 = np.array([self._target_ra_radians, self._target_dec_radians])
        offset_vector = a0 - np.matmul(transform_matrix_2x2, v0)

        transform_matrix_3x3 = np.identity(3)
        transform_matrix_3x3[:2, :2] = transform_matrix_2x2
        transform_matrix_3x3[:2, 2] = offset_vector

        return transform_matrix_3x3

    @cache_result
    def _get_radec2xy_matrix_radians(self) -> np.ndarray:
        return np.linalg.inv(self._get_xy2radec_matrix_radians())

    def _xy2radec_radians(self, x: float, y: float) -> tuple[float, float]:
        a = self._get_xy2radec_matrix_radians().dot(np.array([x, y, 1]))
        return a[0], a[1]

    def _radec2xy_radians(self, ra: float, dec: float) -> tuple[float, float]:
        v = self._get_radec2xy_matrix_radians().dot(np.array([ra, dec, 1]))
        return v[0], v[1]

    @staticmethod
    def rotation_matrix_radians(theta: float) -> np.ndarray:
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    # Composite transformations
    def xy2radec(self, x: float, y: float) -> tuple[float, float]:
        return self._radian_pair2degrees(*self._xy2radec_radians(x, y))

    def radec2xy(self, ra: float, dec: float) -> tuple[float, float]:
        return self._radec2xy_radians(*self._degree_pair2radians(ra, dec))

    def xy2lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self.radec2lonlat(*self.xy2radec(x, y))

    def lonlat2xy(self, lon: float, lat: float) -> tuple[float, float]:
        return self.radec2xy(*self.lonlat2radec(lon, lat))

    def _radec_arrs2xy_arrs(
        self, ra_arr: np.ndarray, dec_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = zip(*[self.radec2xy(r, d) for r, d in zip(ra_arr, dec_arr)])
        return np.array(x), np.array(y)

    def _xy2targvec(self, x: float, y: float) -> np.ndarray:
        return self._obsvec_norm2targvec(
            self._radec2obsvec_norm_radians(*self._xy2radec_radians(x, y))
        )

    # Interface
    def set_params(
        self,
        x0: float | None = None,
        y0: float | None = None,
        r0: float | None = None,
        rotation: float | None = None,
    ) -> None:
        if x0 is not None:
            self.set_x0(x0)
        if y0 is not None:
            self.set_y0(y0)
        if r0 is not None:
            self.set_r0(r0)
        if rotation is not None:
            self.set_rotation(rotation)

    def set_x0(self, x0: float) -> None:
        self._x0 = x0
        self.clear_cache()

    def get_x0(self) -> float:
        return self._x0

    def set_y0(self, y0: float) -> None:
        self._y0 = y0
        self.clear_cache()

    def get_y0(self) -> float:
        return self._y0

    def set_r0(self, r0: float) -> None:
        self._r0 = r0
        self.clear_cache()

    def get_r0(self) -> float:
        return self._r0

    def _set_rotation_radians(self, rotation: float) -> None:
        self._rotation_radians = rotation % (2 * np.pi)
        self.clear_cache()

    def _get_rotation_radians(self) -> float:
        return self._rotation_radians

    def set_rotation(self, rotation_degrees: float) -> None:
        self._set_rotation_radians(np.deg2rad(rotation_degrees))

    def get_rotation(self) -> float:
        return np.rad2deg(self._get_rotation_radians())

    def set_img_size(self, nx: int | None = None, ny: int | None = None) -> None:
        if nx is not None:
            self._nx = nx
        if ny is not None:
            self._ny = ny
        self.clear_cache()

    def get_img_size(self) -> tuple[int, int]:
        return (self._nx, self._ny)

    def set_disc_method(self, method: str):
        # Save disc method to the cahce. It will then be wiped automatically whenever
        # the disc is moved. The key used in the cache contains a space, so will never
        # collide with an auto-generated key from a function name (when @cache_result is
        # used).
        self._cache['disc method'] = method

    def get_disc_method(self) -> str:
        return self._cache.get('disc method', self._default_disc_method)

    # Illumination functions etc. # TODO remove these?
    def limb_xy(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radec_arrs2xy_arrs(*self.limb_radec(**kw))

    def limb_xy_by_illumination(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination(**kw)
        return (
            *self._radec_arrs2xy_arrs(ra_day, dec_day),
            *self._radec_arrs2xy_arrs(ra_night, dec_night),
        )

    def terminator_xy(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radec_arrs2xy_arrs(*self.terminator_radec(**kw))

    def visible_latlon_grid_xy(
        self, *args, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            self._radec_arrs2xy_arrs(*np.deg2rad(rd))
            for rd in self.visible_latlon_grid_radec(*args, **kw)
        ]

    # Matplotlib transforms
    def _get_matplotlib_radec2xy_transform_radians(
        self,
    ) -> matplotlib.transforms.Affine2D:
        if self._matplotlib_transform_radians is None:
            self._matplotlib_transform_radians = matplotlib.transforms.Affine2D(
                self._get_radec2xy_matrix_radians()
            )
        return self._matplotlib_transform_radians

    def get_matplotlib_radec2xy_transform(self) -> matplotlib.transforms.Affine2D:
        if self._matplotlib_transform is None:
            transform_rad2deg = matplotlib.transforms.Affine2D().scale(np.deg2rad(1))
            self._matplotlib_transform = (
                transform_rad2deg + self._get_matplotlib_radec2xy_transform_radians()
            )  #  type: ignore
        return self._matplotlib_transform  #  type: ignore

    def update_transform(self) -> None:
        self._get_matplotlib_radec2xy_transform_radians().set_matrix(
            self._get_radec2xy_matrix_radians()
        )

    # Plotting
    def plot_wireframe_xy(self, ax: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot basic wireframe representation of the observation
        """
        # Generate affine transformation from radec in degrees -> xy
        transform = self.get_matplotlib_radec2xy_transform()
        ax = self._plot_wireframe(transform=transform, ax=ax)

        if self._test_if_img_size_valid():
            ax.set_xlim(-0.5, self._nx - 0.5)
            ax.set_ylim(-0.5, self._ny - 0.5)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_aspect(1, adjustable='box')

        if show:
            plt.show()
        return ax

    # Coordinate images
    def _test_if_img_size_valid(self) -> bool:
        return (self._nx > 0) and (self._ny > 0)

    def _make_empty_img(self, nz: int | None = None) -> np.ndarray:
        if not self._test_if_img_size_valid():
            raise ValueError('nx and ny must be positive to create a backplane image')
        if nz is None:
            shape = (self._ny, self._nx)
        else:
            shape = (self._ny, self._nx, nz)
        return np.full(shape, np.nan)

    def _get_max_pixel_radius(self) -> float:
        # r0 corresponds to r_eq, but for the radius here we want to make sure we have
        # the largest radius
        r = self.get_r0() * max(self.radii) / self.r_eq
        return r

    @cache_result
    def _get_targvec_img(self) -> np.ndarray:
        out = self._make_empty_img(3)

        # Precalculate short circuit stuff here for speed
        r_cutoff = self._get_max_pixel_radius() * 1.05 + 1
        r2 = r_cutoff**2  # square here to save having to run sqrt every loop
        x0 = self.get_x0()
        y0 = self.get_y0()

        for y, x in self._iterate_yx():
            if (
                self._optimize_speed
                and ((x - x0) * (x - x0) + (y - y0) * (y - y0)) > r2
            ):
                # The spice calculations in _xy2targvec are slow, so to optimize speed
                # we can skip the spice calculation step completely for pixels which we
                # know aren't on the disc, by calculating if the distance from the
                # centre of the disc (x0,y0) is greater than the radius (+ a buffer).
                # The distance calculation uses manual multiplication rather than using
                # power (e.g. (x - x0)**2) to square the x and y distances as this is
                # faster.
                continue

            try:
                targvec = self._xy2targvec(x, y)
                out[y, x] = targvec
            except NotFoundError:
                continue  # leave values as nan if pixel is not on the disc
        return out

    def _iterate_yx(self) -> Iterable[tuple[int, int]]:
        for y in range(self._ny):
            for x in range(self._nx):
                yield y, x

    def _enumerate_targvec_img(self) -> Iterable[tuple[int, int, np.ndarray]]:
        targvec_img = self._get_targvec_img()
        for y, x in self._iterate_yx():
            targvec = targvec_img[y, x]
            if math.isnan(targvec[0]):
                # only check if first element nan for efficiency
                continue
            yield y, x, targvec

    @cache_result
    def _get_lonlat_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self._targvec2lonlat_radians(targvec)
        return np.rad2deg(out)

    def get_lon_img(self) -> np.ndarray:
        return self._get_lonlat_img()[:, :, 0]

    def get_lat_img(self) -> np.ndarray:
        return self._get_lonlat_img()[:, :, 1]

    @cache_result
    def _get_radec_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x in self._iterate_yx():
            out[y, x] = self._xy2radec_radians(x, y)
        return np.rad2deg(out)

    def get_ra_img(self) -> np.ndarray:
        return self._get_radec_img()[:, :, 0]

    def get_dec_img(self) -> np.ndarray:
        return self._get_radec_img()[:, :, 1]

    @cache_result
    def _get_illumination_gie_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self._illumination_angles_from_targvec_radians(targvec)
        return np.rad2deg(out)

    def get_phase_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 0]

    def get_incidence_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 1]

    def get_emission_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 2]

    @cache_result
    def _get_state_imgs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_img = self._make_empty_img(3)
        velocity_img = self._make_empty_img(3)
        lt_img = self._make_empty_img()
        for y, x, targvec in self._enumerate_targvec_img():
            (
                position_img[y, x],
                velocity_img[y, x],
                lt_img[y, x],
            ) = self._state_from_targvec(targvec)
        return position_img, velocity_img, lt_img

    def get_distance_img(self) -> np.ndarray:
        position_img, velocity_img, lt_img = self._get_state_imgs()
        return lt_img * spice.clight()

    @cache_result
    def get_radial_velocity_img(self) -> np.ndarray:
        out = self._make_empty_img()
        position_img, velocity_img, lt_img = self._get_state_imgs()
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self._radial_velocity_from_state(
                position_img[y, x], velocity_img[y, x]
            )
        return out

    def get_doppler_img(self) -> np.ndarray:
        return self.get_radial_velocity_img() / spice.clight()

    # Backplane management
    @staticmethod
    def standardise_backplane_name(name: str) -> str:
        return name.strip().upper()

    def register_backplane(
        self, fn: Callable[[], np.ndarray], name: str, description: str
    ) -> None:
        # TODO add checks for name/description lengths?
        name = self.standardise_backplane_name(name)
        if name in self.backplanes:
            raise ValueError(f'Backplane named {name!r} is already registered')
        self.backplanes[name] = Backplane(name=name, description=description, fn=fn)

    def _register_default_backplanes(self) -> None:
        # TODO double check units and expand descriptions
        self.register_backplane(self.get_lon_img, 'LON', 'Longitude [deg]')
        self.register_backplane(self.get_lat_img, 'LAT', 'Latitude [deg]')
        self.register_backplane(self.get_ra_img, 'RA', 'Right ascension [deg]')
        self.register_backplane(self.get_dec_img, 'DEC', 'Declination [deg]')
        self.register_backplane(self.get_phase_angle_img, 'phase', 'Phase angle [deg]')
        self.register_backplane(
            self.get_incidence_angle_img, 'INCIDENCE', 'Incidence angle [deg]'
        )
        self.register_backplane(
            self.get_emission_angle_img, 'EMISSION', 'Emission angle [dec]'
        )
        self.register_backplane(
            self.get_distance_img, 'DISTANCE', 'Distance [km] from observer'
        )
        self.register_backplane(
            self.get_radial_velocity_img,
            'RADIAL_VELOCITY',
            'Radial velocity [km/s] relative to observer',
        )
        self.register_backplane(
            self.get_doppler_img,
            'DOPPLER',
            '(Radial velocity)/(speed of light) [dimensionless]',
        )

    def get_backplane_img(self, name: str) -> np.ndarray:
        name = self.standardise_backplane_name(name)
        return self.backplanes[name].fn()

    def plot_backplane(
        self, name: str, ax: Axes | None = None, show: bool = True, **kw
    ) -> Axes:
        name = self.standardise_backplane_name(name)
        backplane = self.backplanes[name]
        ax = self.plot_wireframe_xy(ax, show=False)
        im = ax.imshow(backplane.fn(), origin='lower', **kw)
        plt.colorbar(im, label=backplane.description)
        if show:
            plt.show()
        return ax


class Observation(BodyXY):
    FITS_FILE_EXTENSIONS = ('.fits', '.fits.gz')
    IMAGE_FILE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    FITS_KEYWORD = 'PLANMAP'

    def __init__(
        self,
        path: str | None = None,
        *,
        data: np.ndarray | None = None,
        header: fits.Header | None = None,
        **kw,
    ) -> None:
        self.path = path
        self.header: fits.Header = None  # type: ignore
        self.data: np.ndarray

        # TODO add warning about header being modified in place? Or copy header?
        if self.path is None:
            if data is None:
                raise ValueError('Either `path` or `data` must be provided')
            self.data = data
        else:
            if data is not None:
                raise ValueError('`path` and `data` are mutually exclusive')
            if header is not None:
                raise ValueError('`path` and `header` are mutually exclusive')
            self._load_data_from_path()

        # TODO validate/standardise shape of data here (cube etc.)
        self.data = np.asarray(self.data)
        if self.header is not None:
            # use values from header to fill in arguments (e.g. target) which aren't
            # specified by the user
            self._add_kw_from_header(kw)
        super().__init__(nx=self.data.shape[2], ny=self.data.shape[1], **kw)

        if self.header is None:
            self.header = fits.Header(
                {
                    'OBJECT': self.target,
                    'DATE-OBS': self.utc,
                }
            )
        self.centre_disc()

    def _load_data_from_path(self):
        assert self.path is not None
        if any(self.path.endswith(ext) for ext in self.FITS_FILE_EXTENSIONS):
            self._load_fits_data()
        elif any(self.path.endswith(ext) for ext in self.IMAGE_FILE_EXTENSIONS):
            self._load_image_data()
        else:
            raise ValueError(f'Unexpected file type for {self.path!r}')

    def _load_fits_data(self):
        assert self.path is not None
        self.data, self.header = fits.getdata(self.path, header=True)  #  type: ignore
        # TODO add check data is a cube

    def _load_image_data(self):
        assert self.path is not None
        image = np.array(PIL.Image.open(self.path))

        if len(image.shape) == 2:
            # If greyscale image, add another dimension so that it is an image cube with
            # a single frame. This will ensure that data will always be a cube.
            image = np.array([image])
        else:
            # If RGB image, change the axis order so wavelength is the first axis (i.e.
            # consistent with FITS)
            image = np.moveaxis(image, 2, 0)
        self.data = image

    def _add_kw_from_header(self, kw: dict):
        # fill in kwargs with values from header (if they aren't specified by the user)
        # TODO deal with more FITS files (e.g. DATE-OBS doesn't work for JWST)
        # TODO deal with missing values
        kw.setdefault('target', self.header['OBJECT'])
        kw.setdefault('utc', self.header['DATE-OBS'])

    def __repr__(self) -> str:
        return f'Observation({self.path!r})'  # TODO make more explicit?

    # Auto disc id
    def centre_disc(self) -> None:
        """Centre disc and make it fill ~90% of the observation"""
        self.set_x0(self._nx / 2)
        self.set_y0(self._ny / 2)
        self.set_r0(0.9 * (min(self.get_x0(), self.get_y0())))
        self.set_disc_method('centre_disc')

    # Output
    def append_to_header(
        self,
        keyword: str,
        value: str | float | bool | complex,
        comment: str | None = None,
        hierarch_keyword: bool = True,
    ):
        if hierarch_keyword:
            keyword = f'HIERARCH {self.FITS_KEYWORD} {keyword}'
        with warnings.catch_warnings():
            # Suppress warning about comments being truncated
            warnings.filterwarnings(
                'ignore',
                message='Card is too long, comment will be truncated.',
                module='astropy.io.fits.card',
            )
            self.header.append(fits.Card(keyword=keyword, value=value, comment=comment))

    def add_header_metadata(self):
        self.append_to_header('VERSION', __version__, 'Planet Mapper version.')
        self.append_to_header('URL', __url__, 'Webpage.')
        self.append_to_header(
            'DATE',
            datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'File generation datetime.',
        )
        if self.path is not None:
            self.append_to_header(
                'INFILE',
                os.path.split(self.path)[1],
                'Input file name.',
            )
        self.append_to_header(
            'DISC X0', self.get_x0(), '[pixels] x coordinate of disc centre.'
        )
        self.append_to_header(
            'DISC Y0', self.get_y0(), '[pixels] y coordinate of disc centre.'
        )
        self.append_to_header(
            'DISC R0', self.get_r0(), '[pixels] equatorial radius of disc.'
        )
        self.append_to_header(
            'DISC ROT', self.get_rotation(), '[degrees] rotation of disc.'
        )
        self.append_to_header(
            'DISC METHOD', self.get_disc_method(), 'Method used to find disc.'
        )
        self.append_to_header(
            'ET-OBS', self.et, 'J2000 ephemeris seconds of observation.'
        )
        self.append_to_header(
            'TARGET',
            self.target,
            'Target body name used in SPICE.',
        )
        self.append_to_header(
            'TARGET-ID', self.target_body_id, 'Target body ID from SPICE.'
        )
        self.append_to_header(
            'R EQ', self.r_eq, '[km] Target equatorial radius from SPICE.'
        )
        self.append_to_header(
            'R POLAR', self.r_polar, '[km] Target polar radius from SPICE.'
        )
        self.append_to_header(
            'LIGHT-TIME',
            self.target_light_time,
            '[seconds] Light time to target from SPICE.',
        )
        self.append_to_header(
            'DISTANCE', self.target_distance, '[km] Distance to target from SPICE.'
        )
        self.append_to_header(
            'OBSERVER',
            self.observer,
            'Observer name used in SPICE.',
        )
        self.append_to_header(
            'TARGET-FRAME',
            self.target_frame,
            'Target frame used in SPICE.',
        )
        self.append_to_header(
            'OBSERVER-FRAME',
            self.observer_frame,
            'Observer frame used in SPICE.',
        )
        self.append_to_header(
            'ILLUMINATION',
            self.illumination_source,
            'Illumination source used in SPICE.',
        )
        self.append_to_header(
            'ABCORR', self.aberration_correction, 'Aberration correction used in SPICE.'
        )
        self.append_to_header(
            'SUBPOINT-METHOD', self.subpoint_method, 'Subpoint method used in SPICE.'
        )
        self.append_to_header(
            'SURFACE-METHOD',
            self.surface_method,
            'Surface intercept method used in SPICE.',
        )
        self.append_to_header(
            'OPTIMIZATION-USED', self._optimize_speed, 'Speed optimizations used.'
        )

    def save(self, path: str) -> None:
        """Save fits file with backplanes"""
        self.add_header_metadata()

        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdul = fits.HDUList([hdu])

        for name, backplane in self.backplanes.items():
            utils.print_progress(name)
            img = backplane.fn()
            header = fits.Header([('ABOUT', backplane.description)])
            header.add_comment('Backplane generated by Planet Mapper software.')
            hdu = fits.ImageHDU(data=img, header=header, name=name)
            hdul.append(hdu)
        hdul.writeto(path, overwrite=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
