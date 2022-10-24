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
"""
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
import numpy as np
from functools import cached_property
import tkinter as tk
from tkinter import ttk
from typing import Callable
import itertools
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spiceypy as spice
import glob
from typing import TypeVar, cast, TypeAlias, NewType, Protocol
import datetime
import matplotlib.transforms

__version__ = '1.0a1'

Widget = TypeVar('Widget', bound=tk.Widget)
Numeric = TypeVar('Numeric', bound=float | np.ndarray)

KERNEL_PATH = '~/spice/naif/generic_kernels/'


"""TODO
- Make option stirngs 'CN' etc. customisable
- Make stuff work with generic observer location
- Replace all EARTH references with self.observer
- Standardise rad/deg coordinates for longlat/radec
- Add non-affine transformations
- Add non body centre observer locations?
- Ring system
- Mark custom ra/dec and lon/lat
- Click (/mouseover?) image to get values
"""


def main(*args):
    dtm = datetime.datetime.now()
    # o = Observation('moon', dtm)
    # o.rotation = np.pi / 2
    # print(o.flattening)
    # o.plot_wirefeame_xy()
    # o.plot_wirefeame_radec()

    io = InteractiveObservation()
    io.run()


class XyToRadec(Protocol):
    def __call__(self, x: float, y: float) -> tuple[float, float]:
        ...


class RadecToXy(Protocol):
    def __call__(self, ra: float, dec: float) -> tuple[float, float]:
        ...


class SpiceBody:
    """
    Class representing spice data about an observation of an astronomical body.
    """

    C_LIGHT = 299792.458  # Light speed in km/s
    DEFAULT_DTM_FORMAT_STRING = '%Y-%m-%dT%H:%M:%S.%f'

    def __init__(
        self,
        target: str,
        utc: str | datetime.datetime,
        observer: str = 'EARTH',
        observer_frame: str = 'J2000',
        aberration_correction: str = 'CN',
        subpoint_method: str = 'INTERCEPT/ELLIPSOID',
        surface_method: str = 'ELLIPSOID',
        load_kernels: bool = True,
        kernel_path: str = KERNEL_PATH,
        manual_kernels: None | list[str] = None,
    ) -> None:
        self.target = self.standardise_body_name(target)
        if isinstance(utc, datetime.datetime):
            # convert input datetime to UTC, then to a string compatible with spice
            utc = utc.astimezone(datetime.timezone.utc)
            utc = utc.strftime(self.DEFAULT_DTM_FORMAT_STRING)
        self.utc = utc
        self.observer = self.standardise_body_name(observer)
        self.observer_frame = observer_frame
        self.aberration_correction = aberration_correction
        self.subpoint_method = subpoint_method
        self.surface_method = surface_method

        if load_kernels:
            self.load_spice_kernels(
                kernel_path=kernel_path, manual_kernels=manual_kernels
            )

        # Get target properties and state
        self.et = spice.utc2et(self.utc)
        self.dtm = self.et_to_dtm(self.et)
        self.target_frame = 'IAU_' + self.target
        self.target_body_id: int = spice.bodn2c(self.target)

        self.radii = spice.bodvar(self.target_body_id, 'RADII', 3)
        self.r_eq = self.radii[0]
        self.r_polar = self.radii[2]
        self.flattening = (self.r_eq - self.r_polar) / self.r_eq

        starg, lt = spice.spkezr(
            target,
            self.et,
            self.observer_frame,
            self.aberration_correction,
            self.observer,
        )
        self._target_obsvec = cast(np.ndarray, starg)[:3]
        self.light_time = cast(float, lt)
        # cast() calls are only here to make type checking play nicely with spice.spkezr
        self.target_ra, self.target_dec = self.obsvec_to_radec(self._target_obsvec)

        # Find sub observer point
        self._subpoint_targvec, self._subpoint_et, self._subpoint_rayvec = spice.subpnt(
            self.subpoint_method,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
        )
        self.subpoint_lon, self.subpoint_lat = self.targvec_to_lonlat(
            self._subpoint_targvec
        )
        self._subpoint_obsvec = self.rayvec_to_obsvec(
            self._subpoint_rayvec, self._subpoint_et
        )
        self.subpoint_ra, self.subpoint_dec = self.obsvec_to_radec(
            self._subpoint_obsvec
        )

    def __repr__(self) -> str:
        return f'Body({self.target!r}, {self.utc!r})'

    # Coordinate transformations target -> observer direction
    def lonlat_to_targvec(self, lon: float, lat: float) -> np.ndarray:
        """
        Transform lon/lat coordinates on body to rectangular vector in target frame.
        """
        return spice.pgrrec(self.target, lon, lat, 0, self.r_eq, self.flattening)

    def targvec_to_obsvec(self, targvec: np.ndarray) -> np.ndarray:
        """
        Transform rectangular vector in target frame to rectangular vector in observer
        frame.
        """
        # Difference in the LOS distance from the sub-obs point to this point
        dist = np.linalg.norm(
            self._subpoint_rayvec - self._subpoint_targvec + targvec
        ) - np.linalg.norm(self._subpoint_rayvec)
        ep = self._subpoint_et - dist / self.C_LIGHT

        # Transform to the J2000 frame corresponding to when the ray hit the detector
        px = spice.pxfrm2(self.target_frame, self.observer_frame, ep, self.et)
        return self._subpoint_obsvec + np.matmul(px, targvec - self._subpoint_targvec)

    def rayvec_to_obsvec(self, rayvec: np.ndarray, et: float) -> np.ndarray:
        """
        Transform rectangular vector from point to observer in target frame to
        rectangular vector of point in observer frame.
        """
        px = spice.pxfrm2(self.target_frame, self.observer_frame, et, self.et)
        return np.matmul(px, rayvec)

    def obsvec_to_radec(self, obsvec: np.ndarray) -> tuple[float, float]:
        """
        Transform rectangular vector in observer frame to observer ra/dec coordinates.
        """
        dst, ra, dec = spice.recrad(obsvec)
        return ra, dec

    # Coordinate transformations observer -> target direction
    def radec_to_obsvec_norm(self, ra: float, dec: float) -> np.ndarray:
        return spice.radrec(1, ra, dec)

    def obsvec_norm_to_targvec(self, obsvec_norm: np.ndarray) -> np.ndarray:
        """TODO add note about raising NotFoundError"""
        spoint, trgepc, srfvec = spice.sincpt(
            self.surface_method,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
            self.observer_frame,
            obsvec_norm,
        )
        return spoint

    def targvec_to_lonlat(self, targvec: np.ndarray) -> tuple[float, float]:
        lon, lat, alt = spice.recpgr(self.target, targvec, self.r_eq, self.flattening)
        return lon, lat

    # Useful transformations (built from combinations of above transformations)
    def lonlat_to_radec(self, lon: float, lat: float) -> tuple[float, float]:
        return self.obsvec_to_radec(
            self.targvec_to_obsvec(
                self.lonlat_to_targvec(lon, lat),
            )
        )

    def lonlat_to_radec_degrees(self, lon: float, lat: float) -> tuple[float, float]:
        return self._radian_pair_to_degrees(
            *self.lonlat_to_radec(*self._degree_pair_to_radians(lon, lat))
        )

    def radec_to_lonlat(
        self, ra: float, dec: float, not_found_nan: bool = True
    ) -> tuple[float, float]:
        try:
            ra, dec = self.targvec_to_lonlat(
                self.obsvec_norm_to_targvec(
                    self.radec_to_obsvec_norm(ra, dec),
                )
            )
        except spice.stypes.NotFoundError:
            if not_found_nan:
                ra = np.nan
                dec = np.nan
            else:
                raise
        return ra, dec

    def radec_to_lonlat_degrees(
        self, ra: float, dec: float, **kw
    ) -> tuple[float, float]:
        return self._radian_pair_to_degrees(
            *self.radec_to_lonlat(*self._degree_pair_to_radians(ra, dec), **kw)
        )

    def _targvec_arr_to_radec_arrs(
        self,
        targvec_arr: np.ndarray | list[np.ndarray],
        condition_func: None | Callable[[np.ndarray], bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if condition_func is not None:
            ra_dec = [
                self.obsvec_to_radec(self.targvec_to_obsvec(t))
                if condition_func(t)
                else (np.nan, np.nan)
                for t in targvec_arr
            ]
        else:
            ra_dec = [
                self.obsvec_to_radec(self.targvec_to_obsvec(t)) for t in targvec_arr
            ]
        ra = np.array([r for r, d in ra_dec])
        dec = np.array([d for r, d in ra_dec])
        return ra, dec

    # Illumination/visibility methods
    def _illumination_angles_from_targvec(
        self, targvec: np.ndarray
    ) -> tuple[float, float, float]:
        trgepc, srfvec, phase, incdnc, emissn = spice.ilumin(
            self.surface_method,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
            targvec,
        )
        return phase, incdnc, emissn

    def illumination_angles_from_lonlat(
        self, lon: float, lat: float
    ) -> tuple[float, float, float]:
        return self._illumination_angles_from_targvec(self.lonlat_to_targvec(lon, lat))

    def illumination_angles_from_lonlat_degrees(
        self, lon: float, lat: float
    ) -> tuple[float, float, float]:
        phase, incdnc, emissn = self.illumination_angles_from_lonlat(
            *self._degree_pair_to_radians(lon, lat)
        )
        return np.deg2rad(phase), np.deg2rad(incdnc), np.deg2rad(emissn)

    def _limb_targvec(
        self,
        npts: int = 360,
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
            self.light_time * self.C_LIGHT,
            npts,
        )
        if close_loop:
            points = self.close_loop(points)
        return points

    def limb_radec(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        targvec_arr = self._limb_targvec(**kw)
        return self._targvec_arr_to_radec_arrs(targvec_arr)

    def limb_radec_degrees(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radian_pair_to_degrees(*self.limb_radec(**kw))

    def limb_radec_by_illumination(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        targvec_arr = self._limb_targvec(**kw)
        ra_day, dec_day = self._targvec_arr_to_radec_arrs(targvec_arr)
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

    def limb_radec_by_illumination_degrees(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination(**kw)
        return (
            *self._radian_pair_to_degrees(ra_day, dec_day),
            *self._radian_pair_to_degrees(ra_night, dec_night),
        )

    def terminator_radec(
        self,
        npts: int = 360,
        only_visible: bool = True,
        close_loop: bool = True,
        method: str = 'UMBRAL/TANGENT/ELLIPSOID',
        corloc: str = 'ELLIPSOID TERMINATOR',
    ) -> tuple[np.ndarray, np.ndarray]:
        refvec = [0, 0, 1]
        rolstp = 2 * np.pi / npts
        _, targvec_arr, epochs, trmvcs = spice.termpt(
            method,
            'SUN',
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
            self.light_time * self.C_LIGHT,
            npts,
        )
        if close_loop:
            targvec_arr = self.close_loop(targvec_arr)
        ra, dec = self._targvec_arr_to_radec_arrs(
            targvec_arr,
            condition_func=self._test_if_targvec_visible if only_visible else None,
        )
        return ra, dec

    def terminator_radec_degrees(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radian_pair_to_degrees(*self.terminator_radec(**kw))

    def _test_if_targvec_visible(self, targvec: np.ndarray) -> bool:
        trgepc, srfvec, phase, incdnc, emissn, visibl, lit = spice.illumf(
            self.surface_method,
            self.target,
            'SUN',
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
            targvec,
        )
        return visibl

    def test_if_lonlat_visible(self, lon: float, lat: float) -> bool:
        return self._test_if_targvec_visible(self.lonlat_to_targvec(lon, lat))

    def test_if_lonlat_visible_degrees(self, lon: float, lat: float) -> bool:
        return self.test_if_lonlat_visible(*self._degree_pair_to_radians(lon, lat))

    def _test_if_targvec_illuminated(self, targvec: np.ndarray) -> bool:
        trgepc, srfvec, phase, incdnc, emissn, visibl, lit = spice.illumf(
            self.surface_method,
            self.target,
            'SUN',
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
            targvec,
        )
        return lit

    def test_if_lonlat_illuminated(self, lon: float, lat: float) -> bool:
        return self._test_if_targvec_illuminated(self.lonlat_to_targvec(lon, lat))

    def test_if_lonlat_illuminated_degrees(self, lon: float, lat: float) -> bool:
        return self.test_if_lonlat_illuminated(*self._degree_pair_to_radians(lon, lat))

    def visible_lon_grid_radec(
        self, lons: list[float] | np.ndarray, npts: int = 90
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lats = np.deg2rad(np.linspace(-90, 90, npts))
        out = []
        for lon in lons:
            targvecs = [self.lonlat_to_targvec(lon, lat) for lat in lats]
            ra, dec = self._targvec_arr_to_radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lon_grid_radec_degrees(
        self, lons: list[float] | np.ndarray, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out = self.visible_lon_grid_radec(np.deg2rad(lons), **kw)
        return [self._radian_pair_to_degrees(*radec) for radec in out]

    def visible_lat_grid_radec(
        self, lats: list[float] | np.ndarray, npts: int = 180
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lons = np.deg2rad(np.linspace(0, 360, npts))
        out = []
        for lat in lats:
            targvecs = [self.lonlat_to_targvec(lon, lat) for lon in lons]
            ra, dec = self._targvec_arr_to_radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lat_grid_radec_degrees(
        self, lats: list[float] | np.ndarray, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out = self.visible_lat_grid_radec(np.deg2rad(lats), **kw)
        return [self._radian_pair_to_degrees(*radec) for radec in out]

    def visible_latlon_grid_radec_degrees(
        self, interval: float = 30, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lon_radec = self.visible_lon_grid_radec_degrees(
            np.arange(0, 360, interval), **kw
        )
        lat_radec = self.visible_lat_grid_radec_degrees(
            np.arange(-90, 90, interval), **kw
        )
        return lon_radec + lat_radec

    # Utility methods
    def standardise_body_name(self, name: str) -> str:
        return name.upper().strip()

    def et_to_dtm(self, et: float) -> datetime.datetime:
        s = spice.et2utc(et, 'ISOC', 6) + '+0000'
        # manually add '+0000' to string to make it timezone aware
        # i.e. this lets python know it is UTC
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')

    def plot_wirefeame_radec(self) -> None:
        """
        Plot basic wireframe representation of the observation
        """
        fig, ax = plt.subplots()

        ax.plot(*self.limb_radec_degrees(), color='k', linewidth=0.5)
        ax.plot(*self.terminator_radec_degrees(), color='k', linestyle='--')

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination_degrees()
        ax.plot(ra_day, dec_day, color='k')

        for ra, dec in self.visible_latlon_grid_radec_degrees(30):
            ax.plot(ra, dec, color='silver', linestyle=':', zorder=0)

        for lon, lat, s in ((0, 90, 'N'), (0, -90, 'S')):
            if self.test_if_lonlat_visible_degrees(lon, lat):
                ra, dec = self.lonlat_to_radec_degrees(lon, lat)
                ax.annotate(
                    s,
                    (ra, dec),
                    ha='center',
                    va='center',
                    weight='bold',
                    color='grey',
                    path_effects=[
                        path_effects.Stroke(linewidth=3, foreground='w'),
                        path_effects.Normal(),
                    ],
                )

        ax.set_title(self.get_description(newline=True))

        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')

        ax.set_aspect(1 / np.cos(self.target_dec), adjustable='datalim')
        ax.invert_xaxis()

        plt.show()

    def get_description(self, newline: bool = True) -> str:
        return '{t} ({tid}){nl}from {o} at {d}'.format(
            t=self.target,
            tid=self.target_body_id,
            nl=('\n' if newline else ' '),
            o=self.observer,
            d=self.dtm.strftime('%Y-%m-%d %H:%M %Z'),
        )

    @staticmethod
    def load_spice_kernels(
        kernel_path: str = KERNEL_PATH, manual_kernels: None | list[str] = None
    ) -> None:
        if manual_kernels:
            kernels = manual_kernels
        else:
            kernel_path = os.path.expanduser(kernel_path)
            pcks = sorted(glob.glob(kernel_path + 'pck/*.tpc'))
            spks1 = sorted(glob.glob(kernel_path + 'spk/planets/de*.bsp'))
            spks2 = sorted(glob.glob(kernel_path + 'spk/satellites/*.bsp'))
            fks = sorted(glob.glob(kernel_path + 'fk/planets/*.tf'))
            lsks = sorted(glob.glob(kernel_path + 'lsk/naif*.tls'))
            kernels = [pcks[-1], spks1[-1], *spks2, lsks[-1]]
        for kernel in kernels:
            spice.furnsh(kernel)

    @staticmethod
    def get_angular_dist(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        return np.arccos(
            np.sin(dec1) * np.sin(dec2)
            + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        )

    @staticmethod
    def close_loop(arr: np.ndarray) -> np.ndarray:
        return np.append(arr, [arr[0]], axis=0)

    @staticmethod
    def _radian_pair_to_degrees(
        radians0: Numeric, radians1: Numeric
    ) -> tuple[Numeric, Numeric]:
        return np.rad2deg(radians0), np.rad2deg(radians1)  # type: ignore

    @staticmethod
    def _degree_pair_to_radians(
        degrees0: Numeric, degrees1: Numeric
    ) -> tuple[Numeric, Numeric]:
        return np.deg2rad(degrees0), np.deg2rad(degrees1)  # type: ignore


class Observation(SpiceBody):
    def __init__(self, target: str, utc: str | datetime.datetime, *args, **kw) -> None:
        super().__init__(target, utc, *args, **kw)

        self._x0: float = 0
        self._y0: float = 0
        self._r0: float = 10
        self._rotation: float = 0

        self._xy_to_radec_matrix: np.ndarray | None = None
        self._radec_to_xy_matrix: np.ndarray | None = None
        self._transform: matplotlib.transforms.Affine2D | None = None

    # Coordinate transformations
    def set_transformations(
        self, xy_to_radec: XyToRadec, radec_to_xy: RadecToXy
    ) -> None:
        self.xy_to_radec = xy_to_radec
        self.radec_to_xy = radec_to_xy

    def set_dirty(self):
        self._xy_to_radec_matrix = None
        self._radec_to_xy_matrix = None

    def calculate_xy_to_radec_matrix(self) -> np.ndarray:
        # a = M*v + a0 - M*v0

        r_km = self.r_eq
        dist_km = self.light_time * self.C_LIGHT
        r_radians = r_km / dist_km  # TODO do this better

        s = r_radians / self.get_r0()

        theta = self.get_rotation()

        stretch_matrix = np.array([[-1 / np.abs(np.cos(self.target_dec)), 0], [0, 1]])
        rotation_matrix = self.rotation_matrix(theta)
        transform_matrix_2x2 = s * np.matmul(rotation_matrix, stretch_matrix)

        v0 = np.array([self.get_x0(), self.get_y0()])
        a0 = np.array([self.target_ra, self.target_dec])
        offset_vector = a0 - np.matmul(transform_matrix_2x2, v0)

        transform_matrix_3x3 = np.identity(3)
        transform_matrix_3x3[:2, :2] = transform_matrix_2x2
        transform_matrix_3x3[:2, 2] = offset_vector

        return transform_matrix_3x3

    def calculate_radec_to_xy_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.get_xy_to_radec_matrix())

    def get_xy_to_radec_matrix(self) -> np.ndarray:
        if self._xy_to_radec_matrix is None:
            self._xy_to_radec_matrix = self.calculate_xy_to_radec_matrix()
        return self._xy_to_radec_matrix

    def get_radec_to_xy_matrix(self) -> np.ndarray:
        if self._radec_to_xy_matrix is None:
            self._radec_to_xy_matrix = self.calculate_radec_to_xy_matrix()
        return self._radec_to_xy_matrix

    def xy_to_radec(self, x: float, y: float) -> tuple[float, float]:
        a = self.get_xy_to_radec_matrix() @ np.array([x, y, 1])
        return a[0], a[1]

    def radec_to_xy(self, ra: float, dec: float) -> tuple[float, float]:
        v = self.get_radec_to_xy_matrix() @ np.array([ra, dec, 1])
        return v[0], v[1]

    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    # Composite transformations
    def xy_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self.radec_to_lonlat(*self.xy_to_radec(x, y))

    def xy_to_lonlat_degrees(self, x: float, y: float) -> tuple[float, float]:
        return self._radian_pair_to_degrees(*self.xy_to_lonlat(x, y))

    def lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        return self.radec_to_xy(*self.lonlat_to_radec(lon, lat))

    def lonlat_to_xy_degrees(self, lon: float, lat: float) -> tuple[float, float]:
        return self.lonlat_to_xy(*self._degree_pair_to_radians(lon, lat))

    def _radec_arrs_to_xy_arrs(
        self, ra_arr: np.ndarray, dec_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = zip(*[self.radec_to_xy(r, d) for r, d in zip(ra_arr, dec_arr)])
        return np.array(x), np.array(y)

    # Interface
    def set_x0(self, x0: float) -> None:
        self._x0 = x0
        self.set_dirty()

    def get_x0(self) -> float:
        return self._x0

    def set_y0(self, y0: float) -> None:
        self._y0 = y0
        self.set_dirty()

    def get_y0(self) -> float:
        return self._y0

    def set_r0(self, r0: float) -> None:
        self._r0 = r0
        self.set_dirty()

    def get_r0(self) -> float:
        return self._r0

    def set_rotation(self, rotation: float) -> None:
        self._rotation = rotation % (2 * np.pi)
        self.set_dirty()

    def get_rotation(self) -> float:
        return self._rotation

    def set_rotation_degrees(self, rotation_degrees: float) -> None:
        self.set_rotation(np.deg2rad(rotation_degrees))

    def get_rotation_degrees(self) -> float:
        return np.rad2deg(self.get_rotation())

    # Illumination functions etc.
    def limb_xy(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radec_arrs_to_xy_arrs(*self.limb_radec(**kw))

    def limb_xy_by_illumination(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination(**kw)
        return (
            *self._radec_arrs_to_xy_arrs(ra_day, dec_day),
            *self._radec_arrs_to_xy_arrs(ra_night, dec_night),
        )

    def terminator_xy(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radec_arrs_to_xy_arrs(*self.terminator_radec(**kw))

    def visible_latlon_grid_xy(
        self, *args, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            self._radec_arrs_to_xy_arrs(*np.deg2rad(rd))
            for rd in self.visible_latlon_grid_radec_degrees(*args, **kw)
        ]

    # Other
    def get_matplotlib_radec_to_xy_transform(self) -> matplotlib.transforms.Affine2D:
        if self._transform is None:
            self._transform = matplotlib.transforms.Affine2D(
                self.get_radec_to_xy_matrix()
            )
        return self._transform

    def update_transform(self) -> None:
        self.get_matplotlib_radec_to_xy_transform().set_matrix(
            self.get_radec_to_xy_matrix()
        )

    def plot_wirefeame_xy(self) -> None:
        fig, ax = plt.subplots()

        ax.plot(*self.limb_xy(), color='k', linewidth=0.5)
        ax.plot(*self.terminator_xy(), color='k', linestyle='--')

        x_day, y_day, x_night, y_night = self.limb_xy_by_illumination()
        ax.plot(x_day, y_day, color='k')

        for x, y in self.visible_latlon_grid_xy(30):
            ax.plot(x, y, color='silver', linestyle=':', zorder=0)

        for lon, lat, s in ((0, 90, 'N'), (0, -90, 'S')):
            if self.test_if_lonlat_visible_degrees(lon, lat):
                x, y = self.lonlat_to_xy_degrees(lon, lat)
                ax.annotate(
                    s,
                    (x, y),
                    ha='center',
                    va='center',
                    weight='bold',
                    color='grey',
                    path_effects=[
                        path_effects.Stroke(linewidth=3, foreground='w'),
                        path_effects.Normal(),
                    ],
                )

        ax.set_title(self.get_description(newline=True))

        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        ax.set_aspect(1, adjustable='datalim')

        # # Testing transformations
        # ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination()
        # transform = matplotlib.transforms.Affine2D(
        #     self._transform_radec_to_xy
        # ) + ax.transData
        # ax.plot(ra_night, dec_night, color='r', transform=transform)

        # # Checking moon
        # circle = mpatch.Circle((0,0), 0.044, facecolor='none', edgecolor='r')
        # ax.add_patch(circle)
        plt.show()


class InteractiveObservation:
    def __init__(self) -> None:
        self.handles = []

        p = 'test/jupiter_test.jpg'
        self.observation = Observation('jupiter', datetime.datetime(2020, 8, 25, 12))
        self.image = np.flipud(plt.imread(p))

        self.observation.set_x0(self.image.shape[0] / 2)
        self.observation.set_y0(self.image.shape[1] / 2)
        self.observation.set_r0(self.image.shape[0] / 4)
        self.observation.set_rotation(0)

        self.step_size = 10

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

        fig = plt.figure(figsize=(5, 4), dpi=100)
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
                rot=self.observation.get_rotation_degrees(),
            )
        )

    def plot_wireframe(self) -> None:
        ax = self.ax
        transform = (
            self.observation.get_matplotlib_radec_to_xy_transform() + ax.transData
        )

        ax.plot(
            *self.observation.limb_radec(),
            color='w',
            linewidth=0.5,
            transform=transform,
            zorder=5,
        )
        ax.plot(
            *self.observation.terminator_radec(),
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
        ) = self.observation.limb_radec_by_illumination()
        ax.plot(ra_day, dec_day, color='w', transform=transform, zorder=5)

        for ra, dec in self.observation.visible_latlon_grid_radec_degrees(30):
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
            if self.observation.test_if_lonlat_visible_degrees(lon, lat):
                ra, dec = self.observation.lonlat_to_radec(
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
        self.observation.set_rotation_degrees(
            self.observation.get_rotation_degrees() - self.step_size
        )
        self.replot()

    def rotate_right(self) -> None:
        self.observation.set_rotation_degrees(
            self.observation.get_rotation_degrees() + self.step_size
        )
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
