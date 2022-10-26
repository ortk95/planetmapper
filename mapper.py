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
import datetime
import glob
import os
import sys
from typing import Callable, Iterable, TypeVar, ParamSpec, cast, Any
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.transforms import Transform
from matplotlib.axes import Axes
import numpy as np
import spiceypy as spice
import utils

__version__ = '0.1'

Numeric = TypeVar('Numeric', bound=float | np.ndarray)
T = TypeVar('T')
P = ParamSpec('P')

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
    o = Observation('jupiter', dtm)

    o.set_x0(10)
    o.set_y0(10)
    o.set_r0(9)
    o.set_rotation_degrees(10)

    ax = o.plot_wirefeame_xy(show=False)
    im = ax.imshow(
        o.get_phase_angle_img_degrees(), origin='lower', zorder=0, cmap='turbo'
    )
    plt.colorbar(im)
    plt.show()


class Body:
    """
    Class representing spice data about an observation of an astronomical body.
    """

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
        self.dtm = self.et2dtm(self.et)
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
        self.target_light_time = cast(float, lt)
        self.target_distance = self.target_light_time * spice.clight()
        # cast() calls are only here to make type checking play nicely with spice.spkezr
        self.target_ra, self.target_dec = self.obsvec2radec(self._target_obsvec)

        # Find sub observer point
        self._subpoint_targvec, self._subpoint_et, self._subpoint_rayvec = spice.subpnt(
            self.subpoint_method,
            self.target,
            self.et,
            self.target_frame,
            self.aberration_correction,
            self.observer,
        )
        self.subpoint_distance = np.linalg.norm(self._subpoint_rayvec)
        self.subpoint_lon, self.subpoint_lat = self.targvec2lonlat(
            self._subpoint_targvec
        )
        self._subpoint_obsvec = self.rayvec2obsvec(
            self._subpoint_rayvec, self._subpoint_et
        )
        self.subpoint_ra, self.subpoint_dec = self.obsvec2radec(self._subpoint_obsvec)

    def __repr__(self) -> str:
        return f'Body({self.target!r}, {self.utc!r})'

    # Coordinate transformations target -> observer direction
    def lonlat2targvec(self, lon: float, lat: float) -> np.ndarray:
        """
        Transform lon/lat coordinates on body to rectangular vector in target frame.
        """
        return spice.pgrrec(self.target, lon, lat, 0, self.r_eq, self.flattening)

    def targvec2obsvec(self, targvec: np.ndarray) -> np.ndarray:
        """
        Transform rectangular vector in target frame to rectangular vector in observer
        frame.
        """
        # Get the target vector from the subpoint to the point of interest
        targvec_offset = targvec - self._subpoint_targvec

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
            self.target_frame, self.observer_frame, targvec_et, self.et
        )

        # Use the transform matrix to perform the actual transformation
        return self._subpoint_obsvec + np.matmul(transform_matrix, targvec_offset)

    def rayvec2obsvec(self, rayvec: np.ndarray, et: float) -> np.ndarray:
        """
        Transform rectangular vector from point to observer in target frame to
        rectangular vector of point in observer frame.
        """
        px = spice.pxfrm2(self.target_frame, self.observer_frame, et, self.et)
        return np.matmul(px, rayvec)

    def obsvec2radec(self, obsvec: np.ndarray) -> tuple[float, float]:
        """
        Transform rectangular vector in observer frame to observer ra/dec coordinates.
        """
        dst, ra, dec = spice.recrad(obsvec)
        return ra, dec

    # Coordinate transformations observer -> target direction
    def radec2obsvec_norm(self, ra: float, dec: float) -> np.ndarray:
        return spice.radrec(1, ra, dec)

    def obsvec_norm2targvec(self, obsvec_norm: np.ndarray) -> np.ndarray:
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

    def targvec2lonlat(self, targvec: np.ndarray) -> tuple[float, float]:
        lon, lat, alt = spice.recpgr(self.target, targvec, self.r_eq, self.flattening)
        return lon, lat

    # Useful transformations (built from combinations of above transformations)
    def lonlat2radec(self, lon: float, lat: float) -> tuple[float, float]:
        return self.obsvec2radec(
            self.targvec2obsvec(
                self.lonlat2targvec(lon, lat),
            )
        )

    def lonlat2radec_degrees(self, lon: float, lat: float) -> tuple[float, float]:
        return self._radian_pair2degrees(
            *self.lonlat2radec(*self._degree_pair2radians(lon, lat))
        )

    def radec2lonlat(
        self, ra: float, dec: float, not_found_nan: bool = True
    ) -> tuple[float, float]:
        try:
            ra, dec = self.targvec2lonlat(
                self.obsvec_norm2targvec(
                    self.radec2obsvec_norm(ra, dec),
                )
            )
        except spice.stypes.NotFoundError:
            if not_found_nan:
                ra = np.nan
                dec = np.nan
            else:
                raise
        return ra, dec

    def radec2lonlat_degrees(self, ra: float, dec: float, **kw) -> tuple[float, float]:
        return self._radian_pair2degrees(
            *self.radec2lonlat(*self._degree_pair2radians(ra, dec), **kw)
        )

    def _targvec_arr2radec_arrs(
        self,
        targvec_arr: np.ndarray | list[np.ndarray],
        condition_func: None | Callable[[np.ndarray], bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if condition_func is not None:
            ra_dec = [
                self.obsvec2radec(self.targvec2obsvec(t))
                if condition_func(t)
                else (np.nan, np.nan)
                for t in targvec_arr
            ]
        else:
            ra_dec = [self.obsvec2radec(self.targvec2obsvec(t)) for t in targvec_arr]
        ra = np.array([r for r, d in ra_dec])
        dec = np.array([d for r, d in ra_dec])
        return ra, dec

    # Other spice methods
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
        return self._illumination_angles_from_targvec(self.lonlat2targvec(lon, lat))

    def illumination_angles_from_lonlat_degrees(
        self, lon: float, lat: float
    ) -> tuple[float, float, float]:
        phase, incdnc, emissn = self.illumination_angles_from_lonlat(
            *self._degree_pair2radians(lon, lat)
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
            self.target_distance,
            npts,
        )
        if close_loop:
            points = self.close_loop(points)
        return points

    def limb_radec(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        targvec_arr = self._limb_targvec(**kw)
        return self._targvec_arr2radec_arrs(targvec_arr)

    def limb_radec_degrees(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radian_pair2degrees(*self.limb_radec(**kw))

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

    def limb_radec_by_illumination_degrees(
        self, **kw
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination(**kw)
        return (
            *self._radian_pair2degrees(ra_day, dec_day),
            *self._radian_pair2degrees(ra_night, dec_night),
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

    def terminator_radec_degrees(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self._radian_pair2degrees(*self.terminator_radec(**kw))

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
        return self._test_if_targvec_visible(self.lonlat2targvec(lon, lat))

    def test_if_lonlat_visible_degrees(self, lon: float, lat: float) -> bool:
        return self.test_if_lonlat_visible(*self._degree_pair2radians(lon, lat))

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
        return self._test_if_targvec_illuminated(self.lonlat2targvec(lon, lat))

    def test_if_lonlat_illuminated_degrees(self, lon: float, lat: float) -> bool:
        return self.test_if_lonlat_illuminated(*self._degree_pair2radians(lon, lat))

    def visible_lon_grid_radec(
        self, lons: list[float] | np.ndarray, npts: int = 90
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lats = np.deg2rad(np.linspace(-90, 90, npts))
        out = []
        for lon in lons:
            targvecs = [self.lonlat2targvec(lon, lat) for lat in lats]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lon_grid_radec_degrees(
        self, lons: list[float] | np.ndarray, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out = self.visible_lon_grid_radec(np.deg2rad(lons), **kw)
        return [self._radian_pair2degrees(*radec) for radec in out]

    def visible_lat_grid_radec(
        self, lats: list[float] | np.ndarray, npts: int = 180
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lons = np.deg2rad(np.linspace(0, 360, npts))
        out = []
        for lat in lats:
            targvecs = [self.lonlat2targvec(lon, lat) for lon in lons]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lat_grid_radec_degrees(
        self, lats: list[float] | np.ndarray, **kw
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out = self.visible_lat_grid_radec(np.deg2rad(lats), **kw)
        return [self._radian_pair2degrees(*radec) for radec in out]

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

    def _state_from_targvec(
        self, targvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        state, lt = spice.spkcpt(
            trgpos=targvec,
            trgctr=self.target,
            trgref=self.target_frame,
            et=self.et,
            outref=self.observer_frame,
            refloc='OBSERVER',
            abcorr=self.aberration_correction,
            obsrvr=self.observer,
        )
        position = state[:3]
        velocity = state[3:]
        return position, velocity, lt

    def _radial_velocity_from_targvec(self, targvec: np.ndarray) -> float:
        position, velocity, lt = self._state_from_targvec(targvec)
        # dot the velocity with the normalised position vector to get radial component
        radial_velocity = np.dot(velocity, spice.vhat(position))
        return radial_velocity

    def radial_velocity_from_lonlat(self, lon: float, lat: float) -> float:
        return self._radial_velocity_from_targvec(self.lonlat2targvec(lon, lat))

    def radial_velocity_from_lonlat_degrees(self, lon: float, lat: float) -> float:
        return self.radial_velocity_from_lonlat(*self._degree_pair2radians(lon, lat))

    # Utility methods
    def standardise_body_name(self, name: str) -> str:
        name = spice.bodc2s(spice.bods2c(name))
        return name

    def et2dtm(self, et: float) -> datetime.datetime:
        s = spice.et2utc(et, 'ISOC', 6) + '+0000'
        # manually add '+0000' to string to make it timezone aware
        # i.e. this lets python know it is UTC
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')

    def _get_poles_to_plot(self) -> list[tuple[float, float, str]]:
        """
        Get list of poles for a plot.

        If at least one pole is visible, return the visible poles.
        If no poles are visible, return both poles in brackets.
        """
        poles: list[tuple[float, float, str]] = []
        pole_options = ((0, 90, 'N'), (0, -90, 'S'))
        for lon, lat, s in pole_options:
            if self.test_if_lonlat_visible_degrees(lon, lat):
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

        for ra, dec in self.visible_latlon_grid_radec_degrees(30):
            ax.plot(ra, dec, color='silver', linestyle=':', transform=transform)

        ax.plot(
            *self.limb_radec_degrees(), color='k', linewidth=0.5, transform=transform
        )
        ax.plot(
            *self.terminator_radec_degrees(),
            color='k',
            linestyle='--',
            transform=transform,
        )

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination_degrees()
        ax.plot(ra_day, dec_day, color='k', transform=transform)

        for lon, lat, s in self._get_poles_to_plot():
            ra, dec = self.lonlat2radec_degrees(lon, lat)
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
            )
        ax.set_title(self.get_description(newline=True))
        return ax

    def plot_wirefeame_radec(self, ax: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot basic wireframe representation of the observation
        """
        ax = self._plot_wireframe(transform=None, ax=ax)

        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_aspect(1 / np.cos(self.target_dec), adjustable='datalim')
        ax.invert_xaxis()

        if show:
            plt.show()
        return ax

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


class Observation(Body):
    def __init__(self, target: str, utc: str | datetime.datetime, *args, **kw) -> None:
        super().__init__(target, utc, *args, **kw)

        self.nx: int = 25
        self.ny: int = 20

        self._x0: float = 0
        self._y0: float = 0
        self._r0: float = 10
        self._rotation: float = 0

        self._cache: dict[str, Any] = {}
        self._matplotlib_transform: matplotlib.transforms.Affine2D | None = None

    def __repr__(self) -> str:
        return f'Observation({self.target!r}, {self.utc!r})'

    # Cache management
    @staticmethod
    def cache_result(fn: Callable[P, T]) -> Callable[P, T]:
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
    def get_xy2radec_matrix(self) -> np.ndarray:
        # a = M*v + a0 - M*v0

        r_km = self.r_eq
        r_radians = r_km / self.target_distance  # TODO do this better?

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

    @cache_result
    def get_radec2xy_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.get_xy2radec_matrix())

    def xy2radec(self, x: float, y: float) -> tuple[float, float]:
        a = self.get_xy2radec_matrix() @ np.array([x, y, 1])
        return a[0], a[1]

    def radec2xy(self, ra: float, dec: float) -> tuple[float, float]:
        v = self.get_radec2xy_matrix() @ np.array([ra, dec, 1])
        return v[0], v[1]

    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    # Composite transformations
    def xy2lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self.radec2lonlat(*self.xy2radec(x, y))

    def xy2lonlat_degrees(self, x: float, y: float) -> tuple[float, float]:
        return self._radian_pair2degrees(*self.xy2lonlat(x, y))

    def lonlat2xy(self, lon: float, lat: float) -> tuple[float, float]:
        return self.radec2xy(*self.lonlat2radec(lon, lat))

    def lonlat2xy_degrees(self, lon: float, lat: float) -> tuple[float, float]:
        return self.lonlat2xy(*self._degree_pair2radians(lon, lat))

    def _radec_arrs2xy_arrs(
        self, ra_arr: np.ndarray, dec_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = zip(*[self.radec2xy(r, d) for r, d in zip(ra_arr, dec_arr)])
        return np.array(x), np.array(y)

    def _xy2targvec(self, x: float, y: float) -> np.ndarray:
        return self.obsvec_norm2targvec((self.radec2obsvec_norm(*self.xy2radec(x, y))))

    # Interface
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

    def set_rotation(self, rotation: float) -> None:
        self._rotation = rotation % (2 * np.pi)
        self.clear_cache()

    def get_rotation(self) -> float:
        return self._rotation

    def set_rotation_degrees(self, rotation_degrees: float) -> None:
        self.set_rotation(np.deg2rad(rotation_degrees))

    def get_rotation_degrees(self) -> float:
        return np.rad2deg(self.get_rotation())

    # Illumination functions etc.
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
            for rd in self.visible_latlon_grid_radec_degrees(*args, **kw)
        ]

    # Other
    def get_matplotlib_radec2xy_transform(self) -> matplotlib.transforms.Affine2D:
        if self._matplotlib_transform is None:
            self._matplotlib_transform = matplotlib.transforms.Affine2D(
                self.get_radec2xy_matrix()
            )
        return self._matplotlib_transform

    def update_transform(self) -> None:
        self.get_matplotlib_radec2xy_transform().set_matrix(self.get_radec2xy_matrix())

    def plot_wirefeame_xy(self, ax: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot basic wireframe representation of the observation
        """
        # Generate affine transformation from radec in degrees -> xy
        transform_rad2deg = matplotlib.transforms.Affine2D().scale(np.deg2rad(1))
        transform = transform_rad2deg + self.get_matplotlib_radec2xy_transform()

        ax = self._plot_wireframe(transform=transform, ax=ax)

        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_aspect(1, adjustable='datalim')

        if show:
            plt.show()
        return ax

    # Coordinate images
    def _make_empty_img(self, nz: int | None = None) -> np.ndarray:
        if nz is None:
            shape = (self.ny, self.nx)
        else:
            shape = (self.ny, self.nx, nz)
        return np.full(shape, np.nan)

    @cache_result
    def _get_targvec_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        for y, x in self._iterate_yx():
            try:
                targvec = self._xy2targvec(x, y)
                out[y, x] = targvec
            except spice.stypes.NotFoundError:
                # leave values as nan if pixel is not on the disc
                continue
        return out

    def _iterate_yx(self) -> Iterable[tuple[int, int]]:
        for y in range(self.ny):
            for x in range(self.nx):
                yield y, x

    def _enumerate_targvec_img(self) -> Iterable[tuple[int, int, np.ndarray]]:
        targvec_img = self._get_targvec_img()
        for y, x in self._iterate_yx():
            targvec = targvec_img[y, x]
            if np.isnan(targvec[0]):
                # only check if first element nan for efficiency
                continue
            yield y, x, targvec

    @cache_result
    def _get_lonlat_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self.targvec2lonlat(targvec)
        return out

    def get_lon_img(self) -> np.ndarray:
        return self._get_lonlat_img()[:, :, 0]

    def get_lat_img(self) -> np.ndarray:
        return self._get_lonlat_img()[:, :, 1]

    def get_lon_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_lon_img())

    def get_lat_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_lat_img())

    @cache_result
    def get_radial_velocity_img(self) -> np.ndarray:
        """abc"""
        out = self._make_empty_img()
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self._radial_velocity_from_targvec(targvec)
        return out

    @cache_result
    def _get_radec_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x in self._iterate_yx():
            out[y, x] = self.xy2radec(x, y)
        return out

    def get_ra_img(self) -> np.ndarray:
        return self._get_radec_img()[:, :, 0]

    def get_dec_img(self) -> np.ndarray:
        return self._get_radec_img()[:, :, 1]

    def get_ra_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_ra_img())

    def get_dec_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_dec_img())

    @cache_result
    def _get_illumination_gie_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        for y, x, targvec in self._enumerate_targvec_img():
            out[y, x] = self._illumination_angles_from_targvec(targvec)
        return out

    def get_phase_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 0]

    def get_incidence_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 1]

    def get_emission_angle_img(self) -> np.ndarray:
        return self._get_illumination_gie_img()[:, :, 2]

    def get_phase_angle_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_phase_angle_img())

    def get_incidence_angle_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_incidence_angle_img())

    def get_emission_angle_img_degrees(self) -> np.ndarray:
        return np.rad2deg(self.get_emission_angle_img())


if __name__ == '__main__':
    main(*sys.argv[1:])
