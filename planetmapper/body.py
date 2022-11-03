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
from functools import wraps
import PIL.Image
from astropy.io import fits
from . import utils
from . import common
from .mapper import MapperTool

class Body(MapperTool):
    """
    Class representing an astronomical body observed at a specific time.

    Generally only `target`, `utc` and `observer` need to be changed. The additional
    parameters allow customising the exact settings used in the internal SPICE
    functions. Similarly, some methods (e.g. :func:`terminator_radec`) have parameters
    that are passed to SPICE functions which can almost always be left as their default
    values.

    This class inherits from :class:`MapperTool` so the methods described above are also
    available.

    Args:
        target: Name of target body.
        utc: Time of observation. This can be any string datetime representation
            compatible with SPICE (e.g. '2000-12-31T23:59:59') or a Python datetime
            object. The time is assumed to be UTC unless otherwise specified (e.g. by
            using a timezone aware Python datetime). For the string formats accepted by
            SPICE, see
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html.
        observer: Name of observing body. Defaults to 'EARTH'.
        observer_frame: Observer reference frame.
        illumination_source: Illumination source (e.g. the sun).
        aberration_correction: Aberration correction used to correct light travel time
            in SPICE.
        subpoint_method: Method used to calculate the sub-observer point in SPICE.
        surface_method: Method used to calculate surface intercepts in SPICE.
        **kwargs: Additional arguments are passed to `MapperTool`.
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Document instance variables
        self.et: float
        """Ephemeris time of the observation corresponding to `utc`."""
        self.dtm: datetime.datetime
        """Python timezone aware datetime of the observation corresponding to `utc`."""
        self.target_body_id: int
        """SPICE numeric ID of the target body."""
        self.r_eq: float
        """Equatorial radius of the target body in km."""
        self.r_polar: float
        """Polar radius of the target body in km."""
        self.flattening: float
        """Flattening of target body, calculated as `(r_eq - r_polar) / r_eq`."""
        self.target_light_time: float
        """Light time from the target to the observer at the time of the observation."""
        self.target_distance: float
        """Distance from the target to the observer at the time of the observation."""
        self.target_ra: float
        """Right ascension (RA) of the target centre."""
        self.target_dec: float
        """Declination (Dec) of the target centre."""
        self.subpoint_distance: float
        """Distance from the observer to the sub-observer point on the target."""
        self.subpoint_lon: float
        """Longitude of the sub-observer point on the target."""
        self.subpoint_lat: float
        """Latitude of the sub-observer point on the target."""

        # Process inputs
        self.target = self.standardise_body_name(target)
        if isinstance(utc, datetime.datetime):
            # convert input datetime to UTC, then to a string compatible with spice
            utc = utc.astimezone(datetime.timezone.utc)
            utc = utc.strftime(self._DEFAULT_DTM_FORMAT_STRING)
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
        self._aberration_correction_encoded = self._encode_str(
            self.aberration_correction
        )
        self._subpoint_method_encoded = self._encode_str(self.subpoint_method)
        self._surface_method_encoded = self._encode_str(self.surface_method)

        # Get target properties and state
        self.et = spice.utc2et(self.utc)
        self.dtm: datetime.datetime = self.et2dtm(self.et)
        self.target_body_id: int = spice.bodn2c(self.target)
        self.target_frame = 'IAU_' + self.target
        self._target_frame_encoded = self._encode_str(self.target_frame)

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
        self._subpoint_ra, self._subpoint_dec = self._radian_pair2degrees(
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
        """
        Convert longitude/latitude coordinates on the target body to RA/Dec sky
        coordinates for the observer.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point.
        """
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

    def radec2lonlat(
        self,
        ra: float,
        dec: float,
        not_found_nan: bool = True,
    ) -> tuple[float, float]:
        """
        Convert RA/Dec sky coordinates for the observer to longitude/latitude
        coordinates on the target body.

        The provided RA/Dec will not necessarily correspond to any longitude/latitude
        coordinates, as they could be 'missing' the target and instead be observing the
        background sky. In this case, the returned longitude/latitude values will be NaN
        if `not_found_nan` is True (the default) or this function will raise an error if
        `not_found_nan` is False.

        Args:
            ra: Right ascension of point in the sky of the observer.
            dec: Declination of point in the sky of the observer.
            not_found_nan: Controls behaviour when the input `ra` and `dec` coordinates
                are missing the target body.

        Returns:
            `(lon, lat)` tuple containing the longitude/latitude coordinates on the
            target body. If the provided RA/Dec coordinates are missing the target body
            and `not_found_nan` is True, then the `lon` and `lat` values will both be
            NaN.

        Raises:
            NotFoundError: If the provided RA/Dec coordinates are missing the target
                body and `not_found_nan` is False, then NotFoundError will be raised.
        """
        return self._radian_pair2degrees(
            *self._radec2lonlat_radians(
                *self._degree_pair2radians(ra, dec), not_found_nan=not_found_nan
            )
        )

    def lonlat2targvec(self, lon: float, lat: float) -> np.ndarray:
        """
        Convert longitude/latitude coordinates on the target body to rectangular vector
        centred in the target frame (e.g. for use as an input to a SPICE function).

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            Numpy array corresponding to the 3D rectangular vector describing the
            longitude/latitude point in the target frame of reference.
        """
        return self._lonlat2targvec_radians(*self._degree_pair2radians(lon, lat))

    def targvec2lonlat(self, targvec: np.ndarray) -> tuple[float, float]:
        """
        Convert rectangular vector centred in the target frame to longitude/latitude
        coordinates on the target body (e.g. to convert the output from a SPICE
        function).

        Args:
            targvec: 3D rectangular vector in the target frame of reference.

        Returns:
            `(lon, lat)` tuple containing the longitude and latitude corresponding to
            the input vector.
        """
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

    def limb_radec(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the RA/Dec coordinates of the target body's limb.

        Args:
            npts: Number of points in the generated limb.

        Returns:
            `(ra, dec)` tuple of coordinate arrays.
        """
        return self._targvec_arr2radec_arrs(self._limb_targvec(**kwargs))

    def limb_radec_by_illumination(
        self, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate RA/Dec coordinates of the dayside and nightside parts of the target
        body's limb.

        Output arrays are like the outputs of `limb_radec`, but the dayside coordinate
        arrays have non-illuminated locations replaced with NaN and the nightside arrays
        have illuminated locations replaced with NaN.

        Args:
            npts: Number of points in the generated limbs.

        Returns:
            `(ra_day, dec_day, ra_night, dec_night)` tuple of coordinate arrays of the
            dayside then nightside parts of the limb.
        """
        targvec_arr = self._limb_targvec(**kwargs)
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
        """
        Test if longitude/latitude coordinate on the target body are visible.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            True if the point is visible from the observer, otherwise False.
        """
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
        """
        Calculate the illumination angles of a longitude/latitude coordinate on the
        target body.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            `(phase, incidence, emission)` tuple containing the illumination angles.
        """
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
        """
        Calculate the RA/Dec coordinates of the terminator (line between day and night)
        on the target body. By default, only the visible part of the terminator is
        returned (this can be changed with `only_visible`).

        Args:
            npts: Number of points in generated terminator.
            only_visible: Toggle only returning visible part of terminator.
            close_loop: If True, passes coordinate arrays through :func:`close_loop`
                (e.g. to enable nicer plotting).
            method, corloc: Passed to SPICE function.

        Returns:
            `(ra, dec)` tuple of RA/Dec coordinate arrays.
        """
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
        """
        Test if longitude/latitude coordinate on the target body are illuminated.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            True if the point is illuminated, otherwise False.
        """
        return self._test_if_targvec_illuminated(self.lonlat2targvec(lon, lat))

    # Lonlat grid
    def visible_latlon_grid_radec(
        self, interval: float = 30, **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Convenience function to calculate a grid of equally spaced lines of constant
        longitude and latitude for use in plotting lon/lat grids.

        This function effectively combines :func:`visible_lon_grid_radec` and
        :func:`visible_lat_grid_radec` to produce both longitude and latitude gridlines.

        For example, to plot gridlines with a 45 degree interval, use::

            lines = body.visible_latlon_grid_radec(interval=45)
            for ra, dec in lines:
                plt.plot(ra, dec)

        Args:
            interval: Spacing of gridlines. Generally, this should be an integer factor
                of 90 to produce nice looking plots (e.g. 10, 30, 45 etc).
            **kwargs: Additional arguments are passed to :func:`visible_lon_grid_radec` and
                :func:`visible_lat_grid_radec`.

        Returns:
            List of `(ra, dec)` tuples, each of which corresponds to a gridline. `ra`
            and `dec` are arrays of RA/Dec coordinate values for that gridline.
        """

        lon_radec = self.visible_lon_grid_radec(np.arange(0, 360, interval), **kwargs)
        lat_radec = self.visible_lat_grid_radec(np.arange(-90, 90, interval), **kwargs)
        return lon_radec + lat_radec

    def visible_lon_grid_radec(
        self, lons: list[float] | np.ndarray, npts: int = 50
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Calculates the RA/Dec coordinates for visible lines of constant longitude.

        For each longitude in `lons`, a `(ra, dec)` tuple is calculated which contains
        arrays of RA and Dec coordinates. Coordinates which correspond to points which
        are not visible are replaced with NaN.

        See also :func:`visible_latlon_grid_radec`,

        Args:
            lons: List of longitudes to plot.
            npts: Number of points in each full line of constant longitude.

        Returns:
            List of `(ra, dec)` tuples, corresponding to the list of input `lons`. `ra`
            and `dec` are arrays of RA/Dec coordinate values for that gridline.
        """
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
        """
        Constant latitude version of :func:`visible_lon_grid_radec`. See also
        :func:`visible_latlon_grid_radec`.

        Args:
            lats: List of latitudes to plot.
            npts: Number of points in each full line of constant latitude.

        Returns:
            List of `(ra, dec)` tuples, corresponding to the list of input `lats`. `ra`
            and `dec` are arrays of RA/Dec coordinate values for that gridline.
        """
        lons = np.linspace(0, 360, npts)
        out = []
        for lat in lats:
            targvecs = [self.lonlat2targvec(lon, lat) for lon in lons]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

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
        """
        Calculate radial (i.e. line-of-sight) velocity of a point on the target's
        surface relative to the observer. This can be used to calculate the doppler
        shift.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            Radial velocity of the point in km/s.
        """
        return self._radial_velocity_from_targvec(self.lonlat2targvec(lon, lat))

    # Description
    def get_description(self, multiline: bool = True) -> str:
        """
        Generate a useful description of the body.

        Args:
            multiline: Toggles between multi-line and single-line version of the
                description.

        Returns:
            String describing the observation of the body.
        """
        return '{t} ({tid}){nl}from {o}{nl}at {d}'.format(
            t=self.target,
            tid=self.target_body_id,
            nl=('\n' if multiline else ' '),
            o=self.observer,
            d=self.dtm.strftime('%Y-%m-%d %H:%M %Z'),
        )

    # Plotting
    def get_poles_to_plot(self) -> list[tuple[float, float, str]]:
        """
        Get list of poles on the target body for use in plotting.

        If at least one pole is visible, return the visible poles. If no poles are
        visible, return both poles but in brackets. This ensures that at lease one pole
        is always returned (to orientate the observation).

        Returns:
            List of `(ra, dec, label)` tuples describing the poles where `ra` and `dec`
            give the RA/Dec coordinates of the pole in the sky and `label` is a string
            describing the pole. If the pole is visible, the `label` is either 'N' or
            'S'. If neither pole is visible, then both poles are returned with labels
            of '(N)' and '(S)'.
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
        ax.set_title(self.get_description(multiline=True))
        return ax

    def plot_wireframe_radec(self, ax: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot basic wireframe representation of the observation using RA/Dec sky
        coordinates.

        Args:
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), then
                a new figure and axis is created.
            show: Toggle showing the plotted figure with `plt.show()`

        Returns:
            The axis containing the plotted wireframe.
        """
        ax = self._plot_wireframe(transform=None, ax=ax)

        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_aspect(1 / np.cos(self._target_dec_radians), adjustable='datalim')
        ax.invert_xaxis()

        if show:
            plt.show()
        return ax

