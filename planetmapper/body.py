import datetime
from typing import Callable, Literal, cast

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import spiceypy as spice
from matplotlib.axes import Axes
from spiceypy.utils.exceptions import NotFoundError

from . import data_loader, utils
from .base import SpiceBase, Numeric


class Body(SpiceBase):
    """
    Class representing an astronomical body observed at a specific time.

    Generally only `target`, `utc` and `observer` need to be changed. The additional
    parameters allow customising the exact settings used in the internal SPICE
    functions. Similarly, some methods (e.g. :func:`terminator_radec`) have parameters
    that are passed to SPICE functions which can almost always be left as their default
    values.

    The `target` and `observer` names are passed to 
    :func:`SpiceBase.standardise_body_name`, so a variety of formats are acceptable. For
    example `'jupiter'`, `'JUPITER'`, `' Jupiter '`, `'599'` and `599` will
    all resolve to `'JUPITER'`.

    This class inherits from :class:`SpiceBase` so the methods described above are also
    available.

    Args:
        target: Name of target body.
        utc: Time of observation. This can be provided in a variety of formats and is
            assumed to be UTC unless otherwise specified. The accepted formats are: any
            `string` datetime representation compatible with SPICE (e.g.
            `'2000-12-31T23:59:59'` - see
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html
            for the acceptable string formats), a Python `datetime` object, or a `float`
            representing the Modified Julian Date (MJD) of the observation.
        observer: Name of observing body. Defaults to `'EARTH'`.
        observer_frame: Observer reference frame.
        illumination_source: Illumination source (e.g. the sun).
        aberration_correction: Aberration correction used to correct light travel time
            in SPICE.
        subpoint_method: Method used to calculate the sub-observer point in SPICE.
        surface_method: Method used to calculate surface intercepts in SPICE.
        **kwargs: Additional arguments are passed to :class:`SpiceBase`.
    """

    def __init__(
        self,
        target: str|int,
        utc: str | datetime.datetime | float,
        observer: str|int = 'EARTH',
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
        self.prograde: bool
        """Boolean indicating if the target's spin sense is prograde or retrograde."""
        self.positive_longitude_direction: Literal['E', 'W']
        """
        Positive direction of planetographic longitudes. `'W'` implies positive west
        planetographic longitudes and `'E'` implies positive east longitudes. 
        
        This is determined from the target's spin sense (i.e. from :attr:`prograde`), 
        with positive west longitudes for prograde rotation and positive east for 
        retrograde. The earth, moon and sun are exceptions to this and are defined to 
        have positive east longitudes
        
        For more details, see
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pgrrec_c.html#Particulars
        """
        self.target_light_time: float
        """Light time from the target to the observer at the time of the observation."""
        self.target_distance: float
        """Distance from the target to the observer at the time of the observation."""
        self.target_ra: float
        """Right ascension (RA) of the target centre."""
        self.target_diameter_arcsec: float
        """Equatorial angular diameter of the target in arcseconds."""
        self.target_dec: float
        """Declination (Dec) of the target centre."""
        self.subpoint_distance: float
        """Distance from the observer to the sub-observer point on the target."""
        self.subpoint_lon: float
        """Longitude of the sub-observer point on the target."""
        self.subpoint_lat: float
        """Latitude of the sub-observer point on the target."""
        self.ring_radii: set[float]
        """
        Set of ring raddii in km to plot around the target body's equator. Each radius
        is plottted as a single line, so for a wide ring you may want to add both the
        inner and outer edger of the ring. The radii are defined as the distance from
        the centre of the target body to the ring. For Saturn, the A, B and C rings from
        https://nssdc.gsfc.nasa.gov/planetary/factsheet/satringfact.html are included by
        default. For all other bodies, `ring_radii` is empty by default.
        
        See also :func:`ring_radec`.
        
        Example usage: ::

            body.ring_radii.add(122340) # Add new ring radius to plot
            body.ring_radii.add(136780) # Add new ring radius to plot

            body.ring_radii.remove(122340) # Remove a ring radius
            body.ring_radii.clear() # Remove all ring radii
        """
        self.coordinates_of_interest_lonlat: list[tuple[float, float]]
        """
        List of `(lon, lat)` coordinates of interest on the surface of the target body
        to mark when plotting (points which are not visible will not be plotted). To add
        a new point of interest, simply append a coordinate pair to this list: ::

            body.coordinates_of_interest_lonlat.append((0, -22))
        """
        self.coordinates_of_interest_radec: list[tuple[float, float]]
        """
        List of `(ra, dec)` sky coordinates of interest to mark when plotting (e.g. 
        background stars). To add new point of interest, simply append a coordinate pair
        to this list: ::

            body.coordinates_of_interest_radec.append((200, -45))
        """
        self.other_bodies_of_interest: list[Body]
        """
        List of other bodies of interest to mark when plotting. Add to this list using 
        :func:`add_other_bodies_of_interest`.
        """

        # Process inputs
        self.target = self.standardise_body_name(target)
        if isinstance(utc, float):
            utc = self.mjd2dtm(utc)
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

        # Use first degree term of prime meridian Euler angle to identify the spin sense
        # of the target body, then use this spin sense to determine positive longitude
        # direction (taking into account special cases)
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pgrrec_c.html#Particulars
        pm = spice.bodvar(self.target_body_id, 'PM', 3)
        self.prograde = pm[1] >= 0
        if self.prograde and self.target_body_id not in {10, 301, 399}:
            # {10, 301, 399} accounts for special cases of SUN, MOON and EARTH which are
            # positive east even though they are prograde.
            self.positive_longitude_direction = 'W'
        else:
            self.positive_longitude_direction = 'E'

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
        self.target_distance = self.target_light_time * self.speed_of_light()
        self._target_ra_radians, self._target_dec_radians = self._obsvec2radec_radians(
            self._target_obsvec
        )
        self.target_ra, self.target_dec = self._radian_pair2degrees(
            self._target_ra_radians, self._target_dec_radians
        )
        self.target_diameter_arcsec = (
            60 * 60 * np.rad2deg(np.arcsin(2 * self.r_eq / self.target_distance))
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

        # Create empty lists/blank values
        self.ring_radii = set()
        self.other_bodies_of_interest = []
        self.coordinates_of_interest_lonlat = []
        self.coordinates_of_interest_radec = []

        self._matrix_km2radec = None
        self._matrix_radec2km = None
        self._mpl_transform_km2radec_radians = None
        self._mpl_transform_radec2km_radians = None
        self._mpl_transform_km2radec = None
        self._mpl_transform_radec2km = None

        # Run custom setup
        if self.target == 'SATURN':
            ring_data = data_loader.get_ring_radii()['SATURN']
            for k in ['A', 'B', 'C']:
                for r in ring_data.get(k, []):
                    self.ring_radii.add(r)

    def __repr__(self) -> str:
        return f'Body({self.target!r}, {self.utc!r})'

    def create_other_body(self, other_target: str) -> 'Body':
        """
        Create a :class:`Body` instance using identical parameters but just with a
        different target. For example, the `europa` body created here will have
        identical parameters (see below) to the `jupiter` body, just with a different
        target. ::

            jupiter = Body('Jupiter', '2000-01-01', observer='Moon')
            europa = jupiter.create_other_body('Europa')

        The parameters kept the same are `utc`, `observer`, `observer_frame`,
        `illumination_source`, `aberration_correction`, `subpoint_method`, and
        `surface_method`.

        Args:
            other_target: Name of the other target, passed to :class:`Body`

        Returns:
            :class:`Body` instance which corresponds to `other_target`.
        """
        return Body(
            target=other_target,
            utc=self.utc,
            observer=self.observer,
            observer_frame=self.observer_frame,
            illumination_source=self.illumination_source,
            aberration_correction=self.aberration_correction,
            subpoint_method=self.subpoint_method,
            surface_method=self.surface_method,
        )

    def add_other_bodies_of_interest(self, *other_targets: str):
        """
        Add targets to the list of :attr:`other_bodies_of_interest` of interest to mark
        when plotting. The other targets are created using :func:`create_other_body`.
        For example, to add the Galilean moons as other targets to a Jupiter body,
        use ::

            body.add_other_bodies_of_interest('Io', 'Europa', 'Ganymede', 'Callisto')

        Args:
            *other_targets: Names of the other targets, passed to :class:`Body`
        """
        for other_target in other_targets:
            self.other_bodies_of_interest.append(self.create_other_body(other_target))

    # Coordinate transformations target -> observer direction
    def _lonlat2targvec_radians(
        self, lon: float, lat: float, alt: float = 0
    ) -> np.ndarray:
        """
        Transform lon/lat coordinates on body to rectangular vector in target frame.
        """
        return spice.pgrrec(
            self._target_encoded,  # type: ignore
            lon,
            lat,
            alt,  # type: ignore
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
        targvec_et = self._subpoint_et - dist_offset / self.speed_of_light()

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

    # Coordinate transformations km <-> radec
    def _get_km2radec_matrix_radians(self) -> np.ndarray:
        # Based on code in BodyXY._get_xy2radec_matrix_radians()
        # TODO make this actually work
        if self._matrix_km2radec is None:
            r_km = self.r_eq
            r_radians = np.arcsin(r_km / self.target_distance)
            s = r_radians / r_km
            theta = -np.deg2rad(self.north_pole_angle())
            stretch_matrix = np.array(
                [[-1 / np.abs(np.cos(self._target_dec_radians)), 0], [0, 1]]
            )
            rotation_matrix = self._rotation_matrix_radians(theta)
            transform_matrix_2x2 = s * np.matmul(rotation_matrix, stretch_matrix)

            v0 = np.array([0, 0])
            a0 = np.array([self._target_ra_radians, self._target_dec_radians])
            offset_vector = a0 - np.matmul(transform_matrix_2x2, v0)

            transform_matrix_3x3 = np.identity(3)
            transform_matrix_3x3[:2, :2] = transform_matrix_2x2
            transform_matrix_3x3[:2, 2] = offset_vector
            self._matrix_km2radec = transform_matrix_3x3
        return self._matrix_km2radec

    def _get_radec2km_matrix_radians(self) -> np.ndarray:
        if self._matrix_radec2km is None:
            self._matrix_radec2km = np.linalg.inv(self._get_km2radec_matrix_radians())
        return self._matrix_radec2km

    def _km2radec_radians(self, km_x: float, km_y: float) -> tuple[float, float]:
        a = self._get_km2radec_matrix_radians().dot(np.array([km_x, km_y, 1]))
        return a[0], a[1]

    def _radec2km_radians(self, ra: float, dec: float) -> tuple[float, float]:
        v = self._get_radec2km_matrix_radians().dot(np.array([ra, dec, 1]))
        return v[0], v[1]

    def km2radec(self, km_x: float, km_y: float) -> tuple[float, float]:
        # TODO docstring
        return self._radian_pair2degrees(*self._km2radec_radians(km_x, km_y))

    def radec2km(self, ra: float, dec: float) -> tuple[float, float]:
        return self._radec2km_radians(*self._degree_pair2radians(ra, dec))

    def _get_matplotlib_radec2km_transform_radians(
        self,
    ) -> matplotlib.transforms.Affine2D:
        if self._mpl_transform_radec2km_radians is None:
            self._mpl_transform_radec2km_radians = matplotlib.transforms.Affine2D(
                self._get_radec2km_matrix_radians()
            )
        return self._mpl_transform_radec2km_radians

    def matplotlib_radec2km_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        # TODO docstring
        if self._mpl_transform_radec2km is None:
            transform_rad2deg = matplotlib.transforms.Affine2D().scale(np.deg2rad(1))
            self._mpl_transform_radec2km = (
                transform_rad2deg + self._get_matplotlib_radec2km_transform_radians()
            )  #  type: ignore
        transform = self._mpl_transform_radec2km
        if ax:
            transform = transform + ax.transData
        return transform

    def matplotlib_km2radec_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        # TODO dosctring
        if self._mpl_transform_km2radec is None:
            self._mpl_transform_km2radec = (
                self.matplotlib_radec2km_transform().inverted()
            )
        transform = self._mpl_transform_km2radec
        if ax:
            transform = transform + ax.transData
        return transform

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

        Output arrays are like the outputs of :func:`limb_radec`, but the dayside
        coordinate arrays have non-illuminated locations replaced with NaN and the
        nightside arrays have illuminated locations replaced with NaN.

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
        return np.rad2deg(phase), np.rad2deg(incdnc), np.rad2deg(emissn)

    def _azimuth_angle_from_gie_radians(self, phase_radians: Numeric,incidence_radians:Numeric, emission_radians:Numeric) -> Numeric:
        # Based on Henrik's code at:
        # https://github.com/JWSTGiantPlanets/NIRSPEC-Toolkit/blob/5e2e2cc/JWSTSolarSystemPointing.py#L204-L209
        a = np.cos(phase_radians) - np.cos(emission_radians) * np.cos(incidence_radians)
        b = np.sqrt(1.0 - np.cos(emission_radians) ** 2) * np.sqrt(1.0 - np.cos(incidence_radians) ** 2)
        azimuth_radians = np.pi - np.arccos(a / b)
        return azimuth_radians

    def azimuth_angle_from_lonlat(self, lon: float, lat: float) -> float:
        """
        Calculate the azimuth angle of a longitude/latitude coordinate on the target 
        body.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            Azimuth angle in degrees.
        """
        azimuth_radians = self._azimuth_angle_from_gie_radians(*self._illumination_angles_from_targvec_radians(self.lonlat2targvec(lon, lat)))
        return np.rad2deg(azimuth_radians)

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
    def visible_lonlat_grid_radec(
        self, interval: float = 30, **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Convenience function to calculate a grid of equally spaced lines of constant
        longitude and latitude for use in plotting lon/lat grids.

        This function effectively combines :func:`visible_lon_grid_radec` and
        :func:`visible_lat_grid_radec` to produce both longitude and latitude gridlines.

        For example, to plot gridlines with a 45 degree interval, use::

            lines = body.visible_lonlat_grid_radec(interval=45)
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

        See also :func:`visible_lonlat_grid_radec`,

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
        :func:`visible_lonlat_grid_radec`.

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
        # lt argument is meaningless but there for convenience when chaining with
        # _state_from_targvec
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

    def ring_radec(
        self, radius: float, npts: int = 360, only_visible: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate RA/Dec coordinates of a ring around the target body.

        The ring is assumed to be directly above the planet's equator and has a constant
        `radius` for all longitudes. Use :attr:`ring_radii` to set the rings
        automatically plotted.

        Args:
            radius: Radius in km of the ring from the centre of the target body.
            npts: Number of points in the generated ring.
            only_visible: If `True` (default), the coordinates for the part of the ring
                hidden behind the target body are set to NaN. This routine will execute
                slightly faster with `only_visible` set to `False`.

        Returns:
            `(ra, dec)` tuple of coordinate arrays.
        """
        lons = np.deg2rad(np.linspace(0, 360, npts))
        alt = radius - self.r_eq
        targvecs = [self._lonlat2targvec_radians(lon, 0, alt) for lon in lons]
        obsvecs = [self._targvec2obsvec(targvec) for targvec in targvecs]

        ra_arr = np.full(npts, np.nan)
        dec_arr = np.full(npts, np.nan)
        for idx, obsvec in enumerate(obsvecs):
            if only_visible:
                # Test the obsvec ray from the observer to this point on the ring to see
                # if it has a surface intercept with the target body. If there is no
                # intercept (NotFoundError), then this ring point is definitely visible.
                # If there is surface intercept, then we see if the ring point is closer
                # to the observer than the target body's centre (=> this ring point is
                # visible) or if the ring is behind the target body (=> this ring point
                # is hidden).
                try:
                    spice.sincpt(
                        self._surface_method_encoded,
                        self._target_encoded,
                        self.et,
                        self._target_frame_encoded,
                        self._aberration_correction_encoded,
                        self._observer_encoded,
                        self._observer_frame_encoded,
                        obsvec,
                    )
                    # If we reach this point then the ring is either infront/behind the
                    # target body.
                    if self.vector_magnitude(obsvec) > self.target_distance:
                        # This part of the ring is hidden behind the target, so leave
                        # the output array values as NaN.
                        continue
                except NotFoundError:
                    # Ring is to the side of the target body, so is definitely visible,
                    # so continue with the coordinate conversion for this point.
                    pass
            ra_arr[idx], dec_arr[idx] = self._radian_pair2degrees(
                *self._obsvec2radec_radians(obsvec)
            )
        return ra_arr, dec_arr

    # Planetographic <-> planetocentric
    def _targvec2lonlat_centric(self, targvec: np.ndarray) -> tuple[float, float]:
        radius, lon_centric, lat_centric = spice.reclat(targvec)
        return self._radian_pair2degrees(lon_centric, lat_centric)

    def graphic2centric_lonlat(self, lon: float, lat: float) -> tuple[float, float]:
        """
        Convert planetographic longitude/latitude to planetocentric.

        Args:
            lon: Planetographic longitude.
            lat: Planetographic latitude.

        Returns:
            `(lon_centric, lat_centric)` tuple of planetocentric coordinates.
        """
        return self._targvec2lonlat_centric(self.lonlat2targvec(lon, lat))

    def centric2graphic_lonlat(
        self, lon_centric: float, lat_centric: float
    ) -> tuple[float, float]:
        """
        Convert planetocentric longitude/latitude to planetographic.

        Args:
            lon_centric: Planetocentric longitude.
            lat_centric: Planetographic latitude.

        Returns:
            `(lon, lat)` tuple of planetographic coordinates.
        """
        targvec = spice.latsrf(
            self._surface_method_encoded,  # type: ignore
            self._target_encoded,  # type: ignore
            self.et,
            self._target_frame_encoded,  # type: ignore
            [[np.deg2rad(lon_centric), np.deg2rad(lat_centric)]],
        )
        return self.targvec2lonlat(targvec[0])

    # Other
    def north_pole_angle(self) -> float:
        """
        Calculate the angle of the north pole of the target body relative to the 
        positive declination direction.

        Returns:
            Angle of the north pole in degrees.
        """
        np_ra, np_dec = self.lonlat2radec(0, 90)
        theta = np.arctan2(self.target_ra - np_ra, np_dec - self.target_dec)
        return np.rad2deg(theta)

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
            List of `(lon, lat, label)` tuples describing the poles where `lon` and
            `lat` give the coordinates of the pole on the target and `label` is a string
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
        self,
        transform: None | matplotlib.transforms.Transform,
        ax: Axes | None = None,
        color: str | tuple[float, float, float] = 'k',
        **kwargs,
    ) -> Axes:
        """Plot generic wireframe representation of the observation"""
        if ax is None:
            ax = cast(Axes, plt.gca())

        if transform is None:
            transform = ax.transData
        else:
            transform = transform + ax.transData

        for ra, dec in self.visible_lonlat_grid_radec(30):
            ax.plot(
                ra,
                dec,
                color=color,
                linestyle=':',
                alpha=0.5,
                transform=transform,
                **kwargs,
            )

        ax.plot(
            *self.limb_radec(),
            color=color,
            linewidth=0.5,
            transform=transform,
            **kwargs,
        )
        ax.plot(
            *self.terminator_radec(),
            color=color,
            linestyle='--',
            transform=transform,
            **kwargs,
        )

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination()
        ax.plot(ra_day, dec_day, color=color, transform=transform, **kwargs)

        for lon, lat, s in self.get_poles_to_plot():
            ra, dec = self.lonlat2radec(lon, lat)
            ax.text(
                ra,
                dec,
                s,
                ha='center',
                va='center',
                size='small',
                weight='bold',
                color=color,
                path_effects=[
                    path_effects.Stroke(linewidth=3, foreground='w'),
                    path_effects.Normal(),
                ],
                transform=transform,
                clip_on=True,
                **kwargs,
            )

        for lon, lat in self.coordinates_of_interest_lonlat:
            if self.test_if_lonlat_visible(lon, lat):
                ra, dec = self.lonlat2radec(lon, lat)
                ax.scatter(
                    ra,
                    dec,
                    marker='x',  # type: ignore
                    color=color,
                    transform=transform,
                    **kwargs,
                )
        for ra, dec in self.coordinates_of_interest_radec:
            ax.scatter(
                ra,
                dec,
                marker='+',  # type: ignore
                color=color,
                transform=transform,
                **kwargs,
            )

        for radius in self.ring_radii:
            ra, dec = self.ring_radec(radius)
            ax.plot(ra, dec, color=color, linewidth=0.5, transform=transform, **kwargs)

        for body in self.other_bodies_of_interest:
            ra = body.target_ra
            dec = body.target_dec
            ax.text(
                ra,
                dec,
                body.target + '\n',
                size='small',
                ha='center',
                va='center',
                color=color,
                alpha=0.5,
                transform=transform,
                clip_on=True,
                **kwargs,
            )
            ax.scatter(
                ra,
                dec,
                marker='+',  # type: ignore
                color=color,
                transform=transform,
                **kwargs,
            )
        ax.set_title(self.get_description(multiline=True))
        return ax

    def plot_wireframe_radec(
        self,
        ax: Axes | None = None,
        show: bool = False,
        color: str | tuple[float, float, float] = 'k',
        dms_ticks: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation using RA/Dec sky
        coordinates.

        Args:
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), uses
                `plt.gca()` to get the currently active axis.
            show: Toggle immediately showing the plotted figure with `plt.show()`.
            color: Matplotlib color used for to plot the wireframe.
            dms_ticks: Toggle between showing ticks as degrees, minutes and seconds
                (e.g. 12°34′56″) or decimal degrees (e.g. 12.582).

        Returns:
            The axis containing the plotted wireframe.
        """
        ax = self._plot_wireframe(transform=None, ax=ax, color=color, **kwargs)

        utils.format_radec_axes(ax, self.target_dec, dms_ticks)

        if show:
            plt.show()
        return ax

    def plot_wireframe_km(
        self,
        ax: Axes | None = None,
        show: bool = False,
        color: str | tuple[float, float, float] = 'k',
        **kwargs,
    ) -> Axes:
        # TODO docstring

        transform = self.matplotlib_radec2km_transform()
        ax = self._plot_wireframe(transform=transform, ax=ax, color=color, **kwargs)

        ax.set_xlabel('Projected distance (km)')
        ax.set_ylabel('Projected distance (km)')
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.set_aspect(1, adjustable='datalim')

        if show:
            plt.show()
        return ax
