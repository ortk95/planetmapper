import datetime
import functools
import math
from collections import defaultdict
from typing import Any, Callable, Literal, TypedDict, cast, overload

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import spiceypy as spice
from matplotlib.axes import Axes
from spiceypy.utils.exceptions import (
    NotFoundError,
    SpiceBODIESNOTDISTINCT,
    SpiceINVALIDTARGET,
    SpiceKERNELVARNOTFOUND,
    SpiceSPKINSUFFDATA,
)

from . import data_loader, utils
from .base import BodyBase, Numeric, _cache_stable_result
from .basic_body import BasicBody

_WireframeComponent = Literal[
    'all',
    'grid',
    'equator',
    'prime_meridian',
    'limb',
    'limb_illuminated',
    'terminator',
    'ring',
    'pole',
    'coordinate_of_interest_lonlat',
    'coordinate_of_interest_radec',
    'other_body_of_interest_marker',
    'other_body_of_interest_label',
    'hidden_other_body_of_interest_marker',
    'hidden_other_body_of_interest_label',
    'map_boundary',
]


class _WireframeKwargs(TypedDict, total=False):
    label_poles: bool
    add_title: bool
    grid_interval: float
    grid_lat_limit: float
    indicate_equator: bool
    indicate_prime_meridian: bool
    formatting: dict[_WireframeComponent, dict[str, Any]] | None

    # Hints for common formatting parameters to make type checking/autocomplete happy
    color: str | tuple[float, float, float]
    alpha: float
    zorder: float


DEFAULT_WIREFRAME_FORMATTING: dict[_WireframeComponent, dict[str, Any]] = {
    'all': dict(color='k'),
    'grid': dict(alpha=0.5, linestyle=':'),
    'equator': dict(linestyle='-'),
    'prime_meridian': dict(linestyle='-'),
    'limb': dict(linewidth=0.5),
    'limb_illuminated': dict(),
    'terminator': dict(linestyle='--'),
    'ring': dict(linewidth=0.5),
    'pole': dict(
        ha='center',
        va='center',
        size='small',
        weight='bold',
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground='w'),
            path_effects.Normal(),
        ],
        clip_on=True,
    ),
    'coordinate_of_interest_lonlat': dict(marker='x'),
    'coordinate_of_interest_radec': dict(marker='+'),
    'other_body_of_interest_marker': dict(marker='+'),
    'other_body_of_interest_label': dict(
        size='small',
        ha='center',
        va='center',
        alpha=0.5,
        clip_on=True,
    ),
    'hidden_other_body_of_interest_marker': dict(alpha=0.333),
    'hidden_other_body_of_interest_label': dict(),
    'map_boundary': dict(),
}


class _AngularCoordinateKwargs(TypedDict, total=False):
    origin_ra: float | None
    origin_dec: float | None
    coordinate_rotation: float


class Body(BodyBase):
    """
    Class representing an astronomical body observed at a specific time.

    Generally only `target`, `utc` and `observer` need to be changed. The additional
    parameters allow customising the exact settings used in the internal SPICE
    functions. Similarly, some methods (e.g. :func:`terminator_radec`) have parameters
    that are passed to SPICE functions which can almost always be left as their default
    values. See the
    `SPICE documentation <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/subpnt_c.html#Detailed_Input>`_
    for more details about possible parameter values.

    The `target` and `observer` names are passed to
    :func:`SpiceBase.standardise_body_name`, so a variety of formats are acceptable. For
    example `'jupiter'`, `'JUPITER'`, `' Jupiter '`, `'599'` and `599` will
    all resolve to `'JUPITER'`.

    :class:`Body` instances are hashable, so can be used as dictionary keys.

    This class inherits from :class:`SpiceBase` so the methods described above are also
    available.

    Args:
        target: Name of target body.
        utc: Time of observation. This can be provided in a variety of formats and is
            assumed to be UTC unless otherwise specified. The accepted formats are: any
            `string` datetime representation compatible with SPICE (e.g.
            `'2000-12-31T23:59:59'` - see the
            `documentation of acceptable string formats <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html>`_),
            a Python `datetime` object, or a `float` representing the Modified Julian
            Date (MJD) of the observation. Alternatively, if `utc` is `None` (the
            default), then the current time is used.
        observer: Name of observing body. Defaults to `'EARTH'`.
        aberration_correction: Aberration correction used to correct light travel time
            in SPICE. Defaults to `'CN'`.
        observer_frame: Observer reference frame. Defaults to `'J2000'`,
        illumination_source: Illumination source. Defaults to `'SUN'`.
        subpoint_method: Method used to calculate the sub-observer point in SPICE.
            Defaults to `'INTERCEPT/ELLIPSOID'`.
        surface_method: Method used to calculate surface intercepts in SPICE. Defaults
            to `'ELLIPSOID'`.
        **kwargs: Additional arguments are passed to :class:`SpiceBase`.
    """

    def __init__(
        self,
        target: str | int,
        utc: str | datetime.datetime | float | None = None,
        observer: str | int = 'EARTH',
        *,
        aberration_correction: str = 'CN',
        observer_frame: str = 'J2000',
        illumination_source: str = 'SUN',
        subpoint_method: str = 'INTERCEPT/ELLIPSOID',
        surface_method: str = 'ELLIPSOID',
        **kwargs,
    ) -> None:
        super().__init__(
            target=target,
            utc=utc,
            observer=observer,
            aberration_correction=aberration_correction,
            observer_frame=observer_frame,
            **kwargs,
        )

        # Document instance variables
        self.target: str
        """
        Name of the target body, as standardised by 
        :func:`SpiceBase.standardise_body_name`.
        """
        self.utc: str
        """
        String representation of the time of the observation in the format
        `'2000-01-01T00:00:00.000000'`. This time is in the UTC timezone.
        """
        self.observer: str
        """
        Name of the observer body, as standardised by 
        :func:`SpiceBase.standardise_body_name`.
        """
        self.aberration_correction: str
        """Aberration correction used to correct light travel time in SPICE."""
        self.observer_frame: str
        """Observer reference frame."""
        self.illumination_source: str
        """Illumination source."""
        self.subpoint_method: str
        """Method used to calculate the sub-observer point in SPICE."""
        self.surface_method: str
        """Method used to calculate surface intercepts in SPICE."""
        self.et: float
        """Ephemeris time of the observation corresponding to `utc`."""
        self.dtm: datetime.datetime
        """Python timezone aware datetime of the observation corresponding to `utc`."""
        self.target_body_id: int
        """SPICE numeric ID of the target body."""
        self.radii: np.ndarray
        """Array of radii of the target body along the x, y and z axes in km."""
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
        self.target_dec: float
        """Declination (Dec) of the target centre."""
        self.target_diameter_arcsec: float
        """Equatorial angular diameter of the target in arcseconds."""
        self.subpoint_distance: float
        """Distance from the observer to the sub-observer point on the target."""
        self.subpoint_lon: float
        """Longitude of the sub-observer point on the target."""
        self.subpoint_lat: float
        """Latitude of the sub-observer point on the target."""
        self.subsol_lon: float
        """Longitude of the sub-solar point on the target."""
        self.subsol_lat: float
        """Latitude of the sub-solar point on the target."""
        self.named_ring_data: dict[str, list[float]]
        """
        Dictionary of ring radii for the target from :func:`data_loader.get_ring_radii`.

        The dictionary keys are the names of the rings, and values are list of ring 
        radii in km. If the length of this list is 2, then the values give the inner and 
        outer radii of the ring respectively. Otherwise, the length should be 1, meaning
        the ring has a single radius. These ring radii values are sourced from
        `planetary factsheets <https://nssdc.gsfc.nasa.gov/planetary/planetfact.html>`_.
        If no ring data is available for the target, this dictionary is empty.

        Values from this dictionary can be easily accessed using the convenience
        function :func:`ring_radii_from_name`.
        """
        self.ring_radii: set[float]
        """
        Set of ring radii in km to plot around the target body's equator. Each radius
        is plotted as a single line, so for a wide ring you may want to add both the
        inner and outer edger of the ring. The radii are defined as the distance from
        the centre of the target body to the ring. For Saturn, the A, B and C rings from
        :attr:`named_ring_data` are included by default. For all other bodies,
        `ring_radii` is empty by default.

        Ring radii data from the :attr:`named_ring_data` can easily be added to 
        `ring_radii` using :func:`add_named_rings`. Example usage: ::

            body.ring_radii.add(122340) # Add new ring radius to plot
            body.ring_radii.add(136780) # Add new ring radius to plot
            body.ring_radii.update([66900, 74510]) # Add multiple radii to plot at once

            body.ring_radii.remove(122340) # Remove a ring radius
            body.ring_radii.clear() # Remove all ring radii

            # Add specific ring radii using data from planetary factsheets
            body.add_named_rings('main', 'halo')

            # Add all rings defined in the planetary factsheets
            body.add_named_rings()

        See also :func:`ring_radec`.
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
        self.other_bodies_of_interest: list[Body | BasicBody]
        """
        List of other bodies of interest to mark when plotting. Add to this list using 
        :func:`add_other_bodies_of_interest`.
        """

        # Process inputs
        self.illumination_source = illumination_source
        self.subpoint_method = subpoint_method
        self.surface_method = surface_method

        self._illumination_source_encoded = self._encode_str(self.illumination_source)
        self._subpoint_method_encoded = self._encode_str(self.subpoint_method)
        self._surface_method_encoded = self._encode_str(self.surface_method)

        # Get target properties and state
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
        self.subpoint_distance = float(np.linalg.norm(self._subpoint_rayvec))
        self.subpoint_lon, self.subpoint_lat = self.targvec2lonlat(
            self._subpoint_targvec
        )
        self._subpoint_obsvec = self._rayvec2obsvec(
            self._subpoint_rayvec, self._subpoint_et
        )
        self._subpoint_ra, self._subpoint_dec = self._radian_pair2degrees(
            *self._obsvec2radec_radians(self._subpoint_obsvec)
        )

        # Find sub solar point
        try:
            self._subsol_targvec, self._subsol_et, self._subsol_rayvec = spice.subslr(
                self._subpoint_method_encoded,  # type: ignore
                self._target_encoded,  # type: ignore
                self.et,
                self._target_frame_encoded,  # type: ignore
                self._aberration_correction_encoded,  # type: ignore
                self._observer_encoded,  # type: ignore
            )
            self.subsol_lon, self.subsol_lat = self.targvec2lonlat(self._subsol_targvec)
        except SpiceINVALIDTARGET:
            # If the target is the sun, then there is no sub-solar point
            self.subsol_lon = np.nan
            self.subsol_lat = np.nan

        # Set up equatorial plane (for ring calculations)
        targvec_north_pole = self.lonlat2targvec(0, 90)
        obsvec_north_pole = self._targvec2obsvec(targvec_north_pole)
        self._ring_plane = spice.nvp2pl(
            obsvec_north_pole - self._target_obsvec,
            self._target_obsvec,
        )

        # Load additional data
        self.named_ring_data = data_loader.get_ring_radii().get(self.target, {})

        # Create empty lists/blank values
        self.ring_radii = set()
        self.other_bodies_of_interest = []
        self.coordinates_of_interest_lonlat = []
        self.coordinates_of_interest_radec = []

        self._matrix_km2angular = None
        self._matrix_angular2km = None

        # Run custom setup
        if self.target == 'SATURN':
            for k in ['A', 'B', 'C']:
                for r in self.named_ring_data.get(k, []):
                    self.ring_radii.add(r)

    def __repr__(self) -> str:
        return f'Body({self.target!r}, {self.utc!r}, observer={self.observer!r})'

    def _get_equality_tuple(self) -> tuple:
        return (
            self.illumination_source,
            self.subpoint_method,
            self.surface_method,
            super()._get_equality_tuple(),
        )

    def _get_kwargs(self) -> dict[str, Any]:
        return super()._get_kwargs() | dict(
            illumination_source=self.illumination_source,
            subpoint_method=self.subpoint_method,
            surface_method=self.surface_method,
        )

    def _copy_options_to_other(self, other: Self) -> None:
        super()._copy_options_to_other(other)
        other.other_bodies_of_interest = self.other_bodies_of_interest.copy()
        other.coordinates_of_interest_lonlat = (
            self.coordinates_of_interest_lonlat.copy()
        )
        other.coordinates_of_interest_radec = self.coordinates_of_interest_radec.copy()
        other.ring_radii = self.ring_radii.copy()

    @overload
    def create_other_body(
        self, other_target: str | int, fallback_to_basic_body: Literal[False]
    ) -> 'Body': ...

    @overload
    def create_other_body(
        self, other_target: str | int, fallback_to_basic_body: bool = True
    ) -> 'Body|BasicBody': ...

    def create_other_body(
        self, other_target: str | int, fallback_to_basic_body: bool = True
    ) -> 'Body|BasicBody':
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

        If a full :class:`Body` instance cannot be created due to insufficient data in
        the SPICE kernel, a :class:`BasicBody` instance will be created instead. This is
        useful for objects such as minor satellites which do not have known radius data.

        Args:
            other_target: Name of the other target, passed to :class:`Body`
            fallback_to_basic_body: If a full :class:`Body` instance cannot be created
                due to insufficient data in the SPICE kernel, attempt to create a
                :class:`BasicBody` instance instead.

        Returns:
            :class:`Body` or :class:`BasicBody` instance which corresponds to
            `other_target`.
        """
        try:
            try:
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
            except SpiceKERNELVARNOTFOUND:
                if not fallback_to_basic_body:
                    raise
                return BasicBody(
                    target=other_target,
                    utc=self.utc,
                    observer=self.observer,
                    observer_frame=self.observer_frame,
                    aberration_correction=self.aberration_correction,
                )
        except NotFoundError as e:
            e.message += f'\n\nBody name: {other_target!r}'  #  type: ignore
            raise e

    # Stuff to customise wireframe plots
    def add_other_bodies_of_interest(
        self, *other_targets: str | int, only_visible: bool = False
    ) -> None:
        """
        Add targets to the list of :attr:`other_bodies_of_interest` of interest to mark
        when plotting. The other targets are created using :func:`create_other_body`.
        For example, to add the Galilean moons as other targets to a Jupiter body,
        use ::

            body = planetmapper.Body('Jupiter')
            body.add_other_bodies_of_interest('Io', 'Europa', 'Ganymede', 'Callisto')

        Integer SPICE ID codes can also be provided, which can be used to simplify
        adding multiple satellites to plots. ::

            body = planetmapper.Body('Uranus')
            body.add_other_bodies_of_interest(*range(701, 711))
            # Uranus' satellites have ID codes 701, 702, 703 etc, so this adds 10 moons
            # with a single function call

        See also :func:`add_satellites_to_bodies_of_interest`.

        Args:
            *other_targets: Names of the other targets, passed to :class:`Body`
            only_visible: If `True`, other targets which are hidden behind the target
                will not be added to :attr:`other_bodies_of_interest`.
        """
        for other_target in other_targets:
            body = self.create_other_body(other_target)
            if only_visible and not self.test_if_other_body_visible(body):
                continue
            if body not in self.other_bodies_of_interest:
                self.other_bodies_of_interest.append(body)

    def _get_all_satellite_bodies(
        self, skip_insufficient_data: bool = False, only_visible: bool = False
    ) -> 'list[Body | BasicBody]':
        out: 'list[Body | BasicBody]' = []
        id_base = (self.target_body_id // 100) * 100
        for other_target in range(id_base + 1, id_base + 99):
            try:
                body = self.create_other_body(other_target)
                if only_visible and not self.test_if_other_body_visible(body):
                    continue
                out.append(body)
            except SpiceSPKINSUFFDATA:
                if skip_insufficient_data:
                    continue
                raise
            except NotFoundError:
                continue
        return out

    def add_satellites_to_bodies_of_interest(
        self, skip_insufficient_data: bool = False, only_visible: bool = False
    ) -> None:
        """
        Automatically add all satellites in the target planetary system to
        :attr:`other_bodies_of_interest`.

        This uses the NAIF ID codes to identify the satellites. For example, Uranus has
        an ID of 799, and its satellites have codes 701, 702, 703..., so any object with
        a code in the range 701 to 798 is added for Uranus.

        See also :func:`add_other_bodies_of_interest`.

        Args:
            skip_insufficient_data: If True, satellites with insufficient data in the
                SPICE kernel will be skipped. If False, an exception will be raised.
            only_visible: If `True`, satellites which are hidden behind the target body
                will not be added.
        """
        satellites = self._get_all_satellite_bodies(
            skip_insufficient_data=skip_insufficient_data, only_visible=only_visible
        )
        for satellite in satellites:
            if satellite not in self.other_bodies_of_interest:
                self.other_bodies_of_interest.append(satellite)

    @staticmethod
    def _standardise_ring_name(name: str) -> str:
        name = name.casefold().strip().removesuffix('ring')
        for a, b in data_loader.get_ring_aliases().items():
            name = name.replace(a, b)
        return name.casefold().strip()

    def ring_radii_from_name(self, name: str) -> list[float]:
        """
        Get list of ring radii in km for a named ring.

        This is a convenience function to load data from :attr:`named_ring_data`.

        Args:
            name: Name of ring. This is case insensitive and, "ring" suffix is
                optional and non-ASCII versions are allowed. For example, `'liberte'`
                will load the `'Liberté'` ring radii for Uranus and `'amalthea'` will
                load the `'Amalthea Ring'` radii for Jupiter.

        Raises:
            ValueError: if no ring with the provided name is found.

        Returns:
            List of ring radii in km. If the length of this list is 2, then the values
            give the inner and outer radii of the ring respectively. Otherwise, the
            length should be 1, meaning the ring has a single radius.
        """
        name = self._standardise_ring_name(name)
        for n, radii in self.named_ring_data.items():
            if name == self._standardise_ring_name(n):
                return radii
        raise ValueError(
            f'No rings found named {name!r} in named_ring_data.'
            + '\nValid names: {}'.format(
                [self._standardise_ring_name(n) for n in self.named_ring_data.keys()]
            )
        )

    def add_named_rings(self, *names: str) -> None:
        """
        Add named rings to :attr:`ring_radii` so that they appear when creating
        wireframe plots. If no arguments are provided (i.e. calling
        `body.add_named_rings()`), then all rings in :attr:`named_ring_data` are added
        to :attr:`ring_radii`.

        This is a convenience function to add data from :attr:`named_ring_data` to
        :attr:`ring_radii`.

        Args:
            *names: Ring names which are passed to :func:`ring_radii_from_name`. If
                no names are provided then all rings in :attr:`named_ring_data` are
                added.
        """
        if len(names) == 0:
            names = tuple(self.named_ring_data.keys())
        for name in names:
            self.ring_radii.update(self.ring_radii_from_name(name))

    # COORDINATE TRANSFORMATIONS
    # Generally all transformations in the public API should be built from a pair of
    # transformations to/from obsvec. Then any new coordinate system can be added by
    # simply creating a pair of private transformations to/from obsvec, and then adding
    # the relevant public methods.
    # See also BodyBase and BodyXY for some transformations.

    # Coordinate transformations target -> observer direction
    def _lonlat2targvec_radians(
        self, lon: float, lat: float, alt: float = 0
    ) -> np.ndarray:
        """
        Transform lon/lat coordinates on body to rectangular vector in target frame.
        """
        if not (math.isfinite(lon) and math.isfinite(lat) and math.isfinite(alt)):
            return np.array([np.nan, np.nan, np.nan])
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
        targvec_offset = targvec - self._subpoint_targvec

        # Calculate the difference in LOS distance between observer<->subpoint and
        # observer<->point of interest
        dist_offset = (
            np.linalg.norm(self._subpoint_rayvec + targvec_offset)
            - self.subpoint_distance
        )  # pyright: ignore[reportGeneralTypeIssues]

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
            targvec_et,  #  type: ignore
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

    # Coordinate transformations observer -> target direction
    def _radec2obsvec_norm_radians(self, ra: float, dec: float) -> np.ndarray:
        if not (math.isfinite(ra) and math.isfinite(dec)):
            return np.array([np.nan, np.nan, np.nan])
        return spice.radrec(1.0, ra, dec)

    def _radec2obsvec_norm(self, ra: float, dec: float) -> np.ndarray:
        return self._radec2obsvec_norm_radians(*self._degree_pair2radians(ra, dec))

    def _obsvec2targvec(self, obsvec: np.ndarray) -> np.ndarray:
        """
        Transform rectangular vector in observer frame to rectangular vector in target
        frame.

        Based on inverse of _targvec2obsvec
        """

        # Get the target vector from the subpoint to the point of interest
        obsvec_offset = obsvec - self._subpoint_obsvec

        # Calculate the difference in LOS distance between observer<->subpoint and
        # observer<->point of interest
        dist_offset = (
            np.linalg.norm(-self._subpoint_rayvec + obsvec_offset)
            - self.subpoint_distance
        )  # pyright: ignore[reportGeneralTypeIssues]

        # Use the calculated difference in distance relative to the subpoint to
        # calculate the time corresponding to when the ray left the surface at the point
        # of interest
        obsvec_et = self._subpoint_et - dist_offset / self.speed_of_light()

        # Create the transform matrix converting between the target vector at the time
        # the ray left the point of interest -> the observer vector at the time the ray
        # hit the detector
        transform_matrix = spice.pxfrm2(
            self._observer_frame_encoded,  # type: ignore
            self._target_frame_encoded,  # type: ignore
            self.et,
            obsvec_et,  # type: ignore
        )

        # Use the transform matrix to perform the actual transformation
        return self._subpoint_targvec + np.matmul(transform_matrix, obsvec_offset)

    def _obsvec_norm2targvec(self, obsvec_norm: np.ndarray) -> np.ndarray:
        """TODO add note about raising NotFoundError"""
        spoint, *_ = spice.sincpt(
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
        if not (
            math.isfinite(targvec[0])
            and math.isfinite(targvec[1])
            and math.isfinite(targvec[2])
        ):
            # ^ profiling suggests this is the fastest NaN check
            return np.nan, np.nan
        lon, lat, alt = spice.recpgr(
            self._target_encoded,  # type: ignore
            targvec,
            self.r_eq,
            self.flattening,
        )
        return lon, lat

    # Useful transformations (built from combinations of above transformations)
    def _lonlat2obsvec(self, lon: float, lat: float) -> np.ndarray:
        return self._targvec2obsvec(
            self._lonlat2targvec_radians(*self._degree_pair2radians(lon, lat)),
        )

    def _obsvec_norm2lonlat(
        self, obsvec_norm: np.ndarray, not_found_nan: bool
    ) -> tuple[float, float]:
        try:
            lon, lat = self._radian_pair2degrees(
                *self._targvec2lonlat_radians(self._obsvec_norm2targvec(obsvec_norm))
            )
        except NotFoundError:
            if not_found_nan:
                lon = np.nan
                lat = np.nan
            else:
                raise
        return lon, lat

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
        return self._obsvec2radec(self._lonlat2obsvec(lon, lat))

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
        return self._obsvec_norm2lonlat(self._radec2obsvec_norm(ra, dec), not_found_nan)

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
                (
                    self._obsvec2radec_radians(self._targvec2obsvec(t))
                    if condition_func(t)
                    else (np.nan, np.nan)
                )
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

    # Coordinate transformations angular plane
    @_cache_stable_result
    def _get_obsvec2angular_matrix(
        self,
        *,
        origin_ra: float | None = None,
        origin_dec: float | None = None,
        coordinate_rotation: float = 0.0,
    ) -> np.ndarray:
        # any changes to kwargs/defaults should be reflected in radec2angular and
        # plot_wireframe_angular
        # XXX stabilise names (kwargs)
        # XXX test
        # XXX add generic documentation
        # XXX add full set of derived transformations (km, xy)
        if origin_ra is None:
            origin_ra = self.target_ra
        if origin_dec is None:
            origin_dec = self.target_dec
        origin_obsvec = self._radec2obsvec_norm_radians(
            *self._degree_pair2radians(origin_ra, origin_dec)
        )

        _, ra_angle, _ = spice.recrad(origin_obsvec)
        ra_matrix = spice.rotate(ra_angle, 3)

        _, _, dec_angle = spice.recrad(ra_matrix @ origin_obsvec)
        dec_matrix = spice.rotate(-dec_angle, 2)

        rotation_matrix = spice.rotate(np.deg2rad(coordinate_rotation), 1)

        return rotation_matrix @ dec_matrix @ ra_matrix

    def _obsvec2angular(
        self, obsvec: np.ndarray, **angular_kwargs: Unpack[_AngularCoordinateKwargs]
    ) -> tuple[float, float]:
        if not (
            math.isfinite(obsvec[0])
            and math.isfinite(obsvec[1])
            and math.isfinite(obsvec[2])
        ):
            # ^ profiling suggests this is the fastest NaN check
            return np.nan, np.nan
        vec = self._get_obsvec2angular_matrix(**angular_kwargs) @ obsvec
        _, x, y = spice.recrad(vec)
        x = (-np.rad2deg(x)) % 360.0
        if x > 180.0:
            x -= 360.0
        y = np.rad2deg(y)
        return x * 3600.0, y * 3600.0  # convert degrees -> arcseconds

    def _angular2obsvec_norm(
        self,
        angular_x: float,
        angular_y: float,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> np.ndarray:
        vec = spice.radrec(
            1.0, -np.deg2rad(angular_x / 3600.0), np.deg2rad(angular_y / 3600.0)
        )
        # inverse of a roation matrix is just the transpose
        return self._get_obsvec2angular_matrix(**angular_kwargs).T @ vec

    def radec2angular(
        self,
        ra: float,
        dec: float,
        *,
        origin_ra: float | None = None,
        origin_dec: float | None = None,
        coordinate_rotation: float = 0.0,
    ) -> tuple[float, float]:
        """
        Convert RA/Dec sky coordinates for the observer to relative angular coordinates.

        The origin and rotation of the relative angular coordinates can be customised
        using the `origin_ra`, `origin_dec` and `coordinate_rotation` arguments. If
        these are not provided, the origin will be the centre of the target body and the
        rotation will be the same as in RA/Dec coordinates.

        Args:
            ra: Right ascension of point in the sky of the observer.
            dec: Declination of point in the sky of the observer.
            origin_ra: Right ascension (RA) of the origin of the relative angular
                coordinates. If `None`, the RA of the centre of the target body is used.
            origin_dec: Declination (Dec) of the origin of the relative angular
                coordinates. If `None`, the Dec of the centre of the target body is
                used.
            coordinate_rotation: Angle in degrees to rotate the relative angular
                coordinates around the origin, relative to the positive declination
                direction. The default `coordinate_rotation` is 0.0, so the target will
                have the same orientation as in RA/Dec coordinates.

        Returns:
            `(angular_x, angular_y)` tuple containing the relative angular coordinates
            of the point in arcseconds.
        """
        # XXX add detail to the docstring?
        return self._obsvec2angular(
            self._radec2obsvec_norm(ra, dec),
            origin_ra=origin_ra,
            origin_dec=origin_dec,
            coordinate_rotation=coordinate_rotation,
        )

    def angular2radec(
        self,
        angular_x: float,
        angular_y: float,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        """
        Convert relative angular coordinates to RA/Dec sky coordinates for the observer.

        Args:
            angular_x: Angular coordinate in the x direction in arcseconds.
            angular_y: Angular coordinate in the y direction in arcseconds.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`radec2angular` for details.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point.
        """
        return self._obsvec2radec(
            self._angular2obsvec_norm(angular_x, angular_y, **angular_kwargs)
        )

    def angular2lonlat(
        self,
        angular_x: float,
        angular_y: float,
        *,
        not_found_nan: bool = True,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        """
        Convert relative angular coordinates to longitude/latitude coordinates on the
        target body.

        Args:
            angular_x: Angular coordinate in the x direction in arcseconds.
            angular_y: Angular coordinate in the y direction in arcseconds.
            not_found_nan: Controls behaviour when the input `angular_x` and `angular_y`
                coordinates are missing the target body.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`radec2angular` for details.

        Returns:
            `(lon, lat)` tuple containing the longitude and latitude of the point. If
            the provided angular coordinates are missing the target body and
            `not_found_nan` is True, then the `lon` and `lat` values will both be NaN.

        Raises:
            NotFoundError: If the provided angular coordinates are missing the target
                body and `not_found_nan` is False, then NotFoundError will be raised.
        """
        return self._obsvec_norm2lonlat(
            self._angular2obsvec_norm(angular_x, angular_y, **angular_kwargs),
            not_found_nan,
        )

    def lonlat2angular(
        self,
        lon: float,
        lat: float,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        """
        Convert longitude/latitude coordinates on the target body to relative angular
        coordinates.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`radec2angular` for details.

        Returns:
            `(angular_x, angular_y)` tuple containing the relative angular coordinates
            of the point in arcseconds.
        """
        return self._obsvec2angular(self._lonlat2obsvec(lon, lat), **angular_kwargs)

    # Coordinate transformations km <-> angular
    def _get_km2angular_matrix(self) -> np.ndarray:
        if self._matrix_km2angular is None:
            # angular coords are centred on the target, so just need to convert
            # arcsec to km with a constant scale factor (s), and rotate so the north
            # pole is at the top
            s = (self.target_diameter_arcsec / 2) / self.r_eq
            theta_radians = np.deg2rad(self.north_pole_angle())
            transform_matrix = s * self._rotation_matrix_radians(theta_radians)
            self._matrix_km2angular = transform_matrix
        return self._matrix_km2angular

    def _get_angular2km_matrix(self) -> np.ndarray:
        if self._matrix_angular2km is None:
            self._matrix_angular2km = np.linalg.inv(self._get_km2angular_matrix())
        return self._matrix_angular2km

    def _km2obsvec_norm(self, km_x: float, km_y: float) -> np.ndarray:
        return self._angular2obsvec_norm(
            *(self._get_km2angular_matrix().dot(np.array([km_x, km_y])))
        )

    def _obsvec2km(self, obsvec: np.ndarray) -> tuple[float, float]:
        km_x, km_y = self._get_angular2km_matrix().dot(
            np.array(self._obsvec2angular(obsvec))
        )
        return km_x, km_y

    def km2radec(self, km_x: float, km_y: float) -> tuple[float, float]:
        """
        Convert distance in target plane to RA/Dec sky coordinates for the observer.

        Args:
            km_x: Distance in target plane in km in the East-West direction.
            km_y: Distance in target plane in km in the North-South direction.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point.
        """
        return self._obsvec2radec(self._km2obsvec_norm(km_x, km_y))

    def radec2km(self, ra: float, dec: float) -> tuple[float, float]:
        """
        Convert RA/Dec sky coordinates for the observer to distances in the target
        plane.

        Args:
            ra: Right ascension of point in the sky of the observer.
            dec: Declination of point in the sky of the observer.

        Returns:
            `(km_x, km_y)` tuple containing distances in km in the target plane in the
            East-West and North-South directions respectively.
        """
        return self._obsvec2km(self._radec2obsvec_norm(ra, dec))

    def km2lonlat(
        self, km_x: float, km_y: float, not_found_nan: bool = True
    ) -> tuple[float, float]:
        """
        Convert distance in target plane to longitude/latitude coordinates on the target
        body.

        Args:
            km_x: Distance in target plane in km in the East-West direction.
            km_y: Distance in target plane in km in the North-South direction.
            not_found_nan: Controls behaviour when the input `km_x` and `km_y`
                coordinates are missing the target body.

        Returns:
            `(lon, lat)` tuple containing the longitude and latitude of the point. If
            the provided km coordinates are missing the target body, then the `lon` and
            `lat` values will both be NaN if `not_found_nan` is True, otherwise a
            NotFoundError will be raised.

        Raises:
            NotFoundError: If the provided km coordinates are missing the target body
            and `not_found_nan` is False, then NotFoundError will be raised.
        """
        return self._obsvec_norm2lonlat(self._km2obsvec_norm(km_x, km_y), not_found_nan)

    def lonlat2km(self, lon: float, lat: float) -> tuple[float, float]:
        """
        Convert longitude/latitude coordinates on the target body to distances in the
        target plane.

        Args:
            lon: Longitude of point on the target body.
            lat: Latitude of point on the target body.

        Returns:
            `(km_x, km_y)` tuple containing distances in km in the target plane in the
            East-West and North-South directions respectively.
        """
        return self._obsvec2km(self._lonlat2obsvec(lon, lat))

    def km2angular(
        self,
        km_x: float,
        km_y: float,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        """
        Convert distance in target plane to relative angular coordinates.

        Args:
            km_x: Distance in target plane in km in the East-West direction.
            km_y: Distance in target plane in km in the North-South direction.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`radec2angular` for details.

        Returns:
            `(angular_x, angular_y)` tuple containing the relative angular coordinates
            of the point in arcseconds.
        """
        return self._obsvec2angular(self._km2obsvec_norm(km_x, km_y), **angular_kwargs)

    def angular2km(
        self,
        angular_x: float,
        angular_y: float,
        **angular_kwargs: Unpack[_AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        """
        Convert relative angular coordinates to distances in the target plane.

        Args:
            angular_x: Angular coordinate in the x direction in arcseconds.
            angular_y: Angular coordinate in the y direction in arcseconds.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`radec2angular` for details.

        Returns:
            `(km_x, km_y)` tuple containing distances in km in the target plane in the
            East-West and North-South directions respectively.
        """
        return self._obsvec2km(
            self._angular2obsvec_norm(angular_x, angular_y, **angular_kwargs)
        )

    # General
    def _illumf_from_targvec_radians(
        self, targvec: np.ndarray
    ) -> tuple[float, float, float, bool, bool]:
        if not (
            math.isfinite(targvec[0])
            and math.isfinite(targvec[1])
            and math.isfinite(targvec[2])
        ):
            # ^ profiling suggests this is the fastest NaN check
            return np.nan, np.nan, np.nan, False, False
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

    def limb_coordinates_from_radec(
        self, ra: float, dec: float
    ) -> tuple[float, float, float]:
        """
        Calculate the coordinates relative to the target body's limb for a point in the
        sky.

        The coordinates are calculated for the point on the ray (as defined by RA/Dec)
        which is closest to the target body's limb.

        Args:
            ra: Right ascension of point in the sky of the observer.
            dec: Declination of point in the sky of the observer.

        Returns:
            `(lon, lat, dist)` tuple of coordinates relative to the target body's limb.
            `lon` and `lat` give the planetographic longitude and latitude of the point
            on the limb closest to the point defined by `ra` and `dec`. `dist` gives the
            distance from the point defined by `ra` and `dec` to the target's limb.
            Positive values of `dist` mean that the point is above the limb and negative
            values mean that the point is below the limb (i.e. on the target body's
            disc).
        """
        coords = self._limb_coordinates_from_obsvec(
            self._radec2obsvec_norm_radians(*self._degree_pair2radians(ra, dec))
        )
        return coords

    def _limb_coordinates_from_obsvec(
        self, obsvec_norm: np.ndarray
    ) -> tuple[float, float, float]:
        if not (
            math.isfinite(obsvec_norm[0])
            and math.isfinite(obsvec_norm[1])
            and math.isfinite(obsvec_norm[2])
        ):
            return np.nan, np.nan, np.nan

        # Get the point on the RA/Dec ray (defined be obsvec_norm) that is closest to
        # the centre of the target body.
        nearpoint_obsvec, nearpoint_dist = spice.nplnpt(
            np.array([0, 0, 0]),  # centre of observer
            obsvec_norm,  # direction vector from observer to POI
            self._target_obsvec,  # reference point at centre of target body
        )

        # Get the point on the surface of the target body that is closest to the
        # nearpoint.
        surface_targvec = spice.surfpt(
            np.array([0, 0, 0]),
            self._obsvec2targvec(nearpoint_obsvec),
            self.radii[0],
            self.radii[1],
            self.radii[2],
        )
        lon, lat = self.targvec2lonlat(surface_targvec)
        dist = nearpoint_dist - self.vector_magnitude(surface_targvec)
        return lon, lat, dist

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

    def other_body_los_intercept(
        self, other: 'str | int | Body | BasicBody'
    ) -> None | Literal['transit', 'hidden', 'part transit', 'part hidden', 'same']:
        """
        Test for line-of-sight intercept between the target body and another body.

        This can be used to test for if another body (e.g. a moon) is in front of or
        behind the target body (e.g. a planet).

        See also :func:`test_if_other_body_visible`.

        .. warning::

            This method does not perform any checks to ensure that any input
            :class:`Body` or :class:`BasicBody` instances have a consistent observer
            location and observation time as the target body.

        Args:
            other: Other body to test for intercept with. Can be a :class`Body` (or
                :class:`BasicBody`) instance, or a string/integer NAIF ID code which is
                passed to :func:`create_other_body`.

        Returns:
            `None` if there is no intercept, otherwise a string indicating the type of
            intercept. For example, with `jupiter.other_body_los_intercept('europa')`,
            the possible return values mean:

                - `None` - there is no intercept, meaning that Europa and Jupiter do not
                  overlap in the sky.
                - `'hidden'` - all of Europa's disk is hidden behind Jupiter.
                - `'part hidden'` - part of Europa's disk is hidden behind Jupiter and
                  part is visible.
                - `'transit'` - all of Europa's disk is in front of Jupiter.
                - `'part transit'` - part of Europa's disk is in front of Jupiter.

            The return value can also be `'same'`, which means that the other body is
            the same object as the target body (or has an identical location).
        """
        if not isinstance(other, BodyBase):
            other = self.create_other_body(other)

        if isinstance(other, BasicBody):
            try:
                self.radec2lonlat(
                    other.target_ra, other.target_dec, not_found_nan=False
                )
            except NotFoundError:
                return None  # No intercept with the target body
            if other.target_distance == self.target_distance:
                return 'same'
            elif other.target_distance - self.target_distance > 0:
                return 'hidden'
            else:
                return 'transit'

        try:
            occultation = spice.occult(
                self.target,
                'ELLIPSOID',
                self.target_frame,
                other.target,
                'ELLIPSOID',
                other.target_frame,
                self.aberration_correction,
                self.observer,
                self.et,
            )
        except SpiceBODIESNOTDISTINCT:
            return 'same'

        match occultation:
            case 3:
                return 'hidden'
            case 1 | 2:
                return 'part hidden'
            case 0:
                return None
            case -1 | -3:
                return 'part transit'
            case -2:
                return 'transit'
        raise ValueError(f'Unknown occultation code: {occultation}')  # pragma: no cover

    def test_if_other_body_visible(self, other: 'str | int | Body | BasicBody') -> bool:
        """
        Test if another body is visible, or is hidden behind the target body.

        This is a convenience method equivalent to: ::

            body.other_body_los_intercept(other) != 'hidden'

        Args:
            other: Other body to test for visibility, passed to
                :func:`other_body_los_intercept`.

        Returns:
            `False` if the other body is hidden behind the target body, otherwise
            `True`. If any part of the other body is visible, this method will return
            `True`.
        """
        return self.other_body_los_intercept(other) != 'hidden'

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

    def _azimuth_angle_from_gie_radians(
        self,
        phase_radians: Numeric,
        incidence_radians: Numeric,
        emission_radians: Numeric,
    ) -> Numeric:
        # Based on Henrik's code at:
        # https://github.com/JWSTGiantPlanets/NIRSPEC-Toolkit/blob/5e2e2cc/JWSTSolarSystemPointing.py#L204-L209
        a = np.cos(phase_radians) - np.cos(emission_radians) * np.cos(incidence_radians)
        b = np.sqrt(1.0 - np.cos(emission_radians) ** 2) * np.sqrt(
            1.0 - np.cos(incidence_radians) ** 2
        )
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
        azimuth_radians = self._azimuth_angle_from_gie_radians(
            *self._illumination_angles_from_targvec_radians(
                self.lonlat2targvec(lon, lat)
            )
        )
        return np.rad2deg(azimuth_radians)

    def _lst_from_lon(
        self, lon: float
    ) -> tuple[int | float, int | float, int | float, str, str]:
        if not math.isfinite(lon):
            return np.nan, np.nan, np.nan, '', ''
        return spice.et2lst(
            self.et - self.target_light_time,
            self.target_body_id,
            np.deg2rad(lon),
            'planetographic',
        )

    def local_solar_time_from_lon(self, lon: float) -> float:
        """
        Calculate the numerical local solar time for a longitude on the target body. For
        example, `0.0` corresponds to midnight and `12.5` corresponds to 12:30pm.

        See also :func:`local_solar_time_string_from_lon`.

        .. note::

            A 'local hour' of solar time is a defined as 1/24th of the solar day on the
            target body, so will not correspond to a 'normal' hour as measured by a
            clock. See
            `the SPICE documentation <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/et2lst_c.html>`__
            for more details.

        Args:
            lon: Longitude of point on target body.

        Returns:
            Numerical local solar time in 'local hours'.
        """
        hr, mn, sc, time, ampm = self._lst_from_lon(lon)
        return hr + mn / 60 + sc / 3600

    def local_solar_time_string_from_lon(self, lon: float) -> str:
        """
        Local solar time string representation for a longitude on the target body. For
        example, `'00:00:00'` corresponds to midnight and `'12:30:00'` corresponds to
        12:30pm.

        See :func:`local_solar_time_from_lon` for more details.

        Args:
            lon: Longitude of point on target body.

        Returns:
            String representation of local solar time.
        """
        hr, mn, sc, time, ampm = self._lst_from_lon(lon)
        return time

    def terminator_radec(
        self,
        npts: int = 360,
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

    # Rings
    def _ring_coordinates_from_obsvec(
        self, obsvec: np.ndarray, only_visible: bool = True
    ) -> tuple[float, float, float]:
        if not (
            math.isfinite(obsvec[0])
            and math.isfinite(obsvec[1])
            and math.isfinite(obsvec[2])
        ):
            return np.nan, np.nan, np.nan
        nxpts, intercept_obsvec = spice.inrypl(
            np.array([0, 0, 0]), obsvec, self._ring_plane
        )
        if nxpts != 1:
            return np.nan, np.nan, np.nan
        targvec = self._obsvec2targvec(intercept_obsvec)
        lon, lat, alt = spice.recpgr(
            self._target_encoded,  # type: ignore
            targvec,
            self.r_eq,
            self.flattening,
        )
        if only_visible and alt < 0:
            return np.nan, np.nan, np.nan

        distance = self.vector_magnitude(intercept_obsvec)
        if only_visible:
            try:
                position, velocity, lt = self._state_from_targvec(
                    self._obsvec_norm2targvec(obsvec)
                )
                surface_distance = lt * self.speed_of_light()
                if surface_distance < distance:
                    return np.nan, np.nan, np.nan
            except NotFoundError:
                pass

        lon = np.rad2deg(lon)
        radius = alt + self.r_eq
        return radius, lon, distance

    def ring_plane_coordinates(
        self, ra: float, dec: float, only_visible: bool = True
    ) -> tuple[float, float, float]:
        """
        Calculate coordinates in the target body's equatorial (ring) plane. This is
        mainly useful for calculating the coordinates in a body's ring system at a given
        point in the sky.

        To calculate the coordinates corresponding to a location on the target body, you
        can use ::

            body.ring_plane_coordinates(*body.radec2lonlat(lon, lat))

        This form can be useful to identify parts of a planet's surface which are
        obscured by its rings ::

            radius, _, _ = body.ring_plane_coordinates(*body.lonlat2radec(lon, lat))
            ring_data = planetmapper.data_loader.get_ring_radii()['SATURN']
            for name, radii in ring_data.items():
                if min(radii) < radius < max(radii):
                    print(f'Point obscured by {name} ring')
                    break
            else:
                print('Point not obscured by rings')

        Args:
            ra: Right ascension of point in the sky of the observer.
            dec: Declination of point in the sky of the observer.
            only_visible: If `True` (the default), coordinates for parts of the
                equatorial plane hidden behind the target body are set to NaN.

        Returns:
            `(ring_radius, ring_longitude, ring_distance)` tuple for the point on the
            target body's equatorial (ring) plane. `ring_radius` gives the distance of
            the point in km from the centre of the target body. `ring_longitude` gives
            the planetographic longitude of the point in degrees. `ring_distance` gives
            the distance from the observer to the point in km.
        """
        return self._ring_coordinates_from_obsvec(
            self._radec2obsvec_norm_radians(*self._degree_pair2radians(ra, dec)),
            only_visible=only_visible,
        )

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
        self, lons: list[float] | np.ndarray, npts: int = 60, *, lat_limit: float = 90
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
            lat_limit: Latitude limit for gridlines. For example, if `lat_limit=60`,
                the gridlines will be calculated for latitudes between 60°N and 60°S
                (inclusive).

        Returns:
            List of `(ra, dec)` tuples, corresponding to the list of input `lons`. `ra`
            and `dec` are arrays of RA/Dec coordinate values for that gridline.
        """
        lats = np.linspace(-lat_limit, lat_limit, npts)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for lon in lons:
            targvecs = [self.lonlat2targvec(lon, lat) for lat in lats]
            ra, dec = self._targvec_arr2radec_arrs(
                targvecs, condition_func=self._test_if_targvec_visible
            )
            out.append((ra, dec))
        return out

    def visible_lat_grid_radec(
        self, lats: list[float] | np.ndarray, npts: int = 120, *, lat_limit: float = 90
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Constant latitude version of :func:`visible_lon_grid_radec`. See also
        :func:`visible_lonlat_grid_radec`.

        Args:
            lats: List of latitudes to plot.
            npts: Number of points in each full line of constant latitude.
            lat_limit: Latitude limit for gridlines. For example, if `lat_limit=60`,
                only gridlines with latitudes between 60°N and 60°S (inclusive) will be
                calculated.

        Returns:
            List of `(ra, dec)` tuples, corresponding to the list of input `lats`. `ra`
            and `dec` are arrays of RA/Dec coordinate values for that gridline.
        """
        lons = np.linspace(0, 360, npts)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for lat in lats:
            if abs(lat) > lat_limit:
                continue
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

    def distance_from_lonlat(self, lon: float, lat: float) -> float:
        """
        Calculate distance from observer to a point on the target's surface.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            Distance of the point in km.
        """
        position, velocity, lt = self._state_from_targvec(self.lonlat2targvec(lon, lat))
        return lt * self.speed_of_light()

    # Planetographic <-> planetocentric
    def _targvec2lonlat_centric(self, targvec: np.ndarray) -> tuple[float, float]:
        if not (
            math.isfinite(targvec[0])
            and math.isfinite(targvec[1])
            and math.isfinite(targvec[2])
        ):
            return np.nan, np.nan
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
        if not (math.isfinite(lon_centric) and math.isfinite(lat_centric)):
            return np.nan, np.nan
        targvecs = spice.latsrf(
            self._surface_method_encoded,  # type: ignore
            self._target_encoded,  # type: ignore
            self.et,
            self._target_frame_encoded,  # type: ignore
            [[np.deg2rad(lon_centric), np.deg2rad(lat_centric)]],
        )
        return self.targvec2lonlat(targvecs[0])

    # Other
    def north_pole_angle(self) -> float:
        """
        Calculate the angle of the north pole of the target body relative to the
        positive declination direction.

        .. note::

            This method calculates the angle between the centre of the target and its
            north pole, so may produce unexpected results for targets which are located
            at the celestial pole.

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

    @staticmethod
    def _get_local_affine_transform_matrix(
        coordinate_func: Callable[[float, float], tuple[float, float]],
        location: tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate the local affine transformation matrix for a given coordinate
        transformation around a given location.

        Args:
            coordinate_func: Function to convert between coordinate systems (e.g.
                `radec2km`),
            location: Coordinates (in original coordinate system) to calculate the
                transformation matrix around. This is usually the location of the
                target body.

        Returns:
            Augmented affine transformation matrix representing the transformation
            between coordinate systems near the provided `location`.
        """
        x0, y0 = location
        eq1, eq2 = coordinate_func(x0, y0)
        eq3, eq4 = coordinate_func(x0 + 1.0, y0)
        eq5, eq6 = coordinate_func(x0, y0 + 1.0)

        a = eq3 - eq1
        b = eq5 - eq1
        c = eq1 - a * x0 - b * y0

        d = eq4 - eq2
        e = eq6 - eq2
        f = eq2 - d * x0 - e * y0

        return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])

    def _get_matplotlib_transform(
        self,
        coordinate_func: Callable[[float, float], tuple[float, float]],
        location: tuple[float, float],
        ax: plt.Axes | None,
    ) -> matplotlib.transforms.Transform:
        transform = matplotlib.transforms.Affine2D(
            self._get_local_affine_transform_matrix(coordinate_func, location)
        )
        if ax:
            transform = transform + ax.transData
        return transform

    def matplotlib_radec2km_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        """
        Get matplotlib transform which converts between coordinate systems.

        For example, :func:`matplotlib_radec2km_transform` can be used to plot data
        in RA/Dec coordinates directly on a plot in the km coordinate system: ::

            # Create plot in km coordinates
            ax = body.plot_wireframe_km()

            # Plot data using RA/Dec coordinates with the transform
            ax.scatter(
                body.target_ra,
                body.target_dec,
                transform=body.matplotlib_radec2km_transform(ax),
                color='r',
            )
            # This is (almost exactly) equivalent to using
            # ax.scatter(*body.radec2km(body.target_ra, body.target_dec), color='r')

        A full set of transformations are available in :class:`Body` (below) and
        :class:`BodyXY` to convert between various coordinate systems. These are
        mainly convenience functions to simplify plotting data in different coordinate
        systems, and may not be exact in some extreme geometries, due to the non-linear
        nature of spherical coordinates.

        .. note::

            The transformations are performed as affine transformations, which are
            linear transformations. This means that the transformations may be inexact
            at large distances from the target body, or near the celestial poles for
            `radec` coordinates.

            For the vast majority of purposes, these matplotlib transformations are
            accurate, but if you are working with extreme geometries or require exact
            transformations you should convert the coordinates manually before plotting
            (e.g. using :func:`radec2km` rather than
            :func:`matplotlib_radec2km_transform`).

            The `km`, `angular` (with the default values for the origin) and `xy`
            coordinate systems are all affine transformations of each other, so the
            matplotlib transformations between these coordinate systems should be exact.

        Args:
            ax: Optionally specify a matplotlib axis to return
                `transform_radec2km + ax.transData`. This value can then be used in the
                `transform` keyword argument of a Matplotlib function without any
                further modification.

        Returns:
            Matplotlib transformation from `radec` to `km` coordinates.
        """
        # XXX test transforms
        return self._get_matplotlib_transform(
            self.radec2km, (self.target_ra, self.target_dec), ax
        )

    def matplotlib_km2radec_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        return self._get_matplotlib_transform(self.km2radec, (0.0, 0.0), ax)

    def matplotlib_radec2angular_transform(
        self, ax: Axes | None = None, **angular_kwargs: Unpack[_AngularCoordinateKwargs]
    ) -> matplotlib.transforms.Transform:
        return self._get_matplotlib_transform(
            functools.partial(self.radec2angular, **angular_kwargs),
            (self.target_ra, self.target_dec),
            ax,
        )

    def matplotlib_angular2radec_transform(
        self, ax: Axes | None = None, **angular_kwargs: Unpack[_AngularCoordinateKwargs]
    ) -> matplotlib.transforms.Transform:
        return self._get_matplotlib_transform(
            functools.partial(self.angular2radec, **angular_kwargs),
            (0.0, 0.0),
            ax,
        )

    @staticmethod
    def _get_wireframe_kw(
        *,
        base_formatting: dict[str, Any] | None = None,
        common_formatting: dict[str, Any] | None = None,
        formatting: dict[_WireframeComponent, dict[str, Any]] | None = None,
    ) -> dict[_WireframeComponent, dict[str, Any]]:
        formatting = formatting or {}
        base_formatting = base_formatting or {}
        common_formatting = common_formatting or {}

        # deal with passing plot_wireframe_radec args to e.g. plot_wireframe_km
        for k in ('show', 'dms_ticks'):
            common_formatting.pop(k, None)

        kwargs: dict[_WireframeComponent, dict[str, Any]] = defaultdict(dict)
        for k in set(DEFAULT_WIREFRAME_FORMATTING.keys()) | set(formatting.keys()):
            kwargs[k] = (
                base_formatting
                | DEFAULT_WIREFRAME_FORMATTING.get('all', {})
                | DEFAULT_WIREFRAME_FORMATTING.get(k, {})
                | common_formatting
                | formatting.get('all', {})
                | formatting.get(k, {})
            )
        return kwargs

    def _plot_wireframe(
        self,
        *,
        coordinate_func: Callable[[float, float], tuple[float, float]],
        transform: matplotlib.transforms.Transform | None,
        ax: Axes | None = None,
        label_poles: bool = True,
        add_title: bool = True,
        grid_interval: float = 30,
        grid_lat_limit: float = 90,
        indicate_equator: bool = False,
        indicate_prime_meridian: bool = False,
        formatting: dict[_WireframeComponent, dict[str, Any]] | None = None,
        **common_formatting,
    ) -> Axes:
        """
        Plot generic wireframe representation of the observation.

        Args:
            coordinate_func: Function to convert RA/Dec coordinates to the desired
                coordinate system. Takes two arguments (RA, Dec) and returns two
                values (x, y).
            transform: Matplotlib transform to apply to the plotted data, after
                transforming with `coordinate_func`.
        """
        if ax is None:
            ax = cast(Axes, plt.gca())
        if transform is None:
            transform = ax.transData
        else:
            transform = transform + ax.transData

        def array_func(
            ras: np.ndarray, decs: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """Transform arrays of coords with coordinate_func"""
            xs, ys = zip(*(coordinate_func(ra, dec) for ra, dec in zip(ras, decs)))
            return np.array(xs), np.array(ys)

        kwargs = self._get_wireframe_kw(
            base_formatting=dict(transform=transform),
            common_formatting=common_formatting,
            formatting=formatting,
        )

        lons = np.arange(0, 360, grid_interval)
        for lon, (ra, dec) in zip(
            lons, self.visible_lon_grid_radec(lons, lat_limit=grid_lat_limit)
        ):
            ax.plot(
                *array_func(ra, dec),
                **kwargs['grid']
                | (
                    kwargs['prime_meridian']
                    if lon == 0 and indicate_prime_meridian
                    else {}
                ),
            )
        lats = np.arange(-90, 90, grid_interval)
        for lat, (ra, dec) in zip(
            lats, self.visible_lat_grid_radec(lats, lat_limit=grid_lat_limit)
        ):
            ax.plot(
                *array_func(ra, dec),
                **kwargs['grid']
                | (kwargs['equator'] if lat == 0 and indicate_equator else {}),
            )

        ax.plot(*array_func(*self.limb_radec()), **kwargs['limb'])
        ax.plot(*array_func(*self.terminator_radec()), **kwargs['terminator'])

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination()
        ax.plot(*array_func(ra_day, dec_day), **kwargs['limb_illuminated'])

        if label_poles:
            for lon, lat, s in self.get_poles_to_plot():
                x, y = coordinate_func(*self.lonlat2radec(lon, lat))
                ax.text(x, y, s, **kwargs['pole'])

        for lon, lat in self.coordinates_of_interest_lonlat:
            if self.test_if_lonlat_visible(lon, lat):
                x, y = coordinate_func(*self.lonlat2radec(lon, lat))
                ax.scatter(x, y, **kwargs['coordinate_of_interest_lonlat'])
        for ra, dec in self.coordinates_of_interest_radec:
            ax.scatter(
                *coordinate_func(ra, dec), **kwargs['coordinate_of_interest_radec']
            )

        for radius in self.ring_radii:
            x, y = array_func(*self.ring_radec(radius))
            ax.plot(x, y, **kwargs['ring'])

        for body in self.other_bodies_of_interest:
            x, y = coordinate_func(body.target_ra, body.target_dec)
            label = body.target
            hidden = not self.test_if_other_body_visible(body)
            if hidden:
                label = f'({label})'
            ax.text(
                x,
                y,
                label + '\n',
                **kwargs['other_body_of_interest_label']
                | (kwargs['hidden_other_body_of_interest_label'] if hidden else {}),
            )
            ax.scatter(
                x,
                y,
                **kwargs['other_body_of_interest_marker']
                | (kwargs['hidden_other_body_of_interest_marker'] if hidden else {}),
            )

        if add_title:
            ax.set_title(self.get_description(multiline=True))
        return ax

    def plot_wireframe_radec(
        self,
        ax: Axes | None = None,
        *,
        dms_ticks: bool = True,
        add_axis_labels: bool = True,
        aspect_adjustable: Literal['box', 'datalim'] = 'datalim',
        show: bool = False,
        **wireframe_kwargs: Unpack[_WireframeKwargs],
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation using RA/Dec sky
        coordinates.

        See also :func:`plot_wireframe_km`, :func:`plot_wireframe_angular` and
        :func:`BodyXY.plot_wireframe_xy` to plot the wireframe in other coordinate
        systems.

        To plot a wireframe with the default appearance, simply use: ::

            body.plot_wireframe_radec()

        To customise the appearance of the plot, you can use the `formatting` and
        `**kwargs` arguments which can be used to pass arguments to the Matplotlib
        plotting functions. The `formatting` argument can be used to customise
        individual components, and the `**kwargs` argument can be used to customise
        all components at once.

        For example, to change the colour of the entire wireframe to red, you can
        use: ::

            body.plot_wireframe_radec(color='r')

        To change just the plotted terminator and dayside limb to red, use: ::

            body.plot_wireframe_radec(
                formatting={
                    'terminator': {'color': 'r'},
                    'limb_illuminated': {'color': 'r'},
                },
            )

        The order of precedence for the formatting is the `formatting` argument, then
        `**kwargs`, then the default formatting. For example, the following plot will
        be red with a thin blue grid and green poles: ::

            body.plot_wireframe_radec(
                color='r',
                formatting={
                    'grid': {'color': 'b', 'linewidth': 0.5, 'linestyle': '-'},
                    'pole': {'color': 'g'},
                },
            )

        Individual components can be hidden by setting `visible` to `False`. For
        example, to hide the terminator, use: ::

            body.plot_wireframe_radec(formatting={'terminator': {'visible': False}})

        The default formatting is defined in :data:`DEFAULT_WIREFRAME_FORMATTING`. This
        can be modified after importing PlanetMapper to change the default appearance of
        all wireframes: ::

            import planetmapper
            planetmapper.DEFAULT_WIREFRAME_FORMATTING['grid']['color'] = 'b'
            planetmapper.DEFAULT_WIREFRAME_FORMATTING['grid']['linestyle'] = '--'

            body.plot_wireframe_radec() # This would have a blue dashed grid
            body.plot_wireframe_radec(color='r') # This would be red with a dashed grid

        .. note::

            The plot may appear warped or distorted if the target is near the celestial
            pole (i.e. the target's declination is near 90° or -90°). This is due to the
            spherical nature of the RA/Dec coordinate system, which is impossible to
            represent perfectly on a 2D cartesian plot.

            :func:`plot_wireframe_angular` can be used as an alternative to
            :func:`plot_wireframe_radec` to plot the wireframe without distortion from
            the choice of coordinte system. By default, the `angular` coordinate system
            is centred on the target body, which minimises any distortion, but the
            origin and rotation of the `angular` coordinates can also be customised as
            needed (e.g. to align it with an instrument's field of view).

        Args:
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), uses
                `plt.gca()` to get the currently active axis.
            label_poles: Toggle labelling the poles of the target body.
            add_title: Add title generated by :func:`get_description` to the axis.
            add_axis_labels: Add axis labels.
            grid_interval: Spacing between grid lines in degrees.
            grid_lat_limit: Latitude limit for gridlines. For example, if
                `grid_lat_limit=60`, then gridlines will only be plotted for latitudes
                between 60°N and 60°S (inclusive). This can be useful to reduce visual
                clutter around the poles.
            indicate_equator: Toggle indicating the equator with a solid line.
            indicate_prime_meridian: Toggle indicating the prime meridian with a solid
                line.
            aspect_adjustable: Set `adjustable` parameter when setting the aspect ratio.
                Passed to :func:`matplotlib.axes.Axes.set_aspect`.
            dms_ticks: Toggle between showing ticks as degrees, minutes and seconds
                (e.g. 12°34′56″) or decimal degrees (e.g. 12.582). This argument is only
                applicable for :func:`plot_wireframe_radec`.
            show: Toggle immediately showing the plotted figure with `plt.show()`.
            formatting: Dictionary of formatting options for the wireframe components.
                The keys of this dictionary are the names of the wireframe components
                and the values are dictionaries of keyword arguments to pass to the
                Matplotlib plotting function for that component. For example, to set the
                `color` of the plotted rings to red, you could use::

                    body.plot_wireframe_radec(formatting={'ring': {'color': 'r'}})

                The following components can be formatted: `grid`, `equator`,
                `prime_meridian`, `limb`, `limb_illuminated`, `terminator`, `ring`,
                `pole`, `coordinate_of_interest_lonlat`, `coordinate_of_interest_radec`,
                `other_body_of_interest_marker`, `other_body_of_interest_label`,
                `hidden_other_body_of_interest_marker`,
                `hidden_other_body_of_interest_label`.

            **kwargs: Additional arguments are passed to Matplotlib plotting functions
                for all components. This is useful for specifying properties like
                `color` to customise the entire wireframe rather than a single
                component. For example, to make the entire wireframe red, you could
                use::

                    body.plot_wireframe_radec(color='r')

        Returns:
            The axis containing the plotted wireframe.
        """
        # XXX add note about warping at high declinations
        # TODO make aspect adjustable accept None to skip setting aspect
        ax = self._plot_wireframe(
            coordinate_func=lambda ra, dec: (ra, dec),
            transform=None,
            ax=ax,
            **wireframe_kwargs,
        )

        utils.format_radec_axes(
            ax,
            self.target_dec,
            dms_ticks=dms_ticks,
            add_axis_labels=add_axis_labels,
            aspect_adjustable=aspect_adjustable,
        )

        if show:
            plt.show()
        return ax

    def plot_wireframe_km(
        self,
        ax: Axes | None = None,
        *,
        add_axis_labels: bool = True,
        aspect_adjustable: Literal['box', 'datalim'] = 'datalim',
        show: bool = False,
        **wireframe_kwargs: Unpack[_WireframeKwargs],
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation on a target centred
        frame. See :func:`plot_wireframe_radec` for details of accepted arguments.

        Returns:
            The axis containing the plotted wireframe.
        """
        ax = self._plot_wireframe(
            coordinate_func=lambda ra, dec: self.radec2km(ra, dec),
            transform=None,
            ax=ax,
            **wireframe_kwargs,
        )
        if add_axis_labels:
            ax.set_xlabel('Projected distance (km)')
            ax.set_ylabel('Projected distance (km)')
            ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.set_aspect(1, adjustable=aspect_adjustable)

        if show:
            plt.show()
        return ax

    def plot_wireframe_angular(
        self,
        ax: Axes | None = None,
        *,
        origin_ra: float | None = None,
        origin_dec: float | None = None,
        coordinate_rotation: float = 0.0,
        add_axis_labels: bool = True,
        aspect_adjustable: Literal['box', 'datalim'] = 'datalim',
        show: bool = False,
        **wireframe_kwargs: Unpack[_WireframeKwargs],
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation on a relative angular
        coordinate frame. See :func:`plot_wireframe_radec` for details of accepted
        arguments.

        The `origin_ra`, `origin_dec` and `coordinate_rotation` arguments can be used to
        customise the origin and rotation of the relative angular coordinate frame (see
        see :func:`radec2angular`). For example, to plot the wireframe with the origin
        at the north pole, you can use: ::

            body.plot_wireframe_angular(origin_ra=0, origin_dec=90)

        .. note::

            If custom values for `origin_ra` and `origin_dec` are provided, the plot may
            appear warped or distorted if the target is a large distance from the
            origin. This is because spherical coordinates are impossible to represent
            perfectly on a 2D cartesian plot. By default, the `angular` coordinates are
            centred on the target body, minimising any distortion.

        Returns:
            The axis containing the plotted wireframe.
        """
        # XXX test
        # XXX document in examples
        ax = self._plot_wireframe(
            coordinate_func=lambda ra, dec: self.radec2angular(
                ra,
                dec,
                origin_ra=origin_ra,
                origin_dec=origin_dec,
                coordinate_rotation=coordinate_rotation,
            ),
            transform=None,
            ax=ax,
            **wireframe_kwargs,
        )
        if add_axis_labels:
            ax.set_xlabel('Angular distance (arcsec)')
            ax.set_ylabel('Angular distance (arcsec)')
        ax.set_aspect(1, adjustable=aspect_adjustable)

        if show:
            plt.show()
        return ax
