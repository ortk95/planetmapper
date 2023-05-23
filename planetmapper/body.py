import datetime
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
    SpiceKERNELVARNOTFOUND,
    SpiceSPKINSUFFDATA,
)

from . import data_loader, utils
from .base import BodyBase, Numeric
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

        self._matrix_km2radec = None
        self._matrix_radec2km = None
        self._mpl_transform_km2radec_radians = None
        self._mpl_transform_radec2km_radians = None
        self._mpl_transform_km2radec = None
        self._mpl_transform_radec2km = None

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
    ) -> 'Body':
        ...

    @overload
    def create_other_body(
        self, other_target: str | int, fallback_to_basic_body: bool = True
    ) -> 'Body|BasicBody':
        ...

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
        return spice.radrec(1, ra, dec)

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
        )

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
        if self._matrix_km2radec is None:
            r_km = self.r_eq
            r_radians = np.arcsin(r_km / self.target_distance)
            s = r_radians / r_km
            theta = -np.deg2rad(self.north_pole_angle())
            direction_matrix = np.array([[-1, 0], [0, 1]])
            stretch_matrix = np.array(
                [[1 / np.abs(np.cos(self._target_dec_radians)), 0], [0, 1]]
            )
            rotation_matrix = self._rotation_matrix_radians(theta)
            transform_matrix_2x2 = s * np.matmul(stretch_matrix, rotation_matrix)
            transform_matrix_2x2 = np.matmul(transform_matrix_2x2, direction_matrix)

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
        """
        Convert distance in target plane to RA/Dec sky coordinates for the observer.

        Args:
            km_x: Distance in target plane in km in the East-West direction.
            km_y: Distance in target plane in km in the North-South direction.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point.
        """
        return self._radian_pair2degrees(*self._km2radec_radians(km_x, km_y))

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
        return self._radec2km_radians(*self._degree_pair2radians(ra, dec))

    def km2lonlat(self, km_x: float, km_y: float, **kwargs) -> tuple[float, float]:
        """
        Convert distance in target plane to longitude/latitude coordinates on the target
        body.

        Args:
            km_x: Distance in target plane in km in the East-West direction.
            km_y: Distance in target plane in km in the North-South direction.
            **kwargs: Additional arguments are passed to :func:`Body.radec2lonlat`.

        Returns:
            `(lon, lat)` tuple containing the longitude and latitude of the point. If
            the provided km coordinates are missing the target body, then the `lon`
            and `lat` values will both be NaN (see :func:`Body.radec2lonlat`).
        """
        return self.radec2lonlat(*self.km2radec(km_x, km_y), **kwargs)

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
        return self.radec2km(*self.lonlat2radec(lon, lat))

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
        """
        Get matplotlib transform which converts RA/Dec sky coordinates to target plane
        distance coordinates.

        Args:
            ax: Optionally specify a matplotlib axis to return
                `transform_radec2km + ax.transData`. This value can then be used in the
                `transform` keyword argument of a Matplotlib function without any
                further modification.

        Returns:
            Matplotlib transformation from `radec` to `km` coordinates.
        """
        if self._mpl_transform_radec2km is None:
            transform_rad2deg = matplotlib.transforms.Affine2D().scale(np.deg2rad(1))
            self._mpl_transform_radec2km = (
                transform_rad2deg + self._get_matplotlib_radec2km_transform_radians()
            )
        transform = self._mpl_transform_radec2km
        if ax:
            transform = transform + ax.transData
        return transform

    def matplotlib_km2radec_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        """
        Get matplotlib transform which converts target plane distance coordinates to
        RA/Dec sky coordinates.

        Args:
            ax: Optionally specify a matplotlib axis to return
                `transform_km2radec + ax.transData`. This value can then be used in the
                `transform` keyword argument of a Matplotlib function without any
                further modification.

        Returns:
            Matplotlib transformation from `km` to `radec` coordinates.
        """
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
        raise ValueError(f'Unknown occultation code: {occultation}')

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
        a = np.cos(phase_radians) - np.cos(emission_radians) * np.cos(incidence_radians)  # type: ignore
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
        self, lons: list[float] | np.ndarray, npts: int = 60
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
        self, lats: list[float] | np.ndarray, npts: int = 120
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
        transform: None | matplotlib.transforms.Transform,
        ax: Axes | None = None,
        label_poles: bool = True,
        add_title: bool = True,
        grid_interval: float = 30,
        indicate_equator: bool = False,
        indicate_prime_meridian: bool = False,
        formatting: dict[_WireframeComponent, dict[str, Any]] | None = None,
        **common_formatting,
    ) -> Axes:
        """Plot generic wireframe representation of the observation"""
        if ax is None:
            ax = cast(Axes, plt.gca())
        if transform is None:
            transform = ax.transData
        else:
            transform = transform + ax.transData

        kwargs = self._get_wireframe_kw(
            base_formatting=dict(transform=transform),
            common_formatting=common_formatting,
            formatting=formatting,
        )

        lons = np.arange(0, 360, grid_interval)
        for lon, (ra, dec) in zip(lons, self.visible_lon_grid_radec(lons)):
            ax.plot(
                ra,
                dec,
                **kwargs['grid']
                | (
                    kwargs['prime_meridian']
                    if lon == 0 and indicate_prime_meridian
                    else {}
                ),
            )
        lats = np.arange(-90, 90, grid_interval)
        for lat, (ra, dec) in zip(lats, self.visible_lat_grid_radec(lats)):
            ax.plot(
                ra,
                dec,
                **kwargs['grid']
                | (kwargs['equator'] if lat == 0 and indicate_equator else {}),
            )

        ax.plot(*self.limb_radec(), **kwargs['limb'])
        ax.plot(*self.terminator_radec(), **kwargs['terminator'])

        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination()
        ax.plot(ra_day, dec_day, **kwargs['limb_illuminated'])

        if label_poles:
            for lon, lat, s in self.get_poles_to_plot():
                ra, dec = self.lonlat2radec(lon, lat)
                ax.text(ra, dec, s, **kwargs['pole'])

        for lon, lat in self.coordinates_of_interest_lonlat:
            if self.test_if_lonlat_visible(lon, lat):
                ra, dec = self.lonlat2radec(lon, lat)
                ax.scatter(ra, dec, **kwargs['coordinate_of_interest_lonlat'])
        for ra, dec in self.coordinates_of_interest_radec:
            ax.scatter(ra, dec, **kwargs['coordinate_of_interest_radec'])

        for radius in self.ring_radii:
            ra, dec = self.ring_radec(radius)
            ax.plot(ra, dec, **kwargs['ring'])

        for body in self.other_bodies_of_interest:
            ra = body.target_ra
            dec = body.target_dec
            label = body.target
            hidden = not self.test_if_other_body_visible(body)
            if hidden:
                label = f'({label})'
            ax.text(
                ra,
                dec,
                label + '\n',
                **kwargs['other_body_of_interest_label']
                | (kwargs['hidden_other_body_of_interest_label'] if hidden else {}),
            )
            ax.scatter(
                ra,
                dec,
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

        See also :func:`plot_wireframe_km` and :func:`BodyXY.plot_wireframe_xy` to plot
        the wireframe in other coordinate systems.

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

        Args:
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), uses
                `plt.gca()` to get the currently active axis.
            label_poles: Toggle labelling the poles of the target body.
            add_title: Add title generated by :func:`get_description` to the axis.
            add_axis_labels: Add axis labels.
            grid_interval: Spacing between grid lines in degrees.
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
        ax = self._plot_wireframe(transform=None, ax=ax, **wireframe_kwargs)

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

        transform = self.matplotlib_radec2km_transform()
        ax = self._plot_wireframe(transform=transform, ax=ax, **wireframe_kwargs)
        if add_axis_labels:
            ax.set_xlabel('Projected distance (km)')
            ax.set_ylabel('Projected distance (km)')
            ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.set_aspect(1, adjustable=aspect_adjustable)

        if show:
            plt.show()
        return ax
