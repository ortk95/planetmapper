import datetime
import io
import math
import warnings
from collections.abc import Iterator
from typing import Any, Callable, Literal, NamedTuple, Protocol, TypedDict, cast

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pyproj
import scipy.interpolate
import scipy.ndimage
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from spiceypy.utils.exceptions import NotFoundError

from .base import (
    FloatOrArray,
    _as_readonly_view,
    _cache_clearable_result,
    _cache_stable_result,
    _return_readonly_array,
)
from .body import (
    AngularCoordinateKwargs,
    Body,
    LonLatGridKwargs,
    WireframeComponent,
    WireframeKwargs,
    _adjust_surface_altitude_decorator,
    _AdjustedSurfaceAltitude,
    _cache_clearable_alt_dependent_result,
)
from .progress import progress_decorator


class MapKwargs(TypedDict, total=False):
    """
    Class to help type hint keyword arguments of mapping functions.

    See :func:`BodyXY.generate_map_coordinates` for more details.
    """

    projection: str
    degree_interval: float
    lon: float
    lat: float
    size: int
    lon_coords: np.ndarray
    lat_coords: np.ndarray
    projection_x_coords: np.ndarray
    projection_y_coords: np.ndarray | None
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None
    alt: float


_MapKwargs = MapKwargs  # keep for backward compatibility


class _BackplaneMapGetter(Protocol):
    def __call__(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray: ...


class Backplane(NamedTuple):
    """
    NamedTuple containing information about a backplane.

    Backplanes provide a way to generate and save additional information about an
    observation, such as the longitudes/latitudes corresponding to each pixel in the
    observed image. This class provides a standardised way to store a backplane
    generation function, along with some metadata (`name` and `description`) which
    describes what the backplane represents.

    See also :attr:`BodyXY.backplanes`.

    Args:
        name: Short name identifying the backplane. This is used as the `EXTNAME` for
            the backplane when saving FITS files in :func:`Observation.save`.
        description: More detailed description of the backplane (e.g. including units).
        get_img: Function which takes no arguments returns a numpy array containing a
            backplane image when called. This should generally be a method such as
            :func:`BodyXY.get_lon_img`.
        get_map: Function returns a numpy array containing a map of backplane values
            when called. This should take map projection keyword arguments, as described
            in :func:`BodyXY.generate_map_coordinates`. This function should generally
            be a method such as :func:`BodyXY.get_lon_map`.
    """

    name: str
    description: str
    get_img: Callable[[], np.ndarray]
    get_map: _BackplaneMapGetter


class ProjStringError(ValueError):
    """Exception raised for bad or inconsistent proj projection strings."""


class BodyXY(Body):
    """
    Class representing an astronomical body imaged at a specific time.

    This is a subclass of :class:`Body` with additional methods to interact with the
    `image pixel coordinates`_ frame `xy`. This class assumes the tangent plane
    approximation is valid for the conversion between pixel coordinates `xy` and `RA/Dec
    celestial coordinates`_ `radec`.

    The `xy` ↔ `radec` conversion is performed by setting the pixel coordinates of the
    centre of the planet's disc `(x0, y0)`, the (equatorial) pixel radius of the disc
    `r0` and the `rotation` of the disc. These disc parameters can be adjusted using
    methods such as :func:`set_x0` and retrieved using methods such as :func:`get_x0`.
    It is important to note that conversions involving `xy` image pixel coordinates
    (e.g. backplane image generation) will produce different results before and after
    these disc parameter values are adjusted.

    For larger images, the generation of backplane images can be computationally
    intensive and take a large amount of time to execute. Therefore, intermediate
    results are cached to make sure that the slowest parts of code are only called when
    needed. This cache is managed automatically, so the user never needs to worry about
    dealing with it. The cache behaviour can be seen in apparently similar lines of
    code having very different execution times: ::

        # Create a new object
        body = planetmapper.BodyXY('Jupiter', '2000-01-01', sz=500)
        body.set_disc_params(x0=250, y0=250, r0=200)
        # At this point, the cache is completely empty

        # The intermediate results used in generating the incidence angle backplane
        # are cached, speeding up any future calculations which use these
        # intermediate results:
        body.get_backplane_img('INCIDENCE') # Takes ~10s to execute
        body.get_backplane_img('INCIDENCE') # Executes instantly
        body.get_backplane_img('EMISSION') # Executes instantly

        # When any of the disc parameters are changed, the xy <-> radec conversion
        # changes so the cache is automatically cleared (as the cached intermediate
        # results are no longer valid):
        body.set_r0(190) # This automatically clears the cache
        body.get_backplane_img('EMISSION') # Takes ~10s to execute
        body.get_backplane_img('INCIDENCE') # Executes instantly

    You can optionally display a progress bar for long running processes like backplane
    generation by `show_progress=True` when creating a `BodyXY` instance (or any other
    instance which derives from :class:`SpiceBase`).

    The size of the image can be specified by using the `nx` and `ny` parameters to
    specify the number of pixels in the x and y dimensions of the image respectively.
    If `nx` and `ny` are equal (i.e. the image is square), then the parameter `sz` can
    be used instead to set both `nx` and `ny`, where `BodyXY(..., sz=50)` is equivalent
    to `BodyXY(..., nx=50, ny=50)`.

    If `nx` and `ny` are not set, then some functionality (such as generating backplane
    images) will not be available and will raise a `ValueError` if called.

    :class:`BodyXY` instances are mutable and therefore not hashable, meaning that they
    cannot be used as dictionary keys. :func:`to_body` can be used to create a
    :class:`Body` instance which is hashable.

    Args:
        target: Name of target body, passed to :class:`Body`.
        utc: Time of observation, passed to :class:`Body`.
        observer: Name of observing body, passed to :class:`Body`.
        nx: Number of pixels in the x dimension of the image.
        ny: Number of pixels in the y dimension of the image.
        sz: Convenience parameter to set both `nx` and `ny` to the same value.
            `BodyXY(..., sz=50)` is equivalent to `BodyXY(..., nx=50, ny=50)`. If `sz`
            is defined along with `nx` or `ny` then a `ValueError` is raised.
        **kwargs: Additional arguments are passed to :class:`Body`.
    """

    def __init__(
        self,
        target: str,
        utc: str | datetime.datetime | float | None = None,
        observer: str | int = 'EARTH',
        nx: int = 0,
        ny: int = 0,
        *,
        sz: int | None = None,
        **kwargs,
    ) -> None:
        # Validate inputs
        if sz is not None:
            if nx != 0 or ny != 0:
                raise ValueError('`sz` cannot be used if `nx` and/or `ny` are nonzero')
            nx = sz
            ny = sz

        super().__init__(target, utc, observer, **kwargs)

        # Document instance variables
        self.backplanes: dict[str, Backplane]
        """
        Dictionary containing registered backplanes which can be used to calculate
        properties (e.g. longitude/latitude, illumination angles etc.) for each pixel in
        the image.

        By default, this dictionary contains a series of 
        :ref:`default backplanes <default backplanes>`. These can be summarised using 
        :func:`print_backplanes`. Custom backplanes can be added using 
        :func:`register_backplane`.

        Generated backplane images can be easily retrieved using 
        :func:`get_backplane_img` and plotted using :func:`plot_backplane_img`. 
        Similarly, backplane maps cen be retrieved using :func:`get_backplane_map` and
        plotted using :func:`plot_backplane_map`. 
        
        This dictionary of backplanes can also be used directly if more customisation is
        needed: ::

            # Retrieve the image containing right ascension values
            ra_image = body.backplanes['RA'].get_img()

            # Retrieve the map containing emission angles on the target's surface
            emission_map = body.backplanes['EMISSION'].get_img()

            # Print the description of the doppler factor backplane
            print(body.backplanes['DOPPLER'].description)

            # Remove the distance backplane from an instance
            del body.backplanes['DISTANCE']
            
            # Print summary of all registered backplanes
            print(f'{len(body.backplanes)} backplanes currently registered:')
            for bp in body.backplanes.values():
                print(f'    {bp.name}: {bp.description}')
        
        See :class:`Backplane` for more detail about interacting with the backplanes
        directly.

        Note that a generated backplane image will depend on the disc parameters 
        `(x0, y0, r0, rotation)` at the time the backplane is generated (e.g. calling 
        `body.backplanes['LAT-GRAPHIC'].get_img()` or using :func:`get_backplane_img`).
        Generating the same backplane when there are different disc parameter values will
        produce a different image.

        This dictionary is used to define the backplanes saved to the output FITS file
        in :func:`Observation.save`.
        """

        # Run setup
        self._nx: int = nx
        self._ny: int = ny

        self._x0: float = 0
        self._y0: float = 0
        self._r0: float = 10
        self._rotation_radians: float = 0
        self.set_disc_method('default')
        self._default_disc_method = 'manual'

        self._mpl_transform_xy2angular_fixed: matplotlib.transforms.Affine2D | None = (
            None
        )
        self._mpl_transform_angular_fixed2xy: matplotlib.transforms.Affine2D | None = (
            None
        )
        self.backplanes = {}
        self._register_default_backplanes()

        if self._nx > 0 and self._ny > 0:
            # centre disc if dimensions provided
            self.centre_disc()

    @classmethod
    def from_body(
        cls, body: Body, nx: int = 0, ny: int = 0, *, sz: int | None = None
    ) -> Self:
        """
        Create a :class:`BodyXY` instance with the same parameters as a :class:`Body`
        instance.

        Args:
            body: :class:`Body` instance to convert.
            nx: Number of pixels in the x dimension of the image.
            ny: Number of pixels in the y dimension of the image.
            sz: Convenience parameter to set both `nx` and `ny` to the same value.

        Returns:
            :class:`BodyXY` instance with the same parameters as the input :class:`Body`
            instance and the specified image dimensions.
        """
        # pylint: disable=protected-access
        new = cls(**body._get_kwargs(), nx=nx, ny=ny, sz=sz)
        body._copy_options_to_other(new)
        return new

    def to_body(self) -> Body:
        """
        Create a :class:`Body` instance from this :class:`BodyXY` instance.

        Returns:
            :class:`Body` instance with the same parameters as this :class:`BodyXY`
            instance.
        """
        new = Body(**Body._get_kwargs(self))
        Body._copy_options_to_other(self, new)
        return new

    def __repr__(self) -> str:
        return self._generate_repr('target', 'utc', kwarg_keys=['observer', 'nx', 'ny'])

    # BodyXY is mutable, so make it unhashable
    __hash__ = None  # type: ignore

    def _get_equality_tuple(self) -> tuple:
        return (
            self._nx,
            self._ny,
            self._x0,
            self._y0,
            self._r0,
            self._rotation_radians,
            super()._get_equality_tuple(),
        )

    def _get_kwargs(self) -> dict[str, Any]:
        return super()._get_kwargs() | dict(
            nx=self._nx,
            ny=self._ny,
        )

    @classmethod
    def _get_default_init_kwargs(cls) -> dict[str, Any]:
        return dict(
            nx=0,
            ny=0,
            **super()._get_default_init_kwargs(),
        )

    def _copy_options_to_other(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Self
    ) -> None:
        super()._copy_options_to_other(other)
        other.set_disc_params(*self.get_disc_params())
        other.set_disc_method(self.get_disc_method())
        # set_img_size is covered by nx, ny in _get_kwargs, so would be redundant here

    # Coordinate transformations
    @_cache_clearable_result
    def _get_xy2angular_matrix(self) -> np.ndarray:
        # angular coords are centred on the target, so just need to:
        # - convert arcsec to pixels with constant scale factor (s)
        # - rotate by the rotation angle
        # - translate by the target's disc centre
        s = self.get_plate_scale_arcsec()  # arcsec/pixel
        theta_radians = -self._get_rotation_radians()
        transform_matrix_2x2 = s * self._rotation_matrix_radians(theta_radians)
        offset_vector = -transform_matrix_2x2.dot(
            np.array([self.get_x0(), self.get_y0()])
        )
        transform_matrix_3x3 = np.identity(3)
        transform_matrix_3x3[:2, :2] = transform_matrix_2x2
        transform_matrix_3x3[:2, 2] = offset_vector
        return transform_matrix_3x3

    @_cache_clearable_result
    def _get_angular2xy_matrix(self) -> np.ndarray:
        return np.linalg.inv(self._get_xy2angular_matrix())

    def _xy2obsvec_norm(self, x: float, y: float) -> np.ndarray:
        a = self._get_xy2angular_matrix().dot(np.array([x, y, 1.0]))
        return self._angular2obsvec_norm(a[0], a[1])

    def _obsvec2xy(self, obsvec: np.ndarray) -> tuple[float, float]:
        angular_x, angular_y = self._obsvec2angular(obsvec)
        v = self._get_angular2xy_matrix().dot(np.array([angular_x, angular_y, 1.0]))
        return v[0], v[1]

    # Composite transformations
    def xy2radec(
        self, x: FloatOrArray, y: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `image pixel coordinates`_ to `RA/Dec celestial coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            x: Image pixel coordinate(s) in the x direction.
            y: Image pixel coordinate(s) in the y direction.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point(s).
        """
        return self._maybe_transform_as_arrays(self._xy2radec, x, y)

    def _xy2radec(self, x: float, y: float) -> tuple[float, float]:
        return self._obsvec2radec(self._xy2obsvec_norm(x, y))

    def radec2xy(
        self, ra: FloatOrArray, dec: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `RA/Dec celestial coordinates`_ to `image pixel coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            ra: Right ascension of point(s) in the sky of the observer
            dec: Declination of point(s) in the sky of the observer.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point(s).
        """
        return self._maybe_transform_as_arrays(self._radec2xy, ra, dec)

    def _radec2xy(self, ra: float, dec: float) -> tuple[float, float]:
        return self._obsvec2xy(self._radec2obsvec_norm(ra, dec))

    def xy2lonlat(
        self, x: FloatOrArray, y: FloatOrArray, *, not_found_nan=True, alt: float = 0.0
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `image pixel coordinates`_ to `longitude/latitude coordinates`_ on the
        target body.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            x: Image pixel coordinate(s) in the x direction.
            y: Image pixel coordinate(s) in the y direction.
            not_found_nan: Controls the behaviour when the input `x` and `y` coordinates
                are missing the target body.
            alt: Altitude of returned `(lon, lat)` point above the surface of the target
                body in km.

        Returns:
            `(lon, lat)` tuple containing the planetographic longitude/latitude of
            the point(s). If the provided pixel coordinates are missing the target body,
            and `not_found_nan` is `True`, then the `lon` and `lat` values will both be
            NaN.

        Raises:
            NotFoundError: if the input `x` and `y` coordinates are missing the target
                body and `not_found_nan` is `False`.
        """
        return self._maybe_transform_as_arrays(
            self._xy2lonlat, x, y, not_found_nan=not_found_nan, alt=alt
        )

    def _xy2lonlat(
        self, x: float, y: float, *, not_found_nan: bool, alt: float
    ) -> tuple[float, float]:
        return self._obsvec_norm2lonlat(self._xy2obsvec_norm(x, y), not_found_nan, alt)

    def lonlat2xy(
        self,
        lon: FloatOrArray,
        lat: FloatOrArray,
        *,
        alt: float = 0.0,
        not_visible_nan: bool = False,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `longitude/latitude coordinates`_ on the target body to `image pixel
        coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            lon: Planetographic longitude of point(s) on target body.
            lat: Planetographic latitude of point(s) on target body.
            alt: Altitude of point above the surface of the target body in km.
            not_visible_nan: If `True`, then the returned RA/Dec values will be NaN if
                the point is not visible to the observer (e.g. it is on the far side of
                the target). If `False` (the default), then `(ra, dec)` coordinates will
                be returned, even if the point is not directly visible.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point(s).
        """
        return self._maybe_transform_as_arrays(
            self._lonlat2xy, lon, lat, alt=alt, not_visible_nan=not_visible_nan
        )

    def _lonlat2xy(
        self, lon: float, lat: float, *, alt: float, not_visible_nan: bool
    ) -> tuple[float, float]:
        return self._obsvec2xy(
            self._lonlat2obsvec(lon, lat, alt=alt, not_visible_nan=not_visible_nan)
        )

    def xy2km(
        self, x: FloatOrArray, y: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `image pixel coordinates`_ to `distance coordinates`_ in the target
        plane.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            x: Image pixel coordinate(s) in the x direction.
            y: Image pixel coordinate(s) in the y direction.

        Returns:
            `(km_x, km_y)` tuple containing distances in km in the target plane in the
            East-West and North-South directions respectively.
        """
        return self._maybe_transform_as_arrays(self._xy2km, x, y)

    def _xy2km(self, x: float, y: float) -> tuple[float, float]:
        return self._obsvec2km(self._xy2obsvec_norm(x, y))

    def km2xy(
        self, km_x: FloatOrArray, km_y: FloatOrArray
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `distance coordinates`_ in the target plane to `image pixel
        coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            km_x: Distance(s) in target plane in km in the East-West direction.
            km_y: Distance(s) in target plane in km in the North-South direction.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point(s).
        """
        return self._maybe_transform_as_arrays(self._km2xy, km_x, km_y)

    def _km2xy(self, km_x: float, km_y: float) -> tuple[float, float]:
        return self._obsvec2xy(self._km2obsvec_norm(km_x, km_y))

    def xy2angular(
        self,
        x: FloatOrArray,
        y: FloatOrArray,
        **angular_kwargs: Unpack[AngularCoordinateKwargs],
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `image pixel coordinates`_ to `relative angular coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            x: Image pixel coordinate(s) in the x direction.
            y: Image pixel coordinate(s) in the y direction.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`Body.radec2angular` for details.

        Returns:
            `(angular_x, angular_y)` tuple containing the relative angular coordinates
            of the point(s) in arcseconds.
        """
        return self._maybe_transform_as_arrays(self._xy2angular, x, y, **angular_kwargs)

    def _xy2angular(
        self, x: float, y: float, **angular_kwargs: Unpack[AngularCoordinateKwargs]
    ) -> tuple[float, float]:
        return self._obsvec2angular(self._xy2obsvec_norm(x, y), **angular_kwargs)

    def angular2xy(
        self,
        angular_x: FloatOrArray,
        angular_y: FloatOrArray,
        **angular_kwargs: Unpack[AngularCoordinateKwargs],
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """
        Convert `relative angular coordinates`_ to `image pixel coordinates`_.

        The input coordinates can either be floats or NumPy arrays of values. If both
        input coordinates are floats, the output will be a tuple of floats. If either of
        the input coordinates are arrays, the inputs will be `broadcast together
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ and a tuple of
        NumPy arrays will be returned.

        Args:
            angular_x: Angular coordinate(s) in the x direction in arcseconds.
            angular_y: Angular coordinate(s) in the y direction in arcseconds.
            **angular_kwargs: Additional arguments are used to customise the origin and
                rotation of the relative angular coordinates. See
                :func:`Body.radec2angular` for details.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point(s).
        """
        return self._maybe_transform_as_arrays(
            self._angular2xy, angular_x, angular_y, **angular_kwargs
        )

    def _angular2xy(
        self,
        angular_x: float,
        angular_y: float,
        **angular_kwargs: Unpack[AngularCoordinateKwargs],
    ) -> tuple[float, float]:
        return self._obsvec2xy(
            self._angular2obsvec_norm(angular_x, angular_y, **angular_kwargs)
        )

    def _radec_arrs2xy_arrs(
        self, ra_arr: np.ndarray, dec_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = zip(*[self.radec2xy(r, d) for r, d in zip(ra_arr, dec_arr)])
        return np.array(x), np.array(y)

    def _xy2targvec(self, x: float, y: float) -> np.ndarray:
        return self._obsvec_norm2targvec(self._xy2obsvec_norm(x, y))

    # Interface
    def _invalidate_disc_parameters(self) -> None:
        self._clear_cache()
        self.update_transform()

    def set_disc_params(
        self,
        x0: float | None = None,
        y0: float | None = None,
        r0: float | None = None,
        rotation: float | None = None,
    ) -> None:
        """
        Convenience function to set multiple disc parameters at once.

        For example, `body.set_disc_params(x0=10, r0=5)` is equivalent to calling
        `body.set_x0(10)` and `body.set_r0(5)`. Any unspecified parameters will be left
        unchanged.

        Args:
            x0: If specified, passed to :func:`set_x0`.
            y0: If specified, passed to :func:`set_y0`.
            r0: If specified, passed to :func:`set_r0`.
            rotation: If specified, passed to :func:`set_rotation`.
        """
        if x0 is not None:
            self.set_x0(x0)
        if y0 is not None:
            self.set_y0(y0)
        if r0 is not None:
            self.set_r0(r0)
        if rotation is not None:
            self.set_rotation(rotation)

    def adjust_disc_params(
        self, dx: float = 0, dy: float = 0, dr: float = 0, drotation: float = 0
    ) -> None:
        """
        Convenience function to adjust disc parameters.

        This can be used to easily add an offset to disc parameter values without having
        to call multiple `set_...` and `get_...` functions. For example, ::

            body.adjust_disc_params(dy=-3.1, drotation=42)

        is equivalent to ::

            body.set_y0(body.get_y0() - 3.1)
            body.set_rotation(body.get_rotation() + 42)

        The default values of all the arguments are zero, so any unspecified values
        (e.g. `dx` and `dr` in the example above) are unchanged.

        See also :func:`add_arcsec_offset`.

        Args:
            dx: Adjustment to `x0`.
            dy: Adjustment to `y0`.
            dr: Adjustment to `r0`.
            drotation: Adjustment to `rotation`.
        """
        self.set_x0(self.get_x0() + dx)
        self.set_y0(self.get_y0() + dy)
        self.set_r0(self.get_r0() + dr)
        self.set_rotation(self.get_rotation() + drotation)

    def get_disc_params(self) -> tuple[float, float, float, float]:
        """
        Convenience function to get all disc parameters at once.

        Returns:
            `(x0, y0, r0, rotation)` tuple.
        """
        return self.get_x0(), self.get_y0(), self.get_r0(), self.get_rotation()

    def centre_disc(self) -> None:
        """
        Centre the target's planetary disc and make it fill ~90% of the observation.

        This adjusts `x0` and `y0` so that they lie in the centre of the image, and `r0`
        is adjusted so that the disc fills 90% of the shortest side of the image. For
        example, if `nx = 21` and `ny = 31`, then `x0` will be set to 10, `y0` will be
        set to 15 and `r0` will be set to 9. The rotation of the disc is unchanged.
        """
        self.set_x0((self._nx - 1) / 2)
        self.set_y0((self._ny - 1) / 2)
        self.set_r0(0.9 * (min(self.get_x0(), self.get_y0())))
        self.set_disc_method('centre_disc')

    def set_x0(self, x0: float) -> None:
        """
        Args:
            x0: New x `pixel coordinate`_ of the centre of the target body.

        Raises:
            ValueError: if `x0` is not finite.
        """
        if not math.isfinite(x0):
            raise ValueError('x0 must be finite')
        self._x0 = float(x0)
        self._invalidate_disc_parameters()

    def get_x0(self) -> float:
        """
        Returns:
            x `pixel coordinate`_ of the centre of the target body.
        """
        return self._x0

    def set_y0(self, y0: float) -> None:
        """
        Args:
            y0: New y `pixel coordinate`_ of the centre of the target body.

        Raises:
            ValueError: if `y0` is not finite.
        """
        if not math.isfinite(y0):
            raise ValueError('y0 must be finite')
        self._y0 = float(y0)
        self._invalidate_disc_parameters()

    def get_y0(self) -> float:
        """
        Returns:
            y `pixel coordinate`_ of the centre of the target body.
        """
        return self._y0

    def set_r0(self, r0: float) -> None:
        """
        Args:
            r0: New equatorial radius in pixels of the target body.

        Raises:
            ValueError: if `r0` is not greater than zero or `r0` is not finite.
        """
        if not math.isfinite(r0):
            raise ValueError('r0 must be finite')
        if not r0 > 0:
            raise ValueError('r0 must be greater than zero')
        self._r0 = float(r0)
        self._invalidate_disc_parameters()

    def get_r0(self) -> float:
        """
        Returns:
            Equatorial radius in pixels of the target body.
        """
        return self._r0

    def _set_rotation_radians(self, rotation: float) -> None:
        self._rotation_radians = float(rotation % (2 * np.pi))
        self._invalidate_disc_parameters()

    def _get_rotation_radians(self) -> float:
        return self._rotation_radians

    def set_rotation(self, rotation: float) -> None:
        """
        Set the rotation of the target body.

        This rotation defines the angle between the upwards (positive `dec`) direction
        in the RA/Dec sky coordinates and the upwards (positive `y`) direction in the
        `image pixel coordinates`_.

        Args:
            rotation: New rotation of the target body.

        Raises:
            ValueError: if `rotation` is not finite.
        """
        if not math.isfinite(rotation):
            raise ValueError('rotation must be finite')
        self._set_rotation_radians(np.deg2rad(rotation))

    def rotate_north_to_top(self) -> None:
        """
        Set the disc rotation so that the north pole of the target body is at the top of
        the image.

        This is a convenience method, equivalent to calling: ::

            body.set_rotation(-body.north_pole_angle())
        """
        self.set_rotation(-self.north_pole_angle())
        self.set_disc_method('rotate_north_to_top')

    def get_rotation(self) -> float:
        """
        Returns:
            Rotation of the target body.
        """
        return float(np.rad2deg(self._get_rotation_radians()))

    def set_plate_scale_arcsec(self, arcsec_per_px: float) -> None:
        """
        Sets the angular plate scale of the observation by changing `r0`.

        Args:
            arcsec_per_px: Arcseconds per pixel plate scale.
        """
        self.set_r0(self.target_diameter_arcsec / (2 * arcsec_per_px))

    def set_plate_scale_km(self, km_per_px: float) -> None:
        """
        Sets the plate scale of the observation by changing `r0`.

        Args:
            km_per_px: Kilometres per pixel plate scale at the target body.
        """
        self.set_plate_scale_arcsec(km_per_px / self.km_per_arcsec)

    def get_plate_scale_arcsec(self) -> float:
        """
        Returns:
            Plate scale of the observation in arcseconds/pixel.
        """
        return self.target_diameter_arcsec / (2 * self.get_r0())

    def get_plate_scale_km(self) -> float:
        """
        Returns:
            Plate scale of the observation in km/pixel at the target body.
        """
        return self.get_plate_scale_arcsec() * self.km_per_arcsec

    def set_img_size(self, nx: int | None = None, ny: int | None = None) -> None:
        """
        Set the `nx` and `ny` values which specify the number of pixels in the x and y
        dimension of the image respectively. Unspecified values will remain unchanged.

        Args:
            nx: If specified, set the number of pixels in the x dimension.
            ny: If specified, set the number of pixels in the y dimension.

        Raises:
            TypeError: if `set_img_size` is called on an :class:`Observation` instance.
        """
        nx = self._nx if nx is None else int(nx)
        ny = self._ny if ny is None else int(ny)
        if nx < 0 or ny < 0:
            raise ValueError('nx and ny must be non-negative')
        self._nx = nx
        self._ny = ny
        self._clear_cache()

    def get_img_size(self) -> tuple[int, int]:
        """
        Get the size of the image in pixels.

        Returns:
            `(nx, ny)` tuple containing the number of pixels in the x and y dimension of
            the image respectively
        """
        return (self._nx, self._ny)

    def scale_img_size(self, factor: float, *, allow_rounding: bool = False) -> None:
        """
        Scale the image size to increase/decrease the pixel resolution of the image.

        The image size is scaled by multiplying the current `nx` and `ny` values by
        `factor`. The disc parameters `(x0, y0, r0)` are then adjusted to maintain the
        disc's position and size in the image.

        By default, this method will raise an error if the scaled image size does not
        have exactly integer pixel dimensions. This can be changed by setting
        `allow_rounding=True`, which will round the scaled image size up to the nearest
        integer number of pixels. Rounding will mean that the top and right edges of
        the scaled image may not be exactly equivalent to the top and right edges of the
        original image.

        See also :func:`add_img_border`.

        Args:
            factor: Factor by which to scale the image size. For example, a factor of
                2 will double the image size, while a factor of 0.5 will halve the image
                size.
            allow_rounding: If `True`, then the scaled image size will be rounded up to
                the nearest integer number of pixels. If `False` (the default), then
                an error will be raised if the image size cannot be exactly scaled by
                `factor` to an integer number of pixels.

        Raises:
            ValueError: if the image size cannot be exactly scaled by `factor` to an
                integer number of pixels and `allow_rounding` is `False`.
        """
        if factor <= 0:
            raise ValueError('Scaling factor must be greater than zero')

        nx, ny = self.get_img_size()
        nx_f = nx * factor
        ny_f = ny * factor
        nx_ceil = math.ceil(nx_f)
        ny_ceil = math.ceil(ny_f)
        if not allow_rounding and (nx_ceil != nx_f or ny_ceil != ny_f):
            raise ValueError(
                f'Image size ({nx}, {ny}) cannot be exactly scaled by {factor} to an '
                f'integer number of pixels: new size would be ({nx_f}, {ny_f}). '
                'Use `allow_rounding=True` to allow rounding of the image size.'
            )
        self.set_img_size(nx_ceil, ny_ceil)
        self.set_r0(self.get_r0() * factor)
        # Offset accounts for how the disc position varies slightly with the scaling
        # due to the pixel grid extending from 0.5 to nx-0.5 and ny-0.5.
        offset = (factor - 1) / 2
        self.set_x0(self.get_x0() * factor + offset)
        self.set_y0(self.get_y0() * factor + offset)

    def add_img_border(self, border: int) -> None:
        """
        Add a border of pixels around the image.

        This will increase the size of the image by `2 * border` pixels in both the x
        and y dimensions, and will adjust the disc position `(x0, y0)` appropriately.
        If needed, more complex border adjustments can be achieved by using
        :func:`set_img_size` and :func:`set_x0` and :func:`set_y0` directly: ::

            left = 1
            right = 2
            top = 3
            bottom = 4

            nx, ny = body.get_img_size()
            body.set_img_size(
                nx + left + right,
                ny + top + bottom,
            )
            body.set_x0(body.get_x0() + left)
            body.set_y0(body.get_y0() + bottom)

        See also :func:`scale_img_size`.

        Args:
            border: The number of pixels to add (or remove) as a border around the
                image. Negative values will crop the image by that many pixels instead
                of adding a border.
        """
        border = int(border)
        nx, ny = self.get_img_size()
        self.set_img_size(nx + 2 * border, ny + 2 * border)
        self.set_x0(self.get_x0() + border)
        self.set_y0(self.get_y0() + border)

    def set_disc_method(self, method: str) -> None:
        """
        Record the method used to find the coordinates of the target body's disc. This
        recorded method can then be used when metadata is saved, such as in
        :func:`Observation.save`.

        `set_disc_method` is called automatically by functions which find the disc, such
        as :func:`rotate_north_to_top` and :func:`Observation.centre_disc`, so does not
        normally need to be manually called by the user.

        Args:
            method: Short string describing the method used to find the disc.
        """
        # Save disc method to the cahce. It will then be wiped automatically whenever
        # the disc is moved. The key used in the cache contains a space, so will never
        # collide with an auto-generated key from a function name (when @cache_result is
        # used).
        self._cache['disc method'] = method

    def get_disc_method(self) -> str:
        """
        Retrieve the method used to find the coordinates of the target body's disc.

        Returns:
            Short string describing the method.
        """
        return self._cache.get('disc method', self._default_disc_method)

    def add_arcsec_offset(self, dra_arcsec: float = 0, ddec_arcsec: float = 0) -> None:
        """
        Adjust the disc location `(x0, y0)` by applying offsets in arcseconds to the
        RA/Dec celestial coordinates.

        See also :func:`adjust_disc_params`.

        Args:
            dra_arcsec: Offset in arcseconds in the positive right ascension direction.
            ddec_arcsec: Offset in arcseconds in the positive declination direction.
        """
        dra = dra_arcsec / 3600
        ddec = ddec_arcsec / 3600
        ra0, dec0 = self.xy2radec(0, 0)
        dx, dy = self.radec2xy(ra0 + dra, dec0 + ddec)
        self.adjust_disc_params(dx=dx, dy=dy)

    # Limit getters
    def _get_xy_corner_coordinates(self) -> list[tuple[float, float]]:
        return [
            (-0.5, -0.5),
            (-0.5, self._ny - 0.5),
            (self._nx - 0.5, -0.5),
            (self._nx - 0.5, self._ny - 0.5),
        ]

    def _get_img_limits(
        self, func: Callable[[float, float], tuple[float, float]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        xy_lim = [func(x, y) for x, y in self._get_xy_corner_coordinates()]
        xlim = (min(x for x, _ in xy_lim), max(x for x, _ in xy_lim))
        ylim = (min(y for _, y in xy_lim), max(y for _, y in xy_lim))
        return xlim, ylim

    def get_img_limits_radec(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get the limits of the image coordinates in the RA/Dec coordinate system.

        This can be used to set the axis limits of a plot, for example: ::

            xlim, ylim = obs.get_img_limits_radec()
            plt.xlim(*xlim)
            plt.ylim(*ylim)

        See also :func:`get_img_limits_km` and :func:`get_img_limits_xy`.

        Returns:
            `(ra_left, ra_right), (dec_min, dec_max)` tuple containing the minimum and
            maximum RA and Dec coordinates of the pixels in the image respectively.
        """
        xlim, ylim = self._get_img_limits(self.xy2radec)
        xlim = (xlim[1], xlim[0])  # flip xlim because RA increases to the left
        return xlim, ylim

    def get_img_limits_km(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get the limits of the image coordinates in the target centred plane. See
        :func:`get_img_limits_radec` for more details.

        Returns:
            `(km_x_min, km_x_max), (km_y_min, km_y_max)` tuple containing the minimum
            and maximum target plane distance coordinates of the pixels in the image.
        """
        return self._get_img_limits(self.xy2km)

    def get_img_limits_angular(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get the limits of the image coordinates in the relative angular coordinate
        system. See :func:`get_img_limits_radec`

        Returns:
            `(angular_x_min, angular_x_max), (angular_y_min, angular_y_max)` tuple
            containing the minimum and maximum relative angular coordinates of the
            pixels in the image.
        """
        return self._get_img_limits(self.xy2angular)

    def get_img_limits_xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get the limits of the image coordinates. See :func:`get_img_limits_radec` for
        more details.

        Returns:
            `(x_min, x_max), (y_min, y_max)` tuple containing the minimum and maximum
            `image pixel coordinates`_ of the pixels in the image.
        """
        return self._get_img_limits(lambda x, y: (x, y))

    # Illumination functions etc.
    def limb_xy(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        `Image pixel coordinates`_ version of :func:`Body.limb_radec`.

        Args:
            **kwargs: Passed to :func:`Body.limb_radec`.

        Returns:
            `(x, y)` tuple of coordinate arrays.
        """
        return self._radec_arrs2xy_arrs(*self.limb_radec(**kwargs))

    def limb_xy_by_illumination(
        self, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        `Image pixel coordinates`_ version of :func:`Body.limb_radec_by_illumination`.

        Args:
            **kwargs: Passed to :func:`Body.limb_radec_by_illumination`.

        Returns:
            `(x_day, y_day, x_night, y_night)` tuple of coordinate arrays of the dayside
            then nightside parts of the limb.
        """
        ra_day, dec_day, ra_night, dec_night = self.limb_radec_by_illumination(**kwargs)
        return (
            *self._radec_arrs2xy_arrs(ra_day, dec_day),
            *self._radec_arrs2xy_arrs(ra_night, dec_night),
        )

    def terminator_xy(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        `Image pixel coordinates`_ version of :func:`Body.terminator_radec`.

        Args:
            **kwargs: Passed to :func:`Body.terminator_radec`.

        Returns:
            `(x, y)` tuple of coordinate arrays.
        """
        return self._radec_arrs2xy_arrs(*self.terminator_radec(**kwargs))

    def visible_lonlat_grid_xy(
        self, *args, **kwargs: Unpack[LonLatGridKwargs]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        `Image pixel coordinates`_ version of :func:`Body.visible_lonlat_grid_radec`.

        Args:
            *args: Passed to :func:`Body.visible_lonlat_grid_radec`.
            **kwargs: Passed to :func:`Body.visible_lonlat_grid_radec`.

        Returns:
            List of `(x, y)` coordinate array tuples.
        """
        return [
            self._radec_arrs2xy_arrs(*rd)
            for rd in self.visible_lonlat_grid_radec(*args, **kwargs)
        ]

    def ring_xy(self, radius: float, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        `Image pixel coordinates`_ version of :func:`Body.ring_radec`.

        Args:
            radius: Radius in km of the ring from the centre of the target body.
            **kwargs: Passed to :func:`Body.ring_radec`.

        Returns:
            `(x, y)` tuple of coordinate arrays.
        """
        return self._radec_arrs2xy_arrs(*self.ring_radec(radius, **kwargs))

    # Matplotlib transforms
    def _get_matplotlib_xy2angular_fixed_transform(
        self,
    ) -> matplotlib.transforms.Affine2D:
        if self._mpl_transform_xy2angular_fixed is None:
            self._mpl_transform_xy2angular_fixed = matplotlib.transforms.Affine2D(
                self._get_xy2angular_matrix()
            )
        return self._mpl_transform_xy2angular_fixed

    def _get_matplotlib_angular_fixed2xy_transform(
        self,
    ) -> matplotlib.transforms.Affine2D:
        if self._mpl_transform_angular_fixed2xy is None:
            self._mpl_transform_angular_fixed2xy = matplotlib.transforms.Affine2D(
                self._get_angular2xy_matrix()
            )
        return self._mpl_transform_angular_fixed2xy

    def _maybe_get_axis_transform(
        self, ax: Axes | None
    ) -> matplotlib.transforms.Transform:
        return (
            ax.transData
            if ax is not None
            else matplotlib.transforms.IdentityTransform()
        )

    def matplotlib_xy2radec_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        """
        Get matplotlib transform which converts between coordinate systems.

        Transformations to/from the `xy` coordinate system are mutable objects which can
        be dynamically updated using :func:`update_transform` when the `radec` to `xy`
        coordinate conversion changes. This can be useful for plotting data (e.g. an
        observed image) using image xy coordinates onto an axis using RA/Dec
        coordinates. ::

            # Plot an observed image on an RA/Dec axis with a wireframe of the target
            ax = body.plot_wireframe_radec()
            ax.autoscale_view()
            ax.autoscale(False) # Prevent imshow breaking autoscale
            ax.imshow(
                img,
                origin='lower',
                transform=body.matplotlib_xy2radec_transform(ax),
                )

        .. note::
            Matplotlib transformations involving `xy` coordinates are mutable, and will
            be automatically updated whenever the disc parameters are changed. For
            example, in the following code, the plotted image will use the `x0 = 10`
            value that is set after the transform is used: ::

                # ...
                ax.imshow(
                    img,
                    origin='lower',
                    transform=body.matplotlib_xy2radec_transform(ax),
                    )
                body.set_x0(10)
                plt.show()

            If you want to 'fix' the transform to a specific set of disc parameters, you
            can use the transform's `frozen()` method to create a new transform that
            will not be updated when the disc parameters change: ::

                # ...
                ax.imshow(
                    img,
                    origin='lower',
                    transform=body.matplotlib_xy2radec_transform(ax).frozen(),
                    )
                # ...

        See :func:`Body.matplotlib_radec2km_transform` for more details and notes on
        limitations of these linear transformations.
        """
        self.update_transform()
        return (
            self._get_matplotlib_xy2angular_fixed_transform()
            + self._get_matplotlib_transform(self.angular2radec, (0.0, 0.0), ax)
        )

    def matplotlib_radec2xy_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        self.update_transform()
        return (
            self._get_matplotlib_transform(
                self.radec2angular, (self.target_ra, self.target_dec), None
            )
            + self._get_matplotlib_angular_fixed2xy_transform()
            + self._maybe_get_axis_transform(ax)
        )

    def matplotlib_xy2km_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        self.update_transform()
        return (
            self._get_matplotlib_xy2angular_fixed_transform()
            + self._get_matplotlib_transform(self.angular2km, (0.0, 0.0), ax)
        )

    def matplotlib_km2xy_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        self.update_transform()
        return (
            self._get_matplotlib_transform(self.km2angular, (0.0, 0.0), None)
            + self._get_matplotlib_angular_fixed2xy_transform()
            + self._maybe_get_axis_transform(ax)
        )

    def matplotlib_xy2angular_transform(
        self, ax: Axes | None = None, **angular_kwargs: Unpack[AngularCoordinateKwargs]
    ) -> matplotlib.transforms.Transform:
        self.update_transform()
        # f transforms from angular (fixed) -> angular (with kwargs)
        f = lambda ax, ay: self._obsvec2angular(
            self._angular2obsvec_norm(ax, ay), **angular_kwargs
        )
        return (
            self._get_matplotlib_xy2angular_fixed_transform()
            + self._get_matplotlib_transform(f, (0.0, 0.0), ax)
        )

    def matplotlib_angular2xy_transform(
        self, ax: Axes | None = None, **angular_kwargs: Unpack[AngularCoordinateKwargs]
    ) -> matplotlib.transforms.Transform:
        self.update_transform()
        # f transforms from angular (with kwargs) -> angular (fixed)
        f = lambda ax, ay: self._obsvec2angular(
            self._angular2obsvec_norm(ax, ay), **angular_kwargs
        )
        return (
            self._get_matplotlib_transform(f, (0.0, 0.0), None)
            + self._get_matplotlib_angular_fixed2xy_transform()
            + self._maybe_get_axis_transform(ax)
        )

    def update_transform(self) -> None:
        """
        Update the matplotlib transformations involving `xy` coordinates (e.g.
        :func:`matplotlib_radec2xy_transform`) to use the latest disc parameter
        values `(x0, y0, r0, rotation)`.

        .. versionchanged:: 1.12.0
            The transformations are now updated automatically whenever the disc
            parameters are changed, so is generally no longer needed to be called
            manually.
        """
        self._get_matplotlib_xy2angular_fixed_transform().set_matrix(
            self._get_xy2angular_matrix()
        )
        self._get_matplotlib_angular_fixed2xy_transform().set_matrix(
            self._get_angular2xy_matrix()
        )

    # Mapping
    def map_img(
        self,
        img: np.ndarray,
        *,
        interpolation: (
            Literal['nearest', 'linear', 'quadratic', 'cubic'] | int | tuple[int, int]
        ) = 'linear',
        spline_smoothing: float = 0,
        propagate_nan: bool = True,
        warn_nan: bool = False,
        **map_kwargs: Unpack[MapKwargs],
    ) -> np.ndarray:
        """
        Project an observed image to a map. See :func:`generate_map_coordinates` for
        details about customising the projection used.

        If `interpolation` is `'linear'` (the default), `'quadratic'` or `'cubic'`, the
        map projection is performed using `scipy.interpolate.RectBivariateSpline` using
        the specified degree of spline interpolation. This spline interpolation does not
        accept NaN values in the input data, so any NaN pixels are automatically
        replaced by the average value of the surrounding pixels (3x3 footprint) before
        spline interpolation is performed, preventing or significantly reducing the
        magnitude of any artefacts on the edge of NaN regions.

        If `interpolation` is `'nearest'`, no interpolation is performed, and the mapped
        image takes the value of the nearest pixel in the image to that location. This
        can be useful to easily visualise the pixel scale for low spatial resolution
        observations.

        See also :func:`Observation.get_mapped_data`.

        Args:
            img: Observed image where pixel coordinates correspond to the `xy` pixel
                coordinates (e.g. those used in :func:`get_x0`). If `img` is a data cube
                (i.e. has 3 dimensions), then each image in the cube is mapped
                and a mapped cube is returned, equivalent to
                `np.array([body.map_img(img) for img in cube])`.
            interpolation: Interpolation method used when mapping. This can be any of
                `'nearest'`, `'linear'` (the default), `'quadratic'` or `'cubic'`.
                `'linear'`, `'quadratic'` and `'cubic'` are aliases for spline
                interpolations of degree 1, 2 and 3 respectively. Alternatively, the
                degree of spline interpolation can be specified manually by passing an
                integer or tuple of integers. If an integer is passed, the same
                interpolation is used in both the x and y directions (i.e.
                `RectBivariateSpline` with `kx = ky = interpolation`). If a tuple of
                integers is passed, the first integer is used for the x direction and
                the second integer is used for the y direction (i.e.
                `RectBivariateSpline` with `kx, ky = interpolation`).
            spline_smoothing: Smoothing factor passed to
                `RectBivariateSpline(..., s=spline_smoothing)` when spline interpolation
                is used. This can be useful to smooth over noisy data, though may hide
                subtle structure if a large smoothing value is used. This parameter is
                ignored if spline interpolation is not used.
            propagate_nan: By default (`propagate_nan=True`) when performing spline
                interpolation, areas of the map corresponding to NaN values in the image
                are set to NaN, along with areas outside the convex hull of the image's
                pixel centres. If `propagate_nan` is `False`, then areas corresponding
                to NaN pixels will be filled with interpolated/extrapolated values where
                possible, which can be useful to fill in regions of missing data.
                This parameter has no effect if `interpolation='nearest'`.
            warn_nan: Print warning if any values in `img` are NaN when any of the
                spline interpolations are used.
            **map_kwargs: Additional arguments are passed to
                :func:`generate_map_coordinates` to specify and customise the map
                projection.

        Returns:
            Array containing map of the values in `img` at each location on the surface
            of the target body. Locations which are not visible or outside the
            projection domain have a value of NaN.

        Raises:
            ValueError: if the input `img` shape is inconsistent with the body's image
                size.

        .. versionchanged:: 1.11.2
            Added more sophisticated replacement of NaN values in `img` before
            performing spline interpolation, preventing or significantly reducing any
            artefacts on the edge of NaN regions. NaN values are now replaced by the
            average value of the surrounding non-NaN pixels, whereas previously NaN
            values were always replaced with 0.
        """
        img = np.asarray(img)
        if img.ndim == 3:
            return np.array(
                [
                    self.map_img(
                        img_slice,
                        interpolation=interpolation,
                        spline_smoothing=spline_smoothing,
                        propagate_nan=propagate_nan,
                        warn_nan=warn_nan,
                        **map_kwargs,
                    )
                    for img_slice in img
                ]
            )
        if img.shape != (self._ny, self._nx):
            raise ValueError(
                f'The input `img` shape {img.shape!r} is inconsistent with '
                f'the body\'s image size (ny={self._ny}, nx={self._nx})'
            )

        x_map = self.get_x_map(**map_kwargs)
        y_map = self.get_y_map(**map_kwargs)
        projected = self._make_empty_map(**map_kwargs)

        spline_k = {'linear': 1, 'quadratic': 2, 'cubic': 3}
        if interpolation in spline_k:  # pylint: disable=consider-using-get
            interpolation = spline_k[interpolation]

        if interpolation == 'nearest':
            nan_sentinel = -999  # x_map and y_map are always >= 0
            x_map = np.asarray(
                np.nan_to_num(np.round(x_map), nan=nan_sentinel), dtype=int
            )
            y_map = np.asarray(
                np.nan_to_num(np.round(y_map), nan=nan_sentinel), dtype=int
            )
            for a, b in self._iterate_image(projected.shape):
                x = x_map[a, b]
                if x == nan_sentinel:
                    continue
                y = y_map[a, b]  # y should never be nan when x is not nan
                projected[a, b] = img[y, x]
        elif isinstance(interpolation, int | tuple):
            if isinstance(interpolation, int):
                kx = ky = interpolation
            else:
                kx, ky = interpolation
            nans = np.isnan(img)
            if not np.all(nans):
                img = self._replace_nans_with_interpolated_values(img, warn_nan)
                interpolator = scipy.interpolate.RectBivariateSpline(
                    np.arange(img.shape[0]),
                    np.arange(img.shape[1]),
                    img,
                    kx=kx,
                    ky=ky,
                    s=spline_smoothing,  # type: ignore (docs say s is a float)
                )

                # Collect any coordinates to interpolate in these lists, then perform
                # the interpolation at the end with a single call to interpolator.ev.
                # This is directly equivalent to doing the interpolation inside the for
                # loop with `projected[a, b] = interpolator(y, x).item()`, but can be
                # much faster for large images.
                a_vals: list[int] = []
                b_vals: list[int] = []
                x_vals: list[float] = []
                y_vals: list[float] = []
                for a, b in self._iterate_image(projected.shape):
                    x = x_map[a, b]
                    if math.isnan(x):
                        continue
                    y = y_map[a, b]  # y should never be nan when x is not nan
                    if propagate_nan and self._should_propagate_nan_to_map(x, y, nans):
                        continue
                    a_vals.append(a)
                    b_vals.append(b)
                    x_vals.append(x)
                    y_vals.append(y)
                projected[a_vals, b_vals] = interpolator.ev(y_vals, x_vals)
        else:
            raise ValueError(f'Unknown interpolation method {interpolation!r}')
        return projected

    def _should_propagate_nan_to_map(
        self, x: float, y: float, nans: np.ndarray
    ) -> bool:
        # Test if any of the four surrounding integer pixels in the image are NaN or if
        # outside the convex hull of pixel centres
        if x < 0.0 or y < 0.0 or x > self._nx - 1 or y > self._ny - 1:
            return True
        x0 = max(math.floor(x), 0)
        x1 = min(math.ceil(x), self._nx - 1)
        y0 = max(math.floor(y), 0)
        y1 = min(math.ceil(y), self._ny - 1)
        return nans[y0, x0] or nans[y0, x1] or nans[y1, x0] or nans[y1, x1]

    def _xy_in_image_frame(self, x: float, y: float) -> bool:
        return (-0.5 < x < self._nx - 0.5) and (-0.5 < y < self._ny - 0.5)

    def _replace_nans_with_interpolated_values(
        self, img: np.ndarray, warn_nan: bool
    ) -> np.ndarray:
        """
        Return a copy of the input image where NaNs are replaced with the mean of
        surrounding non-NaN pixels (3x3 footprint). All other NaNs are replaced with the
        median of the original data.

        This is mainly useful for preparing an input image before passing it through
        e.g. RectBivariateSpline (which doesn't accept NaNs, and may produce artefacts
        if we just use nan_to_num).
        """
        bad = ~np.isfinite(img)
        if warn_nan and np.any(bad):
            print('Warning, image contains NaN values which will be corrected')
        cleaned = img.astype(float, copy=True)
        if np.any(np.isinf(img)):
            # Treat inf as nan in averaging calculations
            img = np.nan_to_num(
                img, nan=np.nan, posinf=np.nan, neginf=np.nan, copy=True
            )
        if np.all(bad):
            median = 0.0
        else:
            median = np.nanmedian(img)
        cleaned[bad] = median
        # Fix bad pixels that have neighbouring good pixels by replacing them with the
        # mean of the surrounding 3x3 good pixels.
        to_fix = bad & ~scipy.ndimage.uniform_filter(bad, size=3)  #  type: ignore
        for i, j in np.argwhere(to_fix):
            cleaned[i, j] = np.nanmean(
                img[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
            )
        return cleaned

    # Plotting
    def plot_wireframe_xy(
        self,
        ax: Axes | None = None,
        *,
        scale_factor: float | None = None,
        add_axis_labels: bool | None = None,
        aspect_adjustable: Literal['box', 'datalim'] | None = 'box',
        show: bool = False,
        **wireframe_kwargs: Unpack[WireframeKwargs],
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation using image pixel
        coordinates. See :func:`Body.plot_wireframe_radec` for details of accepted
        arguments.

        Returns:
            The axis containing the plotted wireframe.
        """
        if add_axis_labels is None:
            add_axis_labels = scale_factor is None

        # Use combo of corodinate_func and matplotlib transform so that the plot can be
        # updated with new disc parameters without having to replot the entire thing
        ax = self._plot_wireframe(
            coordinate_func=self.radec2angular,
            scale_factor=scale_factor,
            transform=self._get_matplotlib_angular_fixed2xy_transform(),
            aspect_adjustable=aspect_adjustable,
            ax=ax,
            **wireframe_kwargs,
        )

        if self._test_if_img_size_valid() and scale_factor is None:
            ax.set_xlim(-0.5, self._nx - 0.5)
            ax.set_ylim(-0.5, self._ny - 0.5)
        if add_axis_labels:
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

        if show:
            plt.show()
        return ax

    @_adjust_surface_altitude_decorator
    def plot_map_wireframe(
        self,
        ax: Axes | None = None,
        *,
        label_poles: bool = True,
        add_title: bool = True,
        add_axis_labels: bool = True,
        grid_interval: float = 30,
        grid_lat_limit: float = 90,
        indicate_equator: bool = True,
        indicate_prime_meridian: bool = True,
        aspect_adjustable: Literal['box', 'datalim'] | None = 'box',
        formatting: dict[WireframeComponent, dict[str, Any]] | None = None,
        **map_and_formatting_kwargs,
    ) -> Axes:
        """
        Plot wireframe (e.g. gridlines) of the map projection of the observation. See
        :func:`Body.plot_wireframe_radec` for details of accepted arguments.

        For example, to plot an orthographic map's wireframe with a red boundary and
        dashed gridlines, you can use: ::

            body.plot_map_wireframe(
                projection='orthographic',
                lat=45,
                formatting={
                    'grid': {'linestyle': '--'},
                    'map_boundary': {'color': 'red'},
                }
            )
        """
        if ax is None:
            ax = cast(Axes, plt.gca())

        map_kwargs, common_formatting = _extract_map_kwargs_from_dict(
            map_and_formatting_kwargs
        )
        if 'common_formatting' in common_formatting:
            common_formatting |= common_formatting.pop('common_formatting')

        kwargs = self._get_wireframe_kw(
            common_formatting=common_formatting, formatting=formatting
        )
        _, _, _, _, transformer, map_kw_used = self.generate_map_coordinates(
            **map_kwargs
        )
        projection = map_kw_used['projection']

        if aspect_adjustable is not None:
            ax.set_aspect(1, adjustable=aspect_adjustable)

        lon_ticks = np.arange(0, 360.0001, grid_interval)
        lat_ticks = np.arange(-90, 90.0001, grid_interval)

        if projection in {'azimuthal', 'azimuthal equal area'}:
            # Run separately for either side of equator to reduce issues for azimuthal
            # where the grid lines overplot each other. We still can get issues for e.g.
            # lat=45, but this fixes the most common cases of lat=0,90,-90 and it's a
            # relatively minor cosmetic bug so is probably more-or-less fine as-is.
            npts = 360
            lats_to_plot = [
                np.linspace(-grid_lat_limit, 0, npts),
                np.linspace(0, grid_lat_limit, npts),
            ]
        else:
            npts = 720
            lats_to_plot = [np.linspace(-grid_lat_limit, grid_lat_limit, npts)]
        for lon in lon_ticks:
            if lon == 360 or (lon == 0 and projection == 'rectangular'):
                continue
            for lats in lats_to_plot:
                x, y = transformer.transform(lon * np.ones(npts), lats)
                ax.plot(
                    x,
                    y,
                    **kwargs['grid']
                    | (
                        kwargs['prime_meridian']
                        if lon == 0 and indicate_prime_meridian
                        else {}
                    ),
                )
        npts = 720
        for lat in lat_ticks:
            if float(lat) in {-90.0, 90.0}:
                continue
            if abs(lat) > grid_lat_limit:
                continue
            x, y = transformer.transform(np.linspace(0, 360, npts), lat * np.ones(npts))
            ax.plot(
                x,
                y,
                **kwargs['grid']
                | (kwargs['equator'] if lat == 0 and indicate_equator else {}),
            )

        boundary: tuple[np.ndarray, np.ndarray] | None = None
        # Formulae for boundaries based on Cartopy CRS boundaries
        if projection == 'orthographic':
            # Elipse boundary - https://math.stackexchange.com/questions/91132
            x0 = 1
            b = self.r_polar / self.r_eq
            theta = np.radians(map_kw_used['lat'])
            y0 = np.sqrt((np.sin(theta)) ** 2 + b**2 * (np.cos(theta)) ** 2)
            t = np.linspace(0, -2 * np.pi, 100)
            boundary = (x0 * np.cos(t), y0 * np.sin(t))
        elif projection in {'azimuthal', 'azimuthal equal area'}:
            # Circular boundary
            x0 = y0 = 1
            t = np.linspace(0, -2 * np.pi, 100)
            boundary = (x0 * np.cos(t), y0 * np.sin(t))

        if boundary:
            ax.plot(*boundary, **kwargs['map_boundary'])

        if label_poles and projection != 'rectangular':
            for lat, s in ((90, 'N'), (-90, 'S')):
                x, y = transformer.transform(0, lat)
                if math.isfinite(x) and math.isfinite(y):
                    ax.text(x, y, s, **kwargs['pole'])

        if add_axis_labels:
            if projection == 'rectangular':
                if self.positive_longitude_direction == 'W':
                    ax.set_xlim(360, 0)
                else:
                    ax.set_xlim(0, 360)
                ax.set_ylim(-90, 90)
                ax.set_xlabel(
                    f'Planetographic longitude ({self.positive_longitude_direction})'
                )
                ax.set_ylabel('Planetographic latitude')

                ax.set_xticks(lon_ticks)
                ax.set_xticklabels(
                    [f'{x:.0f}°' if x % 90 == 0 else '' for x in lon_ticks]
                )

                ax.set_yticks(lat_ticks)
                ax.set_yticklabels(
                    [f'{y:.0f}°' if y % 90 == 0 else '' for y in lat_ticks]
                )
            elif projection in {'orthographic', 'azimuthal', 'azimuthal equal area'}:
                ax.set_xticks([])
                ax.set_yticks([])

        if add_title:
            ax.set_title(self.get_description(multiline=True))
        return ax

    def plot_map(
        self,
        map_img: np.ndarray,
        ax: Axes | None = None,
        *,
        wireframe_kwargs: dict[str, Any] | None = None,
        add_wireframe: bool = True,
        **kwargs,
    ) -> QuadMesh:
        """
        Utility function to easily plot a mapped image using `plt.pcolormesh` with
        appropriate extents, axis labels, gridlines etc.

        Args:
            map_img: Image to plot.
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), then
                a new figure and axis is created.
            wireframe_kwargs: Dictionary of arguments passed to
                :func:`plot_map_wireframe`.
            add_wireframe: Enable/disable plotting of wireframe.
            **kwargs: Additional arguments are passed to
                :func:`generate_map_coordinates` to specify the map projection used, and
                to Matplotlib's `pcolormesh` to customise the plot. For example, can be
                used to set the colormap of the plot using e.g.
                `body.plot_map(..., projection='orthographic', cmap='Greys')`.

        Returns:
            Handle returned by Matplotlib's `pcolormesh`.
        """
        if ax is None:
            fig, ax = plt.subplots()

        map_kwargs, kwargs = _extract_map_kwargs_from_dict(kwargs)

        _, _, xx, yy, _, _ = self.generate_map_coordinates(**map_kwargs)
        h = ax.pcolormesh(xx, yy, map_img, **kwargs)
        if add_wireframe:
            self.plot_map_wireframe(ax=ax, **(wireframe_kwargs or {}), **map_kwargs)
        return h

    def imshow_map(self, *args, **kwargs):
        """
        Alias for `plot_map` for backwards compatibility.

        :meta private:
        """
        # backwards compatibility
        return self.plot_map(*args, **kwargs)

    # Wireframe generation
    def _get_wireframe_overlay(
        self,
        *,
        output_size: int | None,
        dpi: int,
        nx: int,
        ny: int,
        rgba: bool,
        plot_fn: Callable[[Axes], Any],
    ) -> np.ndarray:
        output_size = output_size or max(nx, ny)
        s = output_size / dpi
        if nx > ny:
            figsize = (s, s * ny / nx)
        else:
            figsize = (s * nx / ny, s)

        # Use Figure rather than plt.figure to avoid segmentation fault when running
        # from tkinter GUI (issue #258)
        fig = Figure(figsize=figsize, dpi=dpi, facecolor='w')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='w')
        plot_fn(ax)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        with io.BytesIO() as io_buf:
            fig.savefig(io_buf, format='raw', dpi=dpi, transparent=rgba)
            io_buf.seek(0)
            img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        plt.close(fig)
        img = img_arr.reshape((fig.canvas.get_width_height()[::-1]) + (4,))
        if not rgba:
            img = np.asarray(np.mean(img[:, :, :3], axis=-1), dtype=np.uint8)
        img = np.flipud(img)  # Make consistent with FITS orientation
        return img

    def get_wireframe_overlay_img(
        self,
        output_size: int | None = 1500,
        dpi: int = 200,
        rgba: bool = False,
        **plot_kwargs,
    ) -> np.ndarray:
        """
        .. warning ::

            This is a beta feature and the API may change in future.

        Generate a wireframe image of the target.

        This effectively generates an image version of :func:`plot_wireframe_xy` which
        can then be used as an overlay on top of the observation when creating figures
        in other applications.

        See also :func:`get_wireframe_overlay_map`.

        .. note ::

            The returned image data follows the FITS orientation convention (with the
            origin at the bottom left) so may need to be flipped vertically in some
            applications. If needed, the image can be flipped in Python using: ::

                np.flipud(body.get_wireframe_overlay_img())

        .. hint ::

            If you are creating plots with Matplotlib, it is generally better to use
            :func:`plot_wireframe_xy` directly rather than generating an image as it
            will produce a higher quality plot.

        Args:
            output_size: Size of the output image in pixels. This will be the length of
                the longest side of the image. The other side will be scaled accordingly
                to maintain the aspect ratio of the observed data. If `size` is `None`,
                then the size is set to match the size of the observed data.
            dpi: Dots per inch of the output image. This can be used to control the size
                of plotted elements in the output image - larger `dpi` values will
                produce larger plotted elements.
            rgba: By default, the returned image only has a single greyscale channel. If
                `rgba` is `True`, then the returned image has 4 channels (red, green,
                blue, alpha) which can be used to more easily overlay the wireframe on
                top of the observed data in other applications.
            **plot_kwargs: Additional arguments passed to :func:`plot_wireframe_xy`.
        Returns:
            Image of the wireframe which has the same aspect ratio as the observed data.
        """
        # TODO remove beta note when stable
        return self._get_wireframe_overlay(
            output_size=output_size,
            dpi=dpi,
            nx=self._nx,
            ny=self._ny,
            rgba=rgba,
            plot_fn=lambda ax: self.plot_wireframe_xy(
                ax=ax,
                add_axis_labels=False,
                add_title=False,
                **(dict(color='k') | plot_kwargs or {}),  # type: ignore
            ),
        )

    def get_wireframe_overlay_map(
        self,
        output_size: int | None = 1500,
        dpi: int = 200,
        rgba: bool = False,
        **map_and_formatting_kwargs,
    ) -> np.ndarray:
        """
        .. warning ::

            This is a beta feature and the API may change in future.

        Generate a wireframe map of the target.

        This effectively generates an image version of :func:`plot_map_wireframe` which
        can then be used as an overlay on top of the mapped observation when creating
        figures in other applications.

        See also :func:`get_wireframe_overlay_img`.

        .. note ::

            The returned image data follows the FITS orientation convention (with the
            origin at the bottom left) so may need to be flipped vertically in some
            applications. If needed, the image can be flipped in Python using: ::

                np.flipud(body.get_wireframe_overlay_map())

        .. hint ::

            If you are creating plots with Matplotlib, it is generally better to use
            :func:`plot_map_wireframe` directly rather than generating an image as it
            will produce a higher quality plot.

        Args:
            output_size: Size of the output image in pixels. This will be the length of
                the longest side of the map. The other side will be scaled accordingly
                to maintain the aspect ratio of the observed data. If `size` is `None`,
                then the size is set to match the pixel size of the map.
            dpi: Dots per inch of the output image. This can be used to control the size
                of plotted elements in the output image - larger `dpi` values will
                produce larger plotted elements.
            rgba: By default, the returned image only has a single greyscale channel. If
                `rgba` is `True`, then the returned image has 4 channels (red, green,
                blue, alpha) which can be used to more easily overlay the wireframe on
                top of the observed data in other applications.
            plot_kwargs: Dictionary of arguments passed to :func:`plot_map_wireframe`.
            **map_and_formatting_kwargs: Passed to :func:`plot_map_wireframe`. This
                can include arguments such as `projection`.
        Returns:
            Image of the map wireframe which has the same aspect ratio as the map.
        """
        # TODO remove beta note when stable
        map_kwargs, plot_kwargs = _extract_map_kwargs_from_dict(
            map_and_formatting_kwargs
        )

        lons, lats, xx, yy, transformer, map_kw_used = self.generate_map_coordinates(
            **map_kwargs
        )
        nx = xx.shape[1]
        ny = yy.shape[0]

        def plot_fn(ax: Axes):
            self.plot_map_wireframe(
                ax=ax,
                add_axis_labels=False,
                add_title=False,
                **(dict(color='k') | plot_kwargs),  # type: ignore
                **map_kwargs,
            )
            # Add dx/dy to the limits to ensure the wireframe covers all of each pixel
            # as the xx and yy coordinates only give the centre of each pixel
            dx = abs(xx[0][1] - xx[0][0]) / 2
            ax.set_xlim(np.nanmin(xx) - dx, np.nanmax(xx) + dx)
            dy = abs(yy[1][0] - yy[0][0]) / 2
            ax.set_ylim(np.nanmin(yy) - dy, np.nanmax(yy) + dy)

        return self._get_wireframe_overlay(
            output_size=output_size, dpi=dpi, nx=nx, ny=ny, rgba=rgba, plot_fn=plot_fn
        )

    # Backplane management
    @staticmethod
    def standardise_backplane_name(name: str) -> str:
        """
        Create a standardised version of a backplane name when finding and registering
        backplanes.

        This standardisation is used in functions like :func:`get_backplane_img` and
        :func:`plot_backplane` so that, for example `body.plot_backplane('DEC')`,
        `body.plot_backplane('Dec')` and `body.plot_backplane('dec')` all produce the
        same plot.

        Args:
            name: Input backplane name.

        Returns:
            Standardised name with leading/trailing spaces removed and converted to
            upper case.
        """
        return name.strip().upper()

    def register_backplane(
        self,
        name: str,
        description: str,
        get_img: Callable[[], np.ndarray],
        get_map: _BackplaneMapGetter,
    ) -> None:
        """
        Create a new :class:`Backplane` and register it to :attr:`backplanes`.

        See :class:`Backplane` for more detail about parameters.

        Args:
            name: Name of backplane. This is standardised using
                :func:`standardise_backplane_name` before being registered.
            description: Longer description of backplane, including units.
            get_img: Function to generate backplane image.
            get_map: Function to generate backplane map.

        Raises:
            ValueError: if provided backplane name is already registered.
        """
        name = self.standardise_backplane_name(name)
        if name in self.backplanes:
            raise ValueError(f'Backplane named {name!r} is already registered')
        self.backplanes[name] = Backplane(
            name=name, description=description, get_img=get_img, get_map=get_map
        )

    def backplane_summary_string(self) -> str:
        """
        Returns:
            String summarising currently registered :attr:`backplanes`.
        """
        return '\n'.join(
            f'{bp.name}: {bp.description}' for bp in self.backplanes.values()
        )

    def print_backplanes(self) -> None:
        """
        Prints output of :func:`backplane_summary_string`.
        """
        print(self.backplane_summary_string())

    def get_backplane(self, name: str) -> Backplane:
        """
        Convenience function to retrieve a backplane registered to :attr:`backplanes`.

        This method is equivalent to ::

            body.backplanes[self.standardise_backplane_name(name)]

        Args:
            name: Name of the desired backplane. This is standardised with
                :func:`standardise_backplane_name` and used to choose a registered
                backplane from :attr:`backplanes`.

        Returns:
            Backplane registered with `name`.

        Raises:
            BackplaneNotFoundError: if `name` is not registered as a backplane.
        """
        name = self.standardise_backplane_name(name)
        try:
            return self.backplanes[name]
        except KeyError as exc:
            raise BackplaneNotFoundError(
                '{n!r} not found. Currently registered backplanes are: {r}.'.format(
                    n=name,
                    r=', '.join([repr(n) for n in self.backplanes.keys()]),
                )
            ) from exc

    def get_backplane_img(self, name: str, *, alt: float = 0.0) -> np.ndarray:
        """
        Generate backplane image.

        Note that a generated backplane image will depend on the disc parameters
        `(x0, y0, r0, rotation)` at the time this function is called. Generating the
        same backplane when there are different disc parameter values will produce a
        different image. This method creates a copy of the generated image, so the
        returned image can be safely modified without affecting the cached value (unlike
        the return values from functions such as :func:`get_lon_img`).

        When `alt=0`, this method is equivalent to ::

            body.get_backplane(name).get_img().copy()

        To create an oversampled backplane image, you can use the method
        :func:`scale_img_size`. For example, ::

            body = planetmapper.BodyXY('jupiter', sz=10)

            ax = body.plot_backplane_img('EMISSION')
            ax.set_title('Original unscaled backplane')
            plt.show()

            body_scaled = body.copy()
            body_scaled.scale_img_size(10)
            ax = body_scaled.plot_backplane_img('EMISSION')
            ax.set_title('Scaled higher resolution backplane')
            plt.show()

        See also :func:`get_backplane_map` and :func:`plot_backplane_img`.

        Args:
            name: Name of the desired backplane. This is standardised with
                :func:`standardise_backplane_name` and used to choose a registered
                backplane from :attr:`backplanes`.
            alt: Altitude adjustment to the body's surface in km.

        Returns:
            Array containing the backplane's values for each pixel in the image.
        """
        with _AdjustedSurfaceAltitude(self, alt):
            return (
                self.backplanes[self.standardise_backplane_name(name)].get_img().copy()
            )

    def get_backplane_map(
        self, name: str, **map_kwargs: Unpack[MapKwargs]
    ) -> np.ndarray:
        """
        Generate map of backplane values.

        This method creates a copy of the generated image, so the returned map can be
        safely modified without affecting the cached value (unlike the return values
        from functions such as :func:`get_lon_map`).

        This method is equivalent to ::

            body.get_backplane(name).get_map(**map_kwargs).copy()

        See also :func:`get_backplane_img` and :func:`plot_backplane_map`.

        Args:
            name: Name of the desired backplane. This is standardised with
                :func:`standardise_backplane_name` and used to choose a registered
                backplane from :attr:`backplanes`.
            **map_kwargs: Additional arguments are passed to
                :func:`generate_map_coordinates` to specify and customise the map
                projection.

        Returns:
            Array containing map of the backplane's values over the surface of the
            target body.
        """
        return (
            self.backplanes[self.standardise_backplane_name(name)]
            .get_map(**map_kwargs)
            .copy()
        )

    def plot_backplane_img(
        self,
        name: str,
        ax: Axes | None = None,
        *,
        alt: float = 0.0,
        show: bool = False,
        **kwargs,
    ) -> Axes:
        """
        Plot a backplane image with the wireframe outline of the target.

        Note that a generated backplane image will depend on the disc parameters
        `(x0, y0, r0, rotation)` at the time this function is called. Generating the
        same backplane when there are different disc parameter values will produce a
        different image.

        See also :func:`plot_backplane_map` and :func:`get_backplane_img`.

        Args:
            name: Name of the desired backplane.
            ax: Passed to :func:`plot_wireframe_xy`.
            alt: Altitude adjustment to the body's surface in km.
            show: Passed to :func:`plot_wireframe_xy`.
            **kwargs: Passed to Matplotlib's `imshow` when plotting the backplane image.
                For example, can be used to set the colormap of the plot using
                `body.plot_backplane_img(..., cmap='Greys')`.

        Returns:
            The axis containing the plotted data.
        """
        with _AdjustedSurfaceAltitude(self, alt):
            backplane = self.get_backplane(name)
            ax = self.plot_wireframe_xy(ax, show=False)
            im = ax.imshow(backplane.get_img(), origin='lower', **kwargs)
            plt.colorbar(im, label=backplane.description)
            if show:
                plt.show()
            return ax

    def plot_backplane_map(
        self, name: str, ax: Axes | None = None, show: bool = False, **kwargs
    ) -> Axes:
        """
        Plot a map of backplane values on the target body.

        For example, top plot a backplane map with the 'Blues' colourmap and a red
        partially transparent wireframe, on an orthographic projection, use: ::

            body.plot_backplane_map(
                'EMISSION',
                projection='orthographic',
                cmap='Blues',
                wireframe_kwargs=dict(color='r', alpha=0.5),
            )

        See also :func:`plot_backplane_img` and :func:`get_backplane_img`.

        Args:
            name: Name of the desired backplane.
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), then
                a new figure and axis is created.
            show: Toggle showing the plotted figure with `plt.show()`
            **kwargs: Additional arguments are passed to :func:`plot_map`. These can be
                used to specify and customise the map projection, and to customise the
                plot formatting.
        Returns:
            The axis containing the plotted data.
        """
        if ax is None:
            fig, ax = plt.subplots()
        backplane = self.get_backplane(name)

        map_kwargs, other_kwargs = _extract_map_kwargs_from_dict(kwargs)
        if 'plot_kwargs' in other_kwargs:
            # backwards compatibility
            other_kwargs |= other_kwargs.pop('plot_kwargs')

        im = self.plot_map(
            backplane.get_map(**map_kwargs), ax=ax, **map_kwargs, **other_kwargs
        )
        plt.colorbar(im, label=backplane.description)
        if show:
            plt.show()
        return ax

    # Mapping projection internals
    @_cache_stable_result
    @_adjust_surface_altitude_decorator
    def generate_map_coordinates(
        self,
        projection: str = 'rectangular',
        *,
        degree_interval: float = 1,
        lon: float = 0,
        lat: float = 0,
        size: int = 100,
        lon_coords: np.ndarray | tuple | None = None,
        lat_coords: np.ndarray | tuple | None = None,
        projection_x_coords: np.ndarray | tuple | None = None,
        projection_y_coords: np.ndarray | tuple | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        alt: float = 0.0,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pyproj.Transformer,
        dict[str, Any],
    ]:
        """
        Generate underlying coordinates and transformation for a given map projection.

        The built-in map projections (i.e. possible values for the `projection`
        argument) are:

        - `'rectangular'`: cylindrical equirectangular projection onto a regular
          longitude and latitude grid. The resolution of the map can be controlled with
          the `degree_interval` argument which sets the spacing in degrees between grid
          points. This is the default map projection.
        - `'orthographic'`: orthographic projection where the central longitude and
          latitude can be customized with the `lon` and `lat` arguments. The size of the
          map can be controlled with the `size` argument.
        - `'azimuthal'`: azimuthal equidistant projection where the central longitude
          and latitude can be customized with the `lon` and `lat` arguments. The size of
          the map can be controlled with the `size` argument.
        - `'azimuthal equal area'`: Lambert azimuthal equal area projection where the
          central longitude and latitude can be customized with the `lon` and `lat`
          arguments. The size of the map can be controlled with the `size` argument.
        - `'manual'`: manually specify the longitude and latitude coordinates to use
          for each point on the map with the `lon_coords` and `lat_coords` arguments.

        Projections can also be specified by passing a proj projection string to the
        `projection` argument. See https://proj.org/operations/projections for a list of
        projections that can be used, and more details on their parameters. The provided
        projection string will be passed to `pyproj.CRS`. If you are manually specifying
        a projection, you must also at least provide `projection_x_coords` to provide
        the coordinates to project the data to. A :class:`ProjStringError` will be
        raised if an incorrect `axis` parameter is used - this parameter should be
        `+axis=wnu` for bodies with positive west coordinates, and `+axis=enu` for
        bodies with positive east coordinates (see
        :attr:`Body.positive_longitude_direction`). If the wrong `axis` parameter is
        used, then the output map will usually be incorrect (e.g. flipped horizontally).
        :func:`create_proj_string` can be used to help build a projection string, and
        will automatically ensure that the correct axis parameters are set.

        .. hint ::

            You generally don't need to call this method directly. Instead, pass your
            desired arguments directly to functions like :func:`get_backplane_map` or
            :func:`map_img`.

        Usage examples: ::

            # Generate default rectangular map for emission backplane
            body.get_backplane_map('EMISSION')

            # Generate default rectangular map at lower resolution and only covering
            # the northern hemisphere
            body.get_backplane_map('EMISSION', degree_interval=10, ylim=(0, np.inf))

            # Generate orthographic map of northern hemisphere
            body.get_backplane_map('EMISSION', projection='orthographic', lat=90)

            # Plot orthographic map of southern hemisphere with higher resolution
            body.plot_backplane_map(
                'EMISSION', projection='orthographic', lat=-90, size=500
                )

            # Get azimuthal equidistant map projection of image, centred on specific
            # coordinate
            body.map_img(img, projection='azimuthal', lon=45, lat=30)

        Args:
            projection: String describing map projection to use (see list of supported
                projections above).
            degree_interval: Degree interval for `'rectangular` projection.
            lon: Central longitude of `'orthographic'`, `'azimuthal'` and `'azimuthal
                equal area'` projections.
            lat: Central latitude of `'orthographic'`, `'azimuthal'` and `'azimuthal
                equal area'` projections.
            size: Pixel size (width and height) of generated `'orthographic'`,
                `'azimuthal'` and `'azimuthal equal area'` projections.
            lon_coords: Longitude coordinate array to use for `'manual'` projection.
            lat_coords: Latitude coordinate array to use for `'manual'` projection.
            projection_x_coords: Projected x coordinate array to use with a pyproj
                projection string.
            projection_y_coords: Projected x coordinate array to use with a pyproj
                projection string.
            xlim: Tuple of `(x_min, x_max)` limits in the projected x coordinates of
                the map. If `None`, the default, then the no limits are applied (i.e.
                the entire globe will be mapped). If `xlim` is provided, it should be a
                tuple of two floats specifying the minimum and maximum x coordinates to
                project the map to. For example, to only plot the western hemisphere,
                you can use use `xlim=(0, 180)` in a rectangular projection. Note that
                these limits are expressed in the projected coordinates of the map.
                Setting the limits can be useful to speed up the performance of mapping
                when only a subset of the map is needed (such as for observations with
                limited spatial extent). If you only want to set one limit, then you can
                pass infinity e.g. `xlim=(315, np.inf)` to only set the minimum limit.
                The limits are implemented using
                `x_to_keep = (x >= min(xlim)) & (x <= max(xlim))`, so the ordering of
                the limits does not matter. Note that the limit calculations assume that
                the data is on a rectangular grid (i.e. all rows have the same x
                coordinates and all columns have the same y coordinates), so may produce
                unexpected results if a custom projection is used.
            ylim: Tuple of `(y_min, y_max)` limits in the projected y coordinates of
                the map. If `None`, the default, then the no limits are applied. See
                `xlim` for more details.
            alt: Altitude adjustment to the body's surface in km.

        Returns:
            `(lons, lats, xx, yy, transformer, info)` tuple where `lons` and `lats` are
            the longitude and latitude coordinates of the map, `xx` and `yy` are the
            projected coordinates of the map, `transformer` is a `pyproj.Transformer`
            object that can be used to transform between the two coordinate systems, and
            `info` is a dictionary containing the arguments used to build the map (e.g.
            for the default case this would be `{'projection': 'rectangular',
            'degree_interval': 1, 'xlim': None, 'ylim': None}`). The `lons`, `lats`,
            `xx` and `yy` arrays are :ref:`read-only<readonly arrays>`.

        Raises:
            ProjStringError: If a custom proj projection string is used that has an
                inconsistent axis parameter.

        .. versionchanged:: 1.12.1
            The axis parameter in custom proj projection strings is now checked for
            consistency with the body's positive longitude direction, and a
            :class:`ProjStringError` is raised if it is inconsistent.
        """
        info: dict[str, Any]  # Explicitly declare type of info to make pyright happy
        if projection == 'rectangular':
            lons = np.arange(degree_interval / 2, 360, degree_interval)
            if self.positive_longitude_direction == 'W':
                lons = lons[::-1]
            lats = np.arange(-90 + degree_interval / 2, 90, degree_interval)
            lons, lats = np.meshgrid(lons, lats)
            xx, yy = lons, lats
            transformer = self._get_pyproj_transformer()
            info = dict(projection=projection, degree_interval=degree_interval)
        elif projection == 'manual':
            lons = lon_coords
            lats = lat_coords
            if lons is None or lats is None:
                raise ValueError(
                    'lon_coords and lat_coords must be provided for manual projection'
                )
            lons = np.asarray(lons)
            lats = np.asarray(lats)
            if lons.ndim != lats.ndim:
                raise ValueError(
                    'lon_coords and lat_coords must have the same number of dimensions'
                )
            if lons.ndim == 1:
                lons, lats = np.meshgrid(lons, lats)
            if lons.ndim != 2:
                raise ValueError('lon_coords and lat_coords must be 1D or 2D arrays')
            if lons.shape != lats.shape:
                raise ValueError('lon_coords and lat_coords must have the same shape')
            xx, yy = lons, lats
            transformer = self._get_pyproj_transformer()
            info = dict(projection=projection)
        elif projection == 'orthographic':
            b = self.r_polar / self.r_eq
            proj = self.create_proj_string(
                'ortho',
                to_meter=self.r_eq,
                lon_0=lon,
                lat_0=lat,
                y_0=self.r_eq * (b - 1) * np.sin(np.radians(lat * 2)),
            )
            lim = max(1, b) * 1.01
            lons, lats, xx, yy, transformer = self._get_pyproj_map_coords(
                proj, np.linspace(-lim, lim, size)
            )
            info = dict(projection=projection, lon=lon, lat=lat, size=size)
        elif projection == 'azimuthal':
            proj = self.create_proj_string(
                'aeqd',
                to_meter=self.r_eq * np.pi,
                b=None,
                lon_0=lon,
                lat_0=lat,
            )
            lim = 1.01
            lons, lats, xx, yy, transformer = self._get_pyproj_map_coords(
                proj, np.linspace(-lim, lim, size)
            )
            info = dict(projection=projection, lon=lon, lat=lat, size=size)
        elif projection == 'azimuthal equal area':
            proj = self.create_proj_string(
                'laea',
                to_meter=self.r_eq * 2,
                b=None,
                lon_0=lon,
                lat_0=lat,
            )
            lim = 1.01
            lons, lats, xx, yy, transformer = self._get_pyproj_map_coords(
                proj, np.linspace(-lim, lim, size)
            )
            info = dict(projection=projection, lon=lon, lat=lat, size=size)
        else:
            if projection_x_coords is None:
                raise ValueError('x coords must be provided')
            lons, lats, xx, yy, transformer = self._get_pyproj_map_coords(
                projection, projection_x_coords, projection_y_coords
            )
            info = dict(
                projection=projection,
                projection_x_coords=projection_x_coords,
                projection_y_coords=projection_y_coords,
            )

        info['xlim'] = xlim
        info['ylim'] = ylim
        if xlim is not None:
            x_arr = xx[0]
            x_to_keep = (x_arr >= min(xlim)) & (x_arr <= max(xlim))
            xx = xx[:, x_to_keep]
            yy = yy[:, x_to_keep]
            lons = lons[:, x_to_keep]
            lats = lats[:, x_to_keep]
        if ylim is not None:
            y_arr = yy[:, 0]
            y_to_keep = (y_arr >= min(ylim)) & (y_arr <= max(ylim))
            xx = xx[y_to_keep, :]
            yy = yy[y_to_keep, :]
            lons = lons[y_to_keep, :]
            lats = lats[y_to_keep, :]

        # Standardise invalid lon/lat points (e.g. inf -> nan)
        lons[~np.isfinite(lons)] = np.nan
        lats[~np.isfinite(lats)] = np.nan

        if alt != 0.0:
            info['alt'] = alt
        return (
            _as_readonly_view(lons),
            _as_readonly_view(lats),
            _as_readonly_view(xx),
            _as_readonly_view(yy),
            transformer,
            info,
        )

    def create_proj_string(
        self,
        proj: str,
        **parameters,
    ) -> str:
        """
        Create projection string for use with pyproj.

        This function will automatically build a projection string that can be used as
        the `projection` argument of :func:`generate_map_coordinates`.


        By default, this function will set the `+a`, `+b` and `+axis` parameters of the
        projection:

        - `+a` is set to the equatorial radius of the target body (i.e. the value of
          :attr:`Body.r_eq`).
        - `+b` is set to the polar radius of the target body (i.e. the value of
          :attr:`Body.r_polar`).
        - `+axis` is set to match the :attr:`Body.positive_longitude_direction` of the
          target body. If the target body has a positive longitude direction of E, then
          the projection will have `+axis=enu`, if the target body has a positive
          longitude direction of W, then the projection will have `+axis=wnu`

        See https://proj.org/en/stable/usage/ellipsoids.html for more details on the
        `+a` and `+b` parameters, and see
        https://proj.org/usage/projections.html#axis-orientation for more details about
        the `+axis` projection parameter.

        To prevent any of these parameters from being set, pass `None` as the value for
        that parameter. You can also override the default by passing a different value
        for `a`, `b` or `axis`.

        Examples: ::

            body = planetmapper.BodyXY('Jupiter')

            body.create_proj_string('ortho')
            # '+proj=ortho +a=71492.0 +b=66854.0 +axis=wnu +type=crs'

            body.create_proj_string('ortho', lon_0=180, lat_0=45)
            # '+proj=ortho +lon_0=180 +lat_0=45 +a=71492.0 +b=66854.0 +axis=wnu +type=crs'

            body.create_proj_string('ortho', lon_0=180, lat_0=45, axis=None, a=None, b=None)
            # '+proj=ortho +lon_0=180 +lat_0=45 +type=crs'

            body.create_proj_string('ortho', a=1, b=None, axis='enu')
            # '+proj=ortho +a=1 +axis=enu +type=crs'

        Args:
            proj: Projection name. See https://proj.org/operations/projections
                for a full list of projections that can be used.
            **parameters: Additional parameters to pass to the projection. These are
                passed to pyproj as `+{key}={value}`. For example, to create a
                projection with a central longitude of 45 degrees, you can use
                `lon_0=45`. If unspecified, the `a`, `b` and `axis` parameters will
                automatically be set to appropriate values - this can be disabled by
                passing `None` as the value for any of these parameters.

        Returns:
            Proj string describing the projection. This can be passed to the
            `projection` argument of :func:`generate_map_coordinates`.

        .. versionchanged:: 1.12.2
            Added `+a` and `+b` parameters to default projection string.
        """
        # https://proj.org/en/stable/usage/ellipsoids.html#ellipsoid-size-parameters
        if 'a' not in parameters:
            parameters['a'] = self.r_eq
        if 'b' not in parameters:
            parameters['b'] = self.r_polar
        if 'axis' not in parameters:
            parameters['axis'] = f'{self.positive_longitude_direction.lower()}nu'

        for k in [k for k, v in parameters.items() if v is None]:
            # Use list comprehension to avoid modifying dict during iteration
            parameters.pop(k)

        parameters_string = ' '.join(f'+{k}={v}' for k, v in parameters.items())
        space = ' ' if parameters_string else ''  # avoid double space if params empty
        return f'+proj={proj} {parameters_string}{space}+type=crs'

    def _check_proj_string_for_axis(self, projection: str) -> None:
        expected_axis = f'+axis={self.positive_longitude_direction.lower()}nu'
        if expected_axis not in projection:
            raise ProjStringError(
                f'Projection string {projection!r} does not have the expected axis '
                f'orientation {expected_axis!r} '
                f'for positive {self.positive_longitude_direction} coordinates.'
            )

    def _get_pyproj_map_coords(
        self,
        projection: str,
        xx: np.ndarray | tuple,
        yy: np.ndarray | tuple | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pyproj.Transformer]:
        if yy is None:
            yy = xx
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        if xx.ndim != yy.ndim:
            raise ValueError('x and y coords must have the same number of dimensions')
        if xx.ndim == 1:
            xx, yy = np.meshgrid(xx, yy)
        if xx.ndim != 2:
            raise ValueError('x and y coords must be 1D or 2D arrays')
        if xx.shape != yy.shape:
            raise ValueError('x and y coords must have the same shape')

        self._check_proj_string_for_axis(projection)
        transformer = self._get_pyproj_transformer(projection)
        lons, lats = transformer.transform(xx, yy, direction='INVERSE')
        return lons, lats, xx, yy, transformer

    def _get_default_pyproj_projection(
        self, ellipsoid: Any | None = None, **parameters
    ) -> str:
        if ellipsoid is None:
            a = None
            b = None
        else:
            a = ellipsoid.semi_major_metre
            b = ellipsoid.semi_minor_metre
        return self.create_proj_string('lonlat', a=a, b=b, **parameters)

    def _get_pyproj_transformer(
        self, projection: str | None = None
    ) -> pyproj.Transformer:
        if projection is None:
            projection = self._get_default_pyproj_projection(axis=None)
        crs_out = pyproj.CRS(projection)
        proj_in = self._get_default_pyproj_projection(
            axis=None, ellipsoid=crs_out.ellipsoid
        )
        return pyproj.Transformer.from_crs(pyproj.CRS(proj_in), crs_out)

    # Backplane generatotrs
    def _test_if_img_size_valid(self) -> bool:
        return (self._nx > 0) and (self._ny > 0)

    def _iterate_image(
        self, shape: tuple[int, ...], progress: bool = False
    ) -> Iterator[tuple[int, int]]:
        ny = shape[0]
        nx = shape[1]
        for y in range(ny):
            if progress:
                self._update_progress_hook(y / ny)
            for x in range(nx):
                yield y, x

    def _make_empty_img(self, nz: int | None = None) -> np.ndarray:
        if not self._test_if_img_size_valid():
            raise ValueError('nx and ny must be positive to create a backplane image')
        if nz is None:
            shape = (self._ny, self._nx)
        else:
            shape = (self._ny, self._nx, nz)
        return np.full(shape, np.nan)

    def _make_empty_map(
        self,
        nz: int | None = None,
        **map_kwargs: Unpack[MapKwargs],
    ) -> np.ndarray:
        lonlat_shape = self._get_lonlat_map(**map_kwargs).shape
        n1 = lonlat_shape[1]
        n0 = lonlat_shape[0]
        if nz is None:
            shape = (n0, n1)
        else:
            shape = (n0, n1, nz)
        return np.full(shape, np.nan)

    def _get_max_pixel_radius(self) -> float:
        # r0 corresponds to r_eq, but for the radius here we want to make sure we have
        # the largest radius
        r = self.get_r0() * max(self.radii) / self.r_eq
        return r

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    def _get_targvec_img(self) -> np.ndarray:
        out = self._make_empty_img(3)

        # Precalculate short circuit stuff here for speed
        r_cutoff = self._get_max_pixel_radius() * 1.05 + 1
        r2 = r_cutoff**2  # square here to save having to run sqrt every loop
        x0 = self.get_x0()
        y0 = self.get_y0()

        for y, x in self._iterate_image(out.shape, progress=True):
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

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    def _get_targvec_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(3, **map_kwargs)
        lonlats = self._get_lonlat_map(**map_kwargs)
        for a, b in self._iterate_image(out.shape, progress=True):
            lon, lat = lonlats[a, b]
            if math.isnan(lon):
                continue
            out[a, b] = self.lonlat2targvec(lon, lat)
        return out

    def _enumerate_targvec_img(
        self, progress: bool = False
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        targvec_img = self._get_targvec_img()
        for y, x in self._iterate_image(targvec_img.shape, progress=progress):
            targvec = targvec_img[y, x]
            if math.isnan(targvec[0]):
                # only check if first element nan for efficiency
                continue
            yield y, x, targvec

    def _enumerate_targvec_map(
        self, progress: bool = False, **map_kwargs: Unpack[MapKwargs]
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        targvec_map = self._get_targvec_map(**map_kwargs)
        for a, b in self._iterate_image(targvec_map.shape, progress=progress):
            targvec = targvec_map[a, b]
            if math.isnan(targvec[0]):
                # only check if first element nan for efficiency
                continue
            yield a, b, targvec_map[a, b]

    @_cache_clearable_result
    def _get_obsvec_norm_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        ra_img = self.get_ra_img()
        dec_img = self.get_dec_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = self._radec2obsvec_norm_radians(
                *self._degree_pair2radians(ra_img[y, x], dec_img[y, x])
            )
        return out

    @_cache_stable_result
    @_adjust_surface_altitude_decorator
    def _get_obsvec_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(3, **map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(**map_kwargs):
            out[a, b] = self._targvec2obsvec(targvec)
        return out

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def _get_lonlat_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._targvec2lonlat_radians(targvec)
        return np.rad2deg(out)

    @_cache_stable_result
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_lonlat_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        lons, lats, xx, yy, transformer, info = self.generate_map_coordinates(
            **map_kwargs
        )
        lons = lons % 360
        lonlat_map = np.stack([lons, lats], axis=-1)
        lonlat_map[~np.isfinite(lonlat_map)] = np.nan
        return lonlat_map

    def get_lon_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetographic
            longitude value of each pixel in the image. Points off the disc have a value
            of NaN.
        """
        return self._get_lonlat_img()[:, :, 0]

    def get_lon_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of planetographic
            longitude values.
        """
        return self._get_lonlat_map(**map_kwargs)[:, :, 0]

    def get_lat_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetographic
            latitude value of each pixel in the image. Points off the disc have a value
            of NaN.
        """
        return self._get_lonlat_img()[:, :, 1]

    def get_lat_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of planetographic
            latitude values.
        """
        return self._get_lonlat_map(**map_kwargs)[:, :, 1]

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def _get_lonlat_centric_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._targvec2lonlat_centric(targvec)
        return out

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_lonlat_centric_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(2, **map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            out[a, b] = self._targvec2lonlat_centric(targvec)
        return out

    def get_lon_centric_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetocentric
            longitude value of each pixel in the image. Points off the disc have a value
            of NaN.
        """
        return self._get_lonlat_centric_img()[:, :, 0]

    def get_lon_centric_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of planetocentric
            longitude values.
        """
        return self._get_lonlat_centric_map(**map_kwargs)[:, :, 0]

    def get_lat_centric_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetocentric
            latitude value of each pixel in the image. Points off the disc have a value
            of NaN.
        """
        return self._get_lonlat_centric_img()[:, :, 1]

    def get_lat_centric_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of planetocentric
            latitude values.
        """
        return self._get_lonlat_centric_map(**map_kwargs)[:, :, 1]

    @_cache_clearable_result
    @_return_readonly_array
    @progress_decorator
    @_return_readonly_array
    def _get_radec_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x in self._iterate_image(out.shape, progress=True):
            out[y, x] = self.xy2radec(x, y)
        return out

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_radec_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(2, **map_kwargs)
        # (phase, incdnc, emissn, visibl, lit) in illumf map
        visible = self._get_illumf_map(**map_kwargs)[:, :, 3]
        obsvec_map = self._get_obsvec_map(**map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            # use targvec iterator to ensure don't have NaNs
            if visible[a, b]:
                out[a, b] = self._obsvec2radec_radians(obsvec_map[a, b])
        return np.rad2deg(out)

    def get_ra_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the right ascension (RA)
            value of each pixel in the image.
        """
        return self._get_radec_img()[:, :, 0]

    def get_ra_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of right ascension
            values as viewed by the observer. Locations which are not visible have a
            value of NaN.
        """
        return self._get_radec_map(**map_kwargs)[:, :, 0]

    def get_dec_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the declination (Dec)
            value of each pixel in the image.
        """
        return self._get_radec_img()[:, :, 1]

    def get_dec_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of declination values
            as viewed by the observer. Locations which are not visible have a value of
            NaN.
        """
        return self._get_radec_map(**map_kwargs)[:, :, 1]

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_xy_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(2, **map_kwargs)
        radec_map = self._get_radec_map(**map_kwargs)
        for a, b in self._iterate_image(out.shape, progress=True):
            ra, dec = radec_map[a, b]
            if not math.isnan(ra):
                x, y = self.radec2xy(ra, dec)
                if self._xy_in_image_frame(x, y):
                    out[a, b] = x, y
        return out

    @_return_readonly_array
    def get_x_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the x pixel coordinate
            value of each pixel in the image.
        """
        out = self._make_empty_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = x
        return out

    def get_x_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the x pixel
            coordinates each location corresponds to in the observation. Locations which
            are not visible or are not in the image frame have a value of NaN.
        """
        return self._get_xy_map(**map_kwargs)[:, :, 0]

    @_return_readonly_array
    def get_y_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the y pixel coordinate
            value of each pixel in the image.
        """
        out = self._make_empty_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = y
        return out

    def get_y_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the y pixel
            coordinates each location corresponds to in the observation. Locations which
            are not visible or are not in the image frame have a value of NaN.
        """
        return self._get_xy_map(**map_kwargs)[:, :, 1]

    @_cache_clearable_result
    @_return_readonly_array
    def _get_km_xy_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        radec_img = self._get_radec_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = self.radec2km(*radec_img[y, x])
        return out

    @_cache_stable_result
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_km_xy_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(2, **map_kwargs)
        radec_map = self._get_radec_map(**map_kwargs)
        for a, b in self._iterate_image(out.shape, progress=True):
            ra, dec = radec_map[a, b]
            if not math.isnan(ra):
                out[a, b] = self.radec2km(ra, dec)
        return out

    def get_km_x_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the distance in target
            plane in km in the East-West direction.
        """
        return self._get_km_xy_img()[:, :, 0]

    def get_km_x_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the distance in
            target plane in km in the East-West direction. Locations which are not
            visible have a value of NaN.
        """
        return self._get_km_xy_map(**map_kwargs)[:, :, 0]

    def get_km_y_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the distance in target
            plane in km in the North-South direction.
        """
        return self._get_km_xy_img()[:, :, 1]

    def get_km_y_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the distance in
            target plane in km in the North-South direction. Locations which are not
            visible have a value of NaN.
        """
        return self._get_km_xy_map(**map_kwargs)[:, :, 1]

    @_return_readonly_array
    def get_angular_x_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the angular distance in
            target plane in arcseconds in the East-West direction.
        """
        return self.get_km_x_img() / self.km_per_arcsec

    @_return_readonly_array
    def get_angular_x_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the angular
            distance in target plane in arcseconds in the East-West direction.
            Locations which are not visible have a value of NaN.
        """
        return self.get_km_x_map(**map_kwargs) / self.km_per_arcsec

    @_return_readonly_array
    def get_angular_y_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the angular distance in
            target plane in arcseconds in the North-South direction.
        """
        return self.get_km_y_img() / self.km_per_arcsec

    @_return_readonly_array
    def get_angular_y_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the angular
            distance in target plane in arcseconds in the North-South direction.
            Locations which are not visible have a value of NaN.
        """
        return self.get_km_y_map(**map_kwargs) / self.km_per_arcsec

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def _get_illumination_gie_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._illumination_angles_from_targvec_radians(targvec)
        return np.rad2deg(out)

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_illumf_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(5, **map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            out[a, b] = self._illumf_from_targvec_radians(targvec)
        return np.rad2deg(out)

    def get_phase_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the phase angle value of
            each pixel in the image. Points off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 0]

    def get_phase_angle_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the phase angle
            value at each point on the target's surface.
        """
        return self._get_illumf_map(**map_kwargs)[:, :, 0]

    def get_incidence_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the incidence angle value
            of each pixel in the image. Points off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 1]

    def get_incidence_angle_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the incidence
            angle value at each point on the target's surface.
        """
        return self._get_illumf_map(**map_kwargs)[:, :, 1]

    def get_emission_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the emission angle value
            of each pixel in the image. Points off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 2]

    def get_emission_angle_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the emission angle
            value at each point on the target's surface.
        """
        return self._get_illumf_map(**map_kwargs)[:, :, 2]

    @_cache_clearable_alt_dependent_result
    @_return_readonly_array
    def get_azimuth_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the azimuth angle value
            of each pixel in the image. Points off the disc have a value of NaN.
        """
        phase_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 0])
        incidence_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 1])
        emission_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero encountered in')
            warnings.filterwarnings('ignore', 'invalid value encountered in')
            azimuth_radians = self._azimuth_angle_from_gie_radians(
                phase_radians, incidence_radians, emission_radians
            )
        return np.rad2deg(azimuth_radians)

    @_cache_stable_result
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def get_azimuth_angle_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the azimuth angle
            value at each point on the target's surface.
        """
        phase_radians = np.deg2rad(self._get_illumf_map(**map_kwargs)[:, :, 0])
        incidence_radians = np.deg2rad(self._get_illumf_map(**map_kwargs)[:, :, 1])
        emission_radians = np.deg2rad(self._get_illumf_map(**map_kwargs)[:, :, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero encountered in')
            warnings.filterwarnings('ignore', 'invalid value encountered in')
            azimuth_radians = self._azimuth_angle_from_gie_radians(
                phase_radians, incidence_radians, emission_radians
            )
        return np.rad2deg(azimuth_radians)

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def get_local_solar_time_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the local solar time
            value of each pixel in the image, as calculated by
            :func:`Body.local_solar_time_from_lon`. Points off the disc have a value of
            NaN.
        """
        lon_img = self.get_lon_img()
        out = self._make_empty_img()
        for y, x in self._iterate_image(out.shape, progress=True):
            lon = lon_img[y, x]
            if not math.isnan(lon):
                out[y, x] = self.local_solar_time_from_lon(lon)
        return out

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def get_local_solar_time_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the local solar
            time at each point on the target's surface, as calculated by
            :func:`Body.local_solar_time_from_lon`.
        """
        lon_map = self.get_lon_map(**map_kwargs)
        out = self._make_empty_map(**map_kwargs)
        for a, b in self._iterate_image(out.shape, progress=True):
            lon = lon_map[a, b]
            if math.isfinite(lon):
                out[a, b] = self.local_solar_time_from_lon(lon)
        return out

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    def _get_state_imgs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_img = self._make_empty_img(3)
        velocity_img = self._make_empty_img(3)
        lt_img = self._make_empty_img()
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            (
                position_img[y, x],
                velocity_img[y, x],
                lt_img[y, x],
            ) = self._state_from_targvec(targvec)
        return (
            _as_readonly_view(position_img),
            _as_readonly_view(velocity_img),
            _as_readonly_view(lt_img),
        )

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    def _get_state_maps(
        self, **map_kwargs: Unpack[MapKwargs]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_map = self._make_empty_map(3, **map_kwargs)
        velocity_map = self._make_empty_map(3, **map_kwargs)
        lt_map = self._make_empty_map(**map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            (
                position_map[a, b],
                velocity_map[a, b],
                lt_map[a, b],
            ) = self._state_from_targvec(targvec)
        return (
            _as_readonly_view(position_map),
            _as_readonly_view(velocity_map),
            _as_readonly_view(lt_map),
        )

    @_return_readonly_array
    def get_distance_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the observer-target
            distance in km of each pixel in the image. Points off the disc have a value
            of NaN.
        """
        position_img, velocity_img, lt_img = self._get_state_imgs()
        return lt_img * self.speed_of_light()

    @_return_readonly_array
    def get_distance_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the
            observer-target distance in km of each point on the target's surface.
        """
        position_map, velocity_map, lt_map = self._get_state_maps(**map_kwargs)
        return lt_map * self.speed_of_light()

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def get_radial_velocity_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the observer-target
            radial velocity in km/s of each pixel in the image. Velocities towards the
            observer are negative. Points off the disc have a value of NaN.
        """
        out = self._make_empty_img()
        position_img, velocity_img, lt_img = self._get_state_imgs()
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._radial_velocity_from_state(
                position_img[y, x], velocity_img[y, x]
            )
        return out

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def get_radial_velocity_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the
            observer-target radial velocity in km/s of each point on the target's
            surface.
        """
        out = self._make_empty_map(**map_kwargs)
        position_map, velocity_map, lt_map = self._get_state_maps(**map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            out[a, b] = self._radial_velocity_from_state(
                position_map[a, b], velocity_map[a, b]
            )
        return out

    @_return_readonly_array
    def get_doppler_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the doppler factor for
            each pixel in the image, calculated using
            :func:`SpiceBase.calculate_doppler_factor` on velocities from
            :func:`get_radial_velocity_img`. Points off the disc have a value of NaN.
        """
        return self.calculate_doppler_factor(self.get_radial_velocity_img())

    @_return_readonly_array
    def get_doppler_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the doppler factor
            of each point on the target's surface. This is calculated using
            :func:`SpiceBase.calculate_doppler_factor` on velocities from
            :func:`get_radial_velocity_map`.
        """
        return self.calculate_doppler_factor(self.get_radial_velocity_map(**map_kwargs))

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    @_return_readonly_array
    def _get_limb_coordinate_imgs(self) -> np.ndarray:
        out = self._make_empty_img(3)
        obsvec_img = self._get_obsvec_norm_img()
        for y, x in self._iterate_image(out.shape, progress=True):
            obsvec = obsvec_img[y, x]
            out[y, x] = self._limb_coordinates_from_obsvec(obsvec)
        return out

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    @_return_readonly_array
    def _get_limb_coordinate_maps(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        out = self._make_empty_map(3, **map_kwargs)
        visible = self._get_illumf_map(**map_kwargs)[:, :, 4]
        obsvec_map = self._get_obsvec_map(**map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            if visible[a, b]:
                out[a, b] = self._limb_coordinates_from_obsvec(obsvec_map[a, b])
        return out

    def get_limb_lon_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetographic
            longitude of the point on the target's limb that is closest to each pixel.
            See :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_imgs()[:, :, 0]

    def get_limb_lon_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the planetographic
            longitude of the point on the target's limb that is closest to each point on
            the target's surface (for the observer). See
            :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_maps(**map_kwargs)[:, :, 0]

    def get_limb_lat_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the planetographic
            latitude of the point on the target's limb that is closest to each pixel.
            See :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_imgs()[:, :, 1]

    def get_limb_lat_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the planetographic
            latitude of the point on the target's limb that is closest to each point on
            the target's surface (for the observer). See
            :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_maps(**map_kwargs)[:, :, 1]

    def get_limb_distance_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the distance in km above
            the target's limb for each pixel. See
            :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_imgs()[:, :, 2]

    def get_limb_distance_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the distance in km
            above the target's limb for each point on the target's surface (for the
            observer). See :func:`Body.limb_coordinates_from_radec` for more detail.
        """
        return self._get_limb_coordinate_maps(**map_kwargs)[:, :, 2]

    @_cache_clearable_alt_dependent_result
    @progress_decorator
    def _get_ring_plane_coordinate_imgs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_img = self._make_empty_img()
        long_img = self._make_empty_img()
        dist_img = self._make_empty_img()

        obsvec_img = self._get_obsvec_norm_img()
        for y, x in self._iterate_image(radius_img.shape, progress=True):
            radius, long, dist = self._ring_coordinates_from_obsvec(
                obsvec_img[y, x], only_visible=False
            )
            radius_img[y, x] = radius
            long_img[y, x] = long
            dist_img[y, x] = dist

        hidden_img = dist_img > self.get_distance_img()
        radius_img[hidden_img] = np.nan
        long_img[hidden_img] = np.nan
        dist_img[hidden_img] = np.nan
        return (
            _as_readonly_view(radius_img),
            _as_readonly_view(long_img),
            _as_readonly_view(dist_img),
        )

    @_cache_stable_result
    @progress_decorator
    @_adjust_surface_altitude_decorator
    def _get_ring_plane_coordinate_maps(
        self, **map_kwargs: Unpack[MapKwargs]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_map = self._make_empty_map(**map_kwargs)
        long_map = self._make_empty_map(**map_kwargs)
        dist_map = self._make_empty_map(**map_kwargs)

        visible = self._get_illumf_map(**map_kwargs)[:, :, 4]
        obsvec_map = self._get_obsvec_map(**map_kwargs)
        for a, b, targvec in self._enumerate_targvec_map(progress=True, **map_kwargs):
            if visible[a, b]:
                radius, long, dist = self._ring_coordinates_from_obsvec(
                    obsvec_map[a, b], only_visible=False
                )
                radius_map[a, b] = radius
                long_map[a, b] = long
                dist_map[a, b] = dist

        hidden_map = dist_map > self.get_distance_map(**map_kwargs)
        radius_map[hidden_map] = np.nan
        long_map[hidden_map] = np.nan
        dist_map[hidden_map] = np.nan
        return (
            _as_readonly_view(radius_map),
            _as_readonly_view(long_map),
            _as_readonly_view(dist_map),
        )

    def get_ring_plane_radius_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the ring plane radius in
            km for each pixel in the image, calculated using
            :func:`Body.ring_plane_coordinates`. Points of the ring plane obscured by
            the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[0]

    def get_ring_plane_radius_map(self, **map_kwargs: Unpack[MapKwargs]) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the ring plane
            radius in km obscuring each point on the target's surface, calculated using
            :func:`Body.ring_plane_coordinates`. Points where the target body is
            unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(**map_kwargs)[0]

    def get_ring_plane_longitude_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the ring plane
            planetographic longitude in degrees for each pixel in the image, calculated
            using :func:`Body.ring_plane_coordinates`. Points of the ring plane obscured
            by the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[1]

    def get_ring_plane_longitude_map(
        self, **map_kwargs: Unpack[MapKwargs]
    ) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the ring plane
            planetographic longitude in degrees obscuring each point on the target's
            surface, calculated using :func:`Body.ring_plane_coordinates`. Points where
            the target body is unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(**map_kwargs)[1]

    def get_ring_plane_distance_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing the ring plane distance
            from the observer in km for each pixel in the image, calculated using
            :func:`Body.ring_plane_coordinates`. Points of the ring plane obscured by
            the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[2]

    def get_ring_plane_distance_map(
        self, **map_kwargs: Unpack[MapKwargs]
    ) -> np.ndarray:
        """
        See :func:`generate_map_coordinates` for accepted arguments. See also
        :func:`get_backplane_map`.

        Returns:
            :ref:`Read-only array<readonly arrays>` containing map of the ring plane
            distance from the observer in km obscuring each point on the target's
            surface, calculated using :func:`Body.ring_plane_coordinates`. Points where
            the target body is unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(**map_kwargs)[2]

    # Default backplane registration
    def _register_default_backplanes(self) -> None:
        self.register_backplane(
            'LON-GRAPHIC',
            'Planetographic longitude, positive {ew} [deg]'.format(
                ew=self.positive_longitude_direction
            ),
            self.get_lon_img,
            self.get_lon_map,
        )
        self.register_backplane(
            'LAT-GRAPHIC',
            'Planetographic latitude [deg]',
            self.get_lat_img,
            self.get_lat_map,
        )
        self.register_backplane(
            'LON-CENTRIC',
            'Planetocentric longitude [deg]',
            self.get_lon_centric_img,
            self.get_lon_centric_map,
        )
        self.register_backplane(
            'LAT-CENTRIC',
            'Planetocentric latitude [deg]',
            self.get_lat_centric_img,
            self.get_lat_centric_map,
        )
        self.register_backplane(
            'RA',
            'Right ascension [deg]',
            self.get_ra_img,
            self.get_ra_map,
        )
        self.register_backplane(
            'DEC',
            'Declination [deg]',
            self.get_dec_img,
            self.get_dec_map,
        )
        self.register_backplane(
            'PIXEL-X',
            'Observation x pixel coordinate [pixels]',
            self.get_x_img,
            self.get_x_map,
        )
        self.register_backplane(
            'PIXEL-Y',
            'Observation y pixel coordinate [pixels]',
            self.get_y_img,
            self.get_y_map,
        )
        self.register_backplane(
            'KM-X',
            'East-West distance in target plane [km]',
            self.get_km_x_img,
            self.get_km_x_map,
        )
        self.register_backplane(
            'KM-Y',
            'North-South distance in target plane [km]',
            self.get_km_y_img,
            self.get_km_y_map,
        )
        self.register_backplane(
            'ANGULAR-X',
            'East-West distance in target plane [arcsec]',
            self.get_angular_x_img,
            self.get_angular_x_map,
        )
        self.register_backplane(
            'ANGULAR-Y',
            'North-South distance in target plane [arcsec]',
            self.get_angular_y_img,
            self.get_angular_y_map,
        )
        self.register_backplane(
            'PHASE',
            'Phase angle [deg]',
            self.get_phase_angle_img,
            self.get_phase_angle_map,
        )
        self.register_backplane(
            'INCIDENCE',
            'Incidence angle [deg]',
            self.get_incidence_angle_img,
            self.get_incidence_angle_map,
        )
        self.register_backplane(
            'EMISSION',
            'Emission angle [deg]',
            self.get_emission_angle_img,
            self.get_emission_angle_map,
        )
        self.register_backplane(
            'AZIMUTH',
            'Azimuth angle [deg]',
            self.get_azimuth_angle_img,
            self.get_azimuth_angle_map,
        )
        self.register_backplane(
            'LOCAL-SOLAR-TIME',
            'Local solar time [local hours]',
            self.get_local_solar_time_img,
            self.get_local_solar_time_map,
        )
        self.register_backplane(
            'DISTANCE',
            'Distance to observer [km]',
            self.get_distance_img,
            self.get_distance_map,
        )
        self.register_backplane(
            'RADIAL-VELOCITY',
            'Radial velocity away from observer [km/s]',
            self.get_radial_velocity_img,
            self.get_radial_velocity_map,
        )
        self.register_backplane(
            'DOPPLER',
            'Doppler factor, sqrt((1 + v/c)/(1 - v/c)) where v is radial velocity',
            self.get_doppler_img,
            self.get_doppler_map,
        )
        self.register_backplane(
            'LIMB-DISTANCE',
            'Distance above limb [km]',
            self.get_limb_distance_img,
            self.get_limb_distance_map,
        )
        self.register_backplane(
            'LIMB-LON-GRAPHIC',
            'Planetographic longitude of closest point on the limb [deg]',
            self.get_limb_lon_img,
            self.get_limb_lon_map,
        )
        self.register_backplane(
            'LIMB-LAT-GRAPHIC',
            'Planetographic latitude of closest point on the limb [deg]',
            self.get_limb_lat_img,
            self.get_limb_lat_map,
        )
        self.register_backplane(
            'RING-RADIUS',
            'Equatorial (ring) plane radius [km]',
            self.get_ring_plane_radius_img,
            self.get_ring_plane_radius_map,
        )
        self.register_backplane(
            'RING-LON-GRAPHIC',
            'Equatorial (ring) plane planetographic longitude [deg]',
            self.get_ring_plane_longitude_img,
            self.get_ring_plane_longitude_map,
        )
        self.register_backplane(
            'RING-DISTANCE',
            'Equatorial (ring) plane distance to observer [km]',
            self.get_ring_plane_distance_img,
            self.get_ring_plane_distance_map,
        )


class BackplaneNotFoundError(Exception):
    pass


def _make_backplane_documentation_str() -> str:
    class _BodyXY_ForDocumentation(BodyXY):
        # pylint: disable-next=super-init-not-called
        def __init__(self):
            self.backplanes = {}
            self.positive_longitude_direction = 'W'
            self._register_default_backplanes()

    body = _BodyXY_ForDocumentation()

    msg = []
    msg.append('..')
    msg.append('   THIS CONTENT IS AUTOMATICALLY GENERATED')
    msg.append('')

    msg.append('.. _default backplanes:')
    msg.append('')
    msg.append('Default backplanes')
    msg.append('*' * len(msg[-1]))
    msg.append('')

    msg.append(
        'This page lists the backplanes which are automatically registered to '
        'every instance of :class:`planetmapper.BodyXY`.'
    )
    msg.append('')

    for bp in body.backplanes.values():
        msg.append('------------')
        msg.append('')
        msg.append('`{}` {}'.format(bp.name, bp.description))
        msg.append('')
        msg.append(
            '- Image function: :func:`planetmapper.{}`'.format(bp.get_img.__qualname__)
        )
        msg.append(
            '- Map function: :func:`planetmapper.{}`'.format(
                bp.get_map.__qualname__  # type: ignore
            )
        )
        msg.append('')

    msg.append('------------')
    msg.append('')
    msg.append('Wireframe images')
    msg.append('=' * len(msg[-1]))
    msg.append('')

    msg.append(
        'In addition to the above backplanes, a `WIREFRAME` backplane is also included '
        'by default in saved FITS files. This backplane contains a "wireframe" image '
        'of the body, which shows latitude/longitude gridlines, labels poles, displays '
        'the body\'s limb etc. These wireframe images can be used to help orient the '
        'observations, and can be used as an overlay if you are creating figures from '
        'the FITS files.'
    )
    msg.append('')

    msg.append(
        'The wireframe images are a graphical guide rather than containing any '
        'scientific data, so they are not registered like the other backplanes. '
        'Note that the wireframe images have a fixed size, so they will not be the '
        'same size as the data/mapped data (although the aspect ratio will be the '
        'same).'
    )
    msg.append('')

    msg.append(
        '- Image function: :func:`planetmapper.{}`'.format(
            body.get_wireframe_overlay_img.__qualname__
        )
    )
    msg.append(
        '- Map function: :func:`planetmapper.{}`'.format(
            body.get_wireframe_overlay_map.__qualname__
        )
    )
    msg.append('')

    return '\n'.join(msg)


def _extract_map_kwargs_from_dict(kwargs_dict) -> tuple[MapKwargs, dict[str, Any]]:
    """
    Split a dictionary of keyword arguments into a dictionary of map kwargs and a
    dictionary of other kwargs.

    Args:
        kwargs_dict: Dictionary of keyword arguments.

    Returns:
        Tuple containing a dictionary of map kwargs and a dictionary of other kwargs.
    """
    # pylint: disable-next=no-member
    map_keys = set(MapKwargs.__optional_keys__) | set(MapKwargs.__required_keys__)
    map_kwargs = MapKwargs()
    other_kwargs = {}
    for k, v in kwargs_dict.items():
        if k in map_keys:
            map_kwargs[k] = v
        else:
            other_kwargs[k] = v
    return map_kwargs, other_kwargs
