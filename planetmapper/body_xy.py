import datetime
import math
from functools import lru_cache, wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeVar,
    Literal,
)
import warnings

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import scipy.interpolate
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from spiceypy.utils.exceptions import NotFoundError

from .body import Body
from .progress import progress_decorator

T = TypeVar('T')
S = TypeVar('S')
P = ParamSpec('P')

_cache_stable_result = lru_cache(maxsize=128)


def _cache_clearable_result(fn: Callable[[S], T]) -> Callable[[S], T]:
    """
    Decorator to cache the output of a method call.

    This requires that the class has a `self._cache` dict which can be used to store
    the cached result. The dictionary key is derived from the name of the decorated
    function.

    The results cached by this decorator can be cleared using `self._cache.clear()`, so
    this is useful for results which need to be invalidated (i.e. backplane images
    which are invalidated the moment the disc params are changed). If the result is
    stable (i.e. backplane maps) then use `_cache_stable_result` instead.
    """

    @wraps(fn)
    def decorated(self):
        k = fn.__name__
        if k not in self._cache:
            self._cache[k] = fn(self)
        return self._cache[k]

    return decorated


def _cache_clearable_result_with_args(
    fn: Callable[Concatenate[S, P], T]
) -> Callable[Concatenate[S, P], T]:
    """
    Decorator to cache the output of a method call with variable arguments.

    This requires that the class has a `self._cache` dict which can be used to store
    the cached result. The dictionary key is derived from the name of the decorated
    function.

    The results cached by this decorator can be cleared using `self._cache.clear()`, so
    this is useful for results which need to be invalidated (i.e. backplane images
    which are invalidated the moment the disc params are changed). If the result is
    stable (i.e. backplane maps) then use `_cache_stable_result` instead.
    """

    @wraps(fn)
    def decorated(self, *args: P.args, **kwargs: P.kwargs):
        k = (fn.__name__, args, frozenset(kwargs.items()))
        if k not in self._cache:
            self._cache[k] = fn(self, *args, **kwargs)
        return self._cache[k]

    return decorated


class _BackplaneMapGetter(Protocol):
    def __call__(self, degree_interval: float = 1) -> np.ndarray:
        ...


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
        get_map: Function returns a numpy array containing a cylindrical map of
            backplane values when called. This should take a single `float` argument,
            `degree_interval` which defines the interval in degrees between the
            longitude/latitude points in the mapped output. `degree_interval` should be
            optional with a default value of 1. This function should generally be a
            method such as :func:`BodyXY.get_lon_map`.
    """

    name: str
    description: str
    get_img: Callable[[], np.ndarray]
    get_map: _BackplaneMapGetter


class BodyXY(Body):
    """
    Class representing an astronomical body imaged at a specific time.

    This is a subclass of :class:`Body` with additional methods to interact with the
    image pixel coordinate frame `xy`. This class assumes the tangent plane
    approximation is valid for the conversion between pixel coordinates `xy` and RA/Dec
    sky coordinates `radec`.

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

        super().__init__(target, utc, **kwargs)

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
        `body.backplanes['LON'].get_img()` or using :func:`get_backplane_img`).
        Generating the same backplane when there are different disc parameter values will
        produce a different image.

        This dictionary is used to define the backplanes saved to the output FITS file
        in :func:`Observation.save`.
        """

        # Run setup
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
            self.centre_disc()

        self._mpl_transform_radec2xy: matplotlib.transforms.Affine2D | None = None
        self._mpl_transform_xy2radec: matplotlib.transforms.Transform | None = None
        self._mpl_transform_radec2xy_radians: matplotlib.transforms.Affine2D | None = (
            None
        )

        self.backplanes = {}
        self._register_default_backplanes()

    def __repr__(self) -> str:
        return f'BodyXY({self.target!r}, {self.utc!r}, {self._nx!r}, {self._ny!r})'

    # Cache management
    def _clear_cache(self):
        """
        Clear cached results from `_cache_result`.
        """
        self._cache.clear()

    # Coordinate transformations
    @_cache_clearable_result
    def _get_xy2radec_matrix_radians(self) -> np.ndarray:
        r_km = self.r_eq
        r_radians = np.arcsin(r_km / self.target_distance)
        s = r_radians / self.get_r0()
        theta = self._get_rotation_radians()
        direction_matrix = np.array([[-1, 0], [0, 1]])
        stretch_matrix = np.array(
            [[1 / np.abs(np.cos(self._target_dec_radians)), 0], [0, 1]]
        )
        rotation_matrix = self._rotation_matrix_radians(theta)
        transform_matrix_2x2 = s * np.matmul(stretch_matrix, rotation_matrix)
        transform_matrix_2x2 = np.matmul(transform_matrix_2x2, direction_matrix)

        v0 = np.array([self.get_x0(), self.get_y0()])
        a0 = np.array([self._target_ra_radians, self._target_dec_radians])
        offset_vector = a0 - np.matmul(transform_matrix_2x2, v0)

        transform_matrix_3x3 = np.identity(3)
        transform_matrix_3x3[:2, :2] = transform_matrix_2x2
        transform_matrix_3x3[:2, 2] = offset_vector

        return transform_matrix_3x3

    @_cache_clearable_result
    def _get_radec2xy_matrix_radians(self) -> np.ndarray:
        return np.linalg.inv(self._get_xy2radec_matrix_radians())

    def _xy2radec_radians(self, x: float, y: float) -> tuple[float, float]:
        a = self._get_xy2radec_matrix_radians().dot(np.array([x, y, 1]))
        return a[0], a[1]

    def _radec2xy_radians(self, ra: float, dec: float) -> tuple[float, float]:
        v = self._get_radec2xy_matrix_radians().dot(np.array([ra, dec, 1]))
        return v[0], v[1]

    # Composite transformations
    def xy2radec(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert image pixel coordinates to RA/Dec sky coordinates.

        Args:
            x: Image pixel coordinate in the x direction.
            y: Image pixel coordinate in the y direction.

        Returns:
            `(ra, dec)` tuple containing the RA/Dec coordinates of the point.
        """
        return self._radian_pair2degrees(*self._xy2radec_radians(x, y))

    def radec2xy(self, ra: float, dec: float) -> tuple[float, float]:
        """
        Convert RA/Dec sky coordinates to image pixel coordinates.

        Args:
            ra: Right ascension of point in the sky of the observer
            dec: Declination of point in the sky of the observer.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point.
        """
        return self._radec2xy_radians(*self._degree_pair2radians(ra, dec))

    def xy2lonlat(self, x: float, y: float, **kwargs) -> tuple[float, float]:
        """
        Convert image pixel coordinates to longitude/latitude coordinates on the target
        body.

        Args:
            x: Image pixel coordinate in the x direction.
            y: Image pixel coordinate in the y direction.
            **kwargs: Additional arguments are passed to :func:`Body.radec2lonlat`.

        Returns:
            `(lon, lat)` tuple containing the longitude and latitude of the point. If
            the provided pixel coordinates are missing the target body, then the `lon`
            and `lat` values will both be NaN (see :func:`Body.radec2lonlat`).
        """
        return self.radec2lonlat(*self.xy2radec(x, y), **kwargs)

    def lonlat2xy(self, lon: float, lat: float) -> tuple[float, float]:
        """
        Convert longitude/latitude on the target body to image pixel coordinates.

        Args:
            lon: Longitude of point on target body.
            lat: Latitude of point on target body.

        Returns:
            `(x, y)` tuple containing the image pixel coordinates of the point.
        """
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
        example, if `nx = 20` and `ny = 30`, then `x0` will be set to 10, `y0` will be
        set to 15 and `r0` will be set to 9. The rotation of the disc is unchanged.
        """
        self.set_x0(self._nx / 2)
        self.set_y0(self._ny / 2)
        self.set_r0(0.9 * (min(self.get_x0(), self.get_y0())))
        self.set_disc_method('centre_disc')

    def set_x0(self, x0: float) -> None:
        """
        Args:
            x0: New x pixel coordinate of the centre of the target body.

        Raises:
            ValueError: if `x0` is not finite.
        """
        if not math.isfinite(x0):
            raise ValueError('x0 must be finite')
        self._x0 = float(x0)
        self._clear_cache()

    def get_x0(self) -> float:
        """
        Returns:
            x pixel coordinate of the centre of the target body.
        """
        return self._x0

    def set_y0(self, y0: float) -> None:
        """
        Args:
            y0: New y pixel coordinate of the centre of the target body.

        Raises:
            ValueError: if `y0` is not finite.
        """
        if not math.isfinite(y0):
            raise ValueError('y0 must be finite')
        self._y0 = float(y0)
        self._clear_cache()

    def get_y0(self) -> float:
        """
        Returns:
            y pixel coordinate of the centre of the target body.
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
        self._clear_cache()

    def get_r0(self) -> float:
        """
        Returns:
            Equatorial radius in pixels of the target body.
        """
        return self._r0

    def _set_rotation_radians(self, rotation: float) -> None:
        self._rotation_radians = float(rotation % (2 * np.pi))
        self._clear_cache()

    def _get_rotation_radians(self) -> float:
        return self._rotation_radians

    def set_rotation(self, rotation: float) -> None:
        """
        Set the rotation of the target body.

        This rotation defines the angle between the upwards (positive `dec`) direction
        in the RA/Dec sky coordinates and the upwards (positive `y`) direction in the
        image pixel coordinates.

        Args:
            rotation: New rotation of the target body.

        Raises:
            ValueError: if `rotation` is not finite.
        """
        if not math.isfinite(rotation):
            raise ValueError('rotation must be finite')
        self._set_rotation_radians(np.deg2rad(rotation))

    def get_rotation(self) -> float:
        """
        Returns:
            Rotation of the target body.
        """
        return np.rad2deg(self._get_rotation_radians())

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
        self.set_r0(self.r_eq / km_per_px)

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
        return self.r_eq / self.get_r0()

    def set_img_size(self, nx: int | None = None, ny: int | None = None) -> None:
        """
        Set the `nx` and `ny` values which specify the number of pixels in the x and y
        dimension of the image respectively. Unspecified values will remain unchanged.

        Args:
            nx: If specified, set the number of pixels in the x dimension.
            ny: If specified, set the number of pixels in the y dimension.
        """
        if nx is not None:
            self._nx = nx
        if ny is not None:
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

    def set_disc_method(self, method: str):
        """
        Record the method used to find the coordinates of the target body's disc. This
        recorded method can then be used when metadata is saved, such as in
        :func:`Observation.save`.

        `set_disc_method` is called automatically by functions which find the disc, such
        as :func:`set_x0` and :func:`Observation.centre_disc`, so does not normally need
        to be manually called by the user.

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

        Args:
            dra_arcsec: Offset in arcseconds in the positive right ascension direction.
            ddec_arcsec: Offset in arcseconds in the positive declination direction.
        """
        dra = dra_arcsec / 3600
        ddec = ddec_arcsec / 3600
        ra0, dec0 = self.xy2radec(0, 0)
        dx, dy = self.radec2xy(ra0 + dra, dec0 + ddec)
        self.adjust_disc_params(dx=dx, dy=dy)

    # Illumination functions etc.
    def limb_xy(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Pixel coordinate version of :func:`Body.limb_radec`.

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
        Pixel coordinate version of :func:`Body.limb_radec_by_illumination`.

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
        Pixel coordinate version of :func:`Body.terminator_radec`.

        Args:
            **kwargs: Passed to :func:`Body.terminator_radec`.

        Returns:
            `(x, y)` tuple of coordinate arrays.
        """
        return self._radec_arrs2xy_arrs(*self.terminator_radec(**kwargs))

    def visible_lonlat_grid_xy(
        self, *args, **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Pixel coordinate version of :func:`Body.visible_lonlat_grid_radec`.

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
        Pixel coordinate version of :func:`Body.ring_radec`.

        Args:
            radius: Radius in km of the ring from the centre of the target body.
            **kwargs: Passed to :func:`Body.ring_radec`.

        Returns:
            `(x, y)` tuple of coordinate arrays.
        """
        return self._radec_arrs2xy_arrs(*self.ring_radec(radius, **kwargs))

    # Matplotlib transforms
    def _get_matplotlib_radec2xy_transform_radians(
        self,
    ) -> matplotlib.transforms.Affine2D:
        if self._mpl_transform_radec2xy_radians is None:
            self._mpl_transform_radec2xy_radians = matplotlib.transforms.Affine2D(
                self._get_radec2xy_matrix_radians()
            )
        return self._mpl_transform_radec2xy_radians

    def matplotlib_radec2xy_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        """
        Get matplotlib transform which converts RA/Dec sky coordinates to image pixel
        coordinates.

        The transform is a mutable object which can be dynamically updated using
        :func:`update_transform` when the `radec` to `xy` coordinate conversion changes.
        This can be useful for plotting data (e.g. the planet's limb) using RA/Dec
        coordinates onto an axis using image pixel coordinates when fitting the disc.

        Args:
            ax: Optionally specify a matplotlib axis to return
                `transform_radec2xy + ax.transData`. This value can then be used in the
                `transform` keyword argument of a Matplotlib function without any
                further modification.

        Returns:
            Matplotlib transformation from `radec` to `xy` coordinates.
        """
        if self._mpl_transform_radec2xy is None:
            transform_rad2deg = matplotlib.transforms.Affine2D().scale(np.deg2rad(1))
            self._mpl_transform_radec2xy = (
                transform_rad2deg + self._get_matplotlib_radec2xy_transform_radians()
            )  #  type: ignore
        transform = self._mpl_transform_radec2xy
        if ax:
            transform = transform + ax.transData
        return transform

    def matplotlib_xy2radec_transform(
        self, ax: Axes | None = None
    ) -> matplotlib.transforms.Transform:
        """
        Get matplotlib transform which converts image pixel coordinates to RA/Dec sky
        coordinates.

        The transform is a mutable object which can be dynamically updated using
        :func:`update_transform` when the `radec` to `xy` coordinate conversion changes.
        This can be useful for plotting data (e.g. an observed image) using image xy
        coordinates onto an axis using RA/Dec coordinates. ::

            # Plot an observed image on an RA/Dec axis with a wireframe of the target
            ax = obs.plot_wireframe_radec()
            ax.autoscale_view()
            ax.autoscale(False) # Prevent imshow breaking autoscale
            ax.imshow(
                img,
                origin='lower',
                transform=obs.matplotlib_xy2radec_transform(ax),
                )

        Args:
            ax: Optionally specify a matplotlib axis to return
                `transform_radec2xy + ax.transData`. This value can then be used in the
                `transform` keyword argument of a Matplotlib function without any
                further modification.

        Returns:
            Matplotlib transformation from `xy` to `radec` coordinates.
        """
        if self._mpl_transform_xy2radec is None:
            self._mpl_transform_xy2radec = (
                self.matplotlib_radec2xy_transform().inverted()
            )
        transform = self._mpl_transform_xy2radec
        if ax:
            transform = transform + ax.transData
        return transform

    def matplotlib_xy2km_transform(
        self, ax: Axes | None
    ) -> matplotlib.transforms.Transform:
        return (
            self.matplotlib_xy2radec_transform()
            + self.matplotlib_radec2km_transform(ax)
        )

    def matplotlib_km2xy_transform(
        self, ax: Axes | None
    ) -> matplotlib.transforms.Transform:
        return (
            self.matplotlib_km2radec_transform()
            + self.matplotlib_radec2xy_transform(ax)
        )

    def update_transform(self) -> None:
        """
        Update the transformation returned by :func:`get_matplotlib_radec2xy_transform`
        to use the latest disc parameter values `(x0, y0, r0, rotation)`.
        """
        self._get_matplotlib_radec2xy_transform_radians().set_matrix(
            self._get_radec2xy_matrix_radians()
        )

    # Plotting
    def plot_wireframe_xy(
        self,
        ax: Axes | None = None,
        show: bool = False,
        color: str | tuple[float, float, float] = 'k',
        label_poles: bool = True,
        add_title: bool = True,
        add_axis_labels: bool = True,
        aspect_adjustable: Literal['box', 'datalim'] = 'box',
        **kwargs,
    ) -> Axes:
        """
        Plot basic wireframe representation of the observation using image pixel
        coordinates.

        Args:
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), uses
                `plt.gca()` to get the currently active axis.
            show: Toggle immediately showing the plotted figure with `plt.show()`.
            color: Matplotlib color used for to plot the wireframe.
            label_poles: Toggle labelling the poles of the target body.
            add_title: Add title generated by :func:`get_description` to the axis.
            add_axis_labels: Add axis labels.
            aspect_adjustable: Set `adjustable` parameter when setting the aspect ratio.
                Passed to :func:`matplotlib.axes.Axes.set_aspect`.
            **kwargs: Additional arguments are passed to Matplotlib plotting functions
                (useful for e.g. specifying `zorder`).

        Returns:
            The axis containing the plotted wireframe.
        """
        # Generate affine transformation from radec in degrees -> xy
        transform = self.matplotlib_radec2xy_transform()
        ax = self._plot_wireframe(
            transform=transform,
            ax=ax,
            color=color,
            label_poles=label_poles,
            add_title=add_title,
            **kwargs,
        )

        if self._test_if_img_size_valid():
            ax.set_xlim(-0.5, self._nx - 0.5)
            ax.set_ylim(-0.5, self._ny - 0.5)
        if add_axis_labels:
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
        ax.set_aspect(1, adjustable=aspect_adjustable)

        if show:
            plt.show()
        return ax

    # Mapping
    def map_img(
        self,
        img: np.ndarray,
        degree_interval: float = 1,
        interpolation: Literal['nearest', 'linear', 'quadratic', 'cubic'] = 'linear',
        propagate_nan: bool = True,
        warn_nan: bool = False,
    ) -> np.ndarray:
        """
        Project an observed image onto a lon/lat grid.

        If `interpolation` is `'linear'`, `'quadratic'` or `'cubic'`, the map projection
        is performed using `scipy.interpolate.RectBivariateSpline` using the specified
        degree of interpolation.

        If `interpolation` is `'nearest'`, no interpolation is performed, and the mapped
        image takes the value of the nearest pixel in the image to that location. This
        can be useful to easily visualise the pixel scale for low spatial resolution
        observations.

        Args:
            img: Observed image where pixel coordinates correspond to the `xy` pixel
                coordinates (e.g. those used in :func:`get_x0`).
            degree_interval: Interval in degrees between the longitude/latitude points
                in the mapped output. Passed to :func:`get_x_map` and :func:`get_y_map`
                when generating the coordinates used for the projection.
            interpolation: Interpolation used when mapping. This can either any of
                `'nearest'`, `'linear'`, `'quadratic'` or `'cubic'`. The default is
                `'linear'`.
            propagate_nan: If using spline interpolation, propagate NaN values from the
                image to the mapped data. If `propagate_nan` is `True` (the default),
                the interpolation is performed as normal (i.e. with NaN values in the
                image set to 0), then any mapped locations where the nearest
                corresponding image pixel is NaN are set to NaN. Note that there may
                still be very small errors on the boundaries of NaN regions caused by
                the interpolation.
            warn_nan: Print warning if any values in `img` are NaN when any of the
                spline interpolations are used.

        Returns:
            Array containing cylindrical map of the values in `img` at each location on
            the surface of the target body. Locations which are not visible have a value
            of NaN.
        """
        x_map = self.get_x_map(degree_interval)
        y_map = self.get_y_map(degree_interval)
        projected = self._make_empty_map(degree_interval)

        spline_k = {
            'linear': 1,
            'quadratic': 2,
            'cubic': 3,
        }

        if interpolation == 'nearest':
            nan_sentinel = -999
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
        elif interpolation in spline_k:
            k = spline_k[interpolation]
            nans = np.isnan(img)
            if np.any(np.isnan(img)):
                if warn_nan:
                    print('Warning, image contains NaN values which will be set to 0')
                img = np.nan_to_num(img)
            interpolator = scipy.interpolate.RectBivariateSpline(
                np.arange(img.shape[0]), np.arange(img.shape[1]), img, kx=k, ky=k
            )
            for a, b in self._iterate_image(projected.shape):
                x = x_map[a, b]
                if math.isnan(x):
                    continue
                y = y_map[a, b]  # y should never be nan when x is not nan
                if propagate_nan and self._should_propagate_nan_to_map(x, y, nans):
                    continue
                projected[a, b] = interpolator(y, x)
        else:
            raise ValueError(f'Unknown interpolation method {interpolation!r}')
        return projected

    def _should_propagate_nan_to_map(
        self, x: float, y: float, nans: np.ndarray
    ) -> bool:
        # Test if any of the four surrounding integer pixels in the image are NaN
        return (
            nans[math.floor(y), math.floor(x)]
            or nans[math.floor(y), math.ceil(x)]
            or nans[math.ceil(y), math.floor(x)]
            or nans[math.ceil(y), math.ceil(x)]
        )

    def _xy_in_image_frame(self, x: float, y: float) -> bool:
        return (-0.5 < x < self._nx - 0.5) and (-0.5 < y < self._ny - 0.5)

    # Backplane management
    @staticmethod
    def standardise_backplane_name(name: str) -> str:
        """
        Create a standardised version of a backplane name when finding and registering
        backplanes.

        This standardisation is used in functions like :func:`get_backplane_img` and
        :func:`plot_backplane` so that, for example `body.plot_backplane('LAT')`,
        `body.plot_backplane('Lat')` and `body.plot_backplane('lat')` all produce the
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
        # TODO add checks for name/description lengths?
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
        except:
            raise BackplaneNotFoundError(
                '{n!r} not found. Currently registered backplanes are: {r}.'.format(
                    n=name,
                    r=', '.join([repr(n) for n in self.backplanes.keys()]),
                )
            )

    def get_backplane_img(self, name: str) -> np.ndarray:
        """
        Generate backplane image.

        Note that a generated backplane image will depend on the disc parameters
        `(x0, y0, r0, rotation)` at the time this function is called. Generating the
        same backplane when there are different disc parameter values will produce a
        different image. This method creates a copy of the generated image, so the
        returned image can be safely modified without affecting the cached value (unlike
        the return values from functions such as :func:`get_lon_img`).

        This method is equivalent to ::

            body.get_backplane(name).get_img().copy()

        Args:
            name: Name of the desired backplane. This is standardised with
                :func:`standardise_backplane_name` and used to choose a registered
                backplane from :attr:`backplanes`.

        Returns:
            Array containing the backplane's values for each pixel in the image.
        """
        return self.backplanes[self.standardise_backplane_name(name)].get_img().copy()

    def get_backplane_map(self, name: str, degree_interval: float = 1) -> np.ndarray:
        """
        Generate map of backplane values.

        This method creates a copy of the generated image, so the returned map can be
        safely modified without affecting the cached value (unlike the return values
        from functions such as :func:`get_lon_map`).

        This method is equivalent to ::

            body.get_backplane(name).get_map(degree_interval).copy()

        Args:
            name: Name of the desired backplane. This is standardised with
                :func:`standardise_backplane_name` and used to choose a registered
                backplane from :attr:`backplanes`.
            degree_interval: Interval in degrees between the longitude/latitude points
                in the mapped output.

        Returns:
            Array containing map of the backplane's values over the surface of the
            target body.
        """
        return (
            self.backplanes[self.standardise_backplane_name(name)]
            .get_map(degree_interval)
            .copy()
        )

    def plot_backplane_img(
        self, name: str, ax: Axes | None = None, show: bool = False, **kwargs
    ) -> Axes:
        """
        Plot a backplane image with the wireframe outline of the target.

        Note that a generated backplane image will depend on the disc parameters
        `(x0, y0, r0, rotation)` at the time this function is called. Generating the
        same backplane when there are different disc parameter values will produce a
        different image.

        Args:
            name: Name of the desired backplane.
            ax: Passed to :func:`plot_wireframe_xy`.
            show: Passed to :func:`plot_wireframe_xy`.
            **kwargs: Passed to Matplotlib's `imshow` when plotting the backplane image.
                For example, can be used to set the colormap of the plot using
                `body.plot_backplane_img(..., cmap='Greys')`.

        Returns:
            The axis containing the plotted data.
        """
        backplane = self.get_backplane(name)
        ax = self.plot_wireframe_xy(ax, show=False)
        im = ax.imshow(backplane.get_img(), origin='lower', **kwargs)
        plt.colorbar(im, label=backplane.description)
        if show:
            plt.show()
        return ax

    def plot_backplane_map(
        self,
        name: str,
        ax: Axes | None = None,
        show: bool = False,
        degree_interval: float = 1,
        **kwargs,
    ) -> Axes:
        """
        Plot a map of backplane values on the target body.

        Args:
            name: Name of the desired backplane.
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), then
                a new figure and axis is created.
            show: Toggle showing the plotted figure with `plt.show()`
            degree_interval: Interval in degrees between the longitude/latitude points
                in the mapped output.
            **kwargs: Passed to Matplotlib's `imshow` when plotting the backplane map.
                For example, can be used to set the colormap of the plot using
                `body.plot_backplane_map(..., cmap='Greys')`.

        Returns:
            The axis containing the plotted data.
        """
        if ax is None:
            fig, ax = plt.subplots()
        backplane = self.get_backplane(name)

        im = self.imshow_map(backplane.get_map(degree_interval), ax=ax)
        plt.colorbar(im, label=backplane.description)
        ax.set_title(self.get_description(multiline=True))
        if show:
            plt.show()
        return ax

    def imshow_map(
        self, map_img: np.ndarray, ax: Axes | None = None, **kwargs
    ) -> AxesImage:
        """
        Utility function to easily plot a mapped image using `plt.imshow` with
        appropriate extents, axis labels etc.

        This is equivalent to calling `ax.imshow(map_img, origin='lower', ...)` with
        additional code to format the axis nicely.

        Args:
            map_img: Image to plot.
            ax: Matplotlib axis to use for plotting. If `ax` is None (the default), then
                a new figure and axis is created.
            **kwargs: Passed to Matplotlib's `imshow` when plotting the backplane map.
                For example, can be used to set the colormap of the plot using
                `body.plot_backplane_map(..., cmap='Greys')`.

        Returns:
            Handle of plotted Matplotlib `AxesImage`.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if self.positive_longitude_direction == 'W':
            extent = [360, 0, -90, 90]
        else:
            extent = [0, 360, -90, 90]

        im = ax.imshow(map_img, origin='lower', extent=extent, **kwargs)

        ax.set_aspect(1, adjustable='box')
        ax.set_xlabel(f'Planetographic longitude ({self.positive_longitude_direction})')
        ax.set_ylabel('Planetographic latitude')

        step = 45
        xt = np.arange(0, 360.1, step)
        ax.set_xticks(xt)
        ax.set_xticklabels([f'{x:.0f}°' for x in xt])

        yt = np.arange(-90, 90.1, step)
        ax.set_yticks(yt)
        ax.set_yticklabels([f'{y:.0f}°' for y in yt])
        return im

    # Backplane generatotrs
    def _test_if_img_size_valid(self) -> bool:
        return (self._nx > 0) and (self._ny > 0)

    def _iterate_image(
        self, shape: tuple[int, ...], progress: bool = False
    ) -> Iterable[tuple[int, int]]:
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

    def _make_map_lonlat_arrays(
        self, degree_interval: float
    ) -> tuple[np.ndarray, np.ndarray]:
        lons = np.arange(degree_interval / 2, 360, degree_interval)
        if self.positive_longitude_direction == 'W':
            lons = lons[::-1]
        lats = np.arange(-90 + degree_interval / 2, 90, degree_interval)
        return lons, lats

    def _make_empty_map(
        self, degree_interval: float, nz: int | None = None
    ) -> np.ndarray:
        # making arrays is slightly more costly, but ensures that we don't get any
        # numerical stability errors from trying to do e.g. 360//degree_interval
        lons, lats = self._make_map_lonlat_arrays(degree_interval)
        nlon = len(lons)
        nlat = len(lats)
        if nz is None:
            shape = (nlat, nlon)
        else:
            shape = (nlat, nlon, nz)
        return np.full(shape, np.nan)

    def _get_max_pixel_radius(self) -> float:
        # r0 corresponds to r_eq, but for the radius here we want to make sure we have
        # the largest radius
        r = self.get_r0() * max(self.radii) / self.r_eq
        return r

    @_cache_clearable_result
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
    def _get_targvec_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 3)
        lons, lats = self._make_map_lonlat_arrays(degree_interval)
        for a, lat in enumerate(lats):
            self._update_progress_hook(a / len(lats))
            for b, lon in enumerate(lons):
                out[a, b] = self.lonlat2targvec(lon, lat)
        return out

    def _enumerate_targvec_img(
        self, progress: bool = False
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        targvec_img = self._get_targvec_img()
        for y, x in self._iterate_image(targvec_img.shape, progress=progress):
            targvec = targvec_img[y, x]
            if math.isnan(targvec[0]):
                # only check if first element nan for efficiency
                continue
            yield y, x, targvec

    def _enumerate_targvec_map(
        self, degree_interval: float, progress: bool = False
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        targvec_map = self._get_targvec_map(degree_interval)
        for a, b in self._iterate_image(targvec_map.shape, progress=progress):
            yield a, b, targvec_map[a, b]

    @_cache_clearable_result
    @progress_decorator
    def _get_lonlat_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._targvec2lonlat_radians(targvec)
        return np.rad2deg(out)

    @_cache_stable_result
    @progress_decorator
    def _get_lonlat_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 2)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            out[a, b] = self._targvec2lonlat_radians(targvec)
        return np.rad2deg(out)

    def get_lon_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the planetographic longitude value of each pixel in the
            image. Points off the disc have a value of NaN.
        """
        return self._get_lonlat_img()[:, :, 0]

    def get_lon_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of planetographic longitude values.
        """
        return self._get_lonlat_map(degree_interval)[:, :, 0]

    def get_lat_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the planetographic latitude value of each pixel in the
            image. Points off the disc have a value of NaN.
        """
        return self._get_lonlat_img()[:, :, 1]

    def get_lat_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of planetographic latitude values.
        """
        return self._get_lonlat_map(degree_interval)[:, :, 1]

    @_cache_clearable_result
    @progress_decorator
    def _get_lonlat_centric_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._targvec2lonlat_centric(targvec)
        return out

    @_cache_stable_result
    @progress_decorator
    def _get_lonlat_centric_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 2)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            out[a, b] = self._targvec2lonlat_centric(targvec)
        return out

    def get_lon_centric_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the planetocentric longitude value of each pixel in the
            image. Points off the disc have a value of NaN.
        """
        return self._get_lonlat_centric_img()[:, :, 0]

    def get_lon_centric_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of planetocentric longitude values.
        """
        return self._get_lonlat_centric_map(degree_interval)[:, :, 0]

    def get_lat_centric_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the planetocentric latitude value of each pixel in the
            image. Points off the disc have a value of NaN.
        """
        return self._get_lonlat_centric_img()[:, :, 1]

    def get_lat_centric_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of planetocentric latitude values.
        """
        return self._get_lonlat_centric_map(degree_interval)[:, :, 1]

    @_cache_clearable_result
    @progress_decorator
    def _get_radec_img(self) -> np.ndarray:
        out = self._make_empty_img(2)
        for y, x in self._iterate_image(out.shape, progress=True):
            out[y, x] = self._xy2radec_radians(x, y)
        return np.rad2deg(out)

    @_cache_stable_result
    @progress_decorator
    def _get_radec_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 2)
        visible = self._get_illumf_map(degree_interval)[:, :, 4]
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            if visible[a, b]:
                out[a, b] = self._obsvec2radec_radians(self._targvec2obsvec(targvec))
        return np.rad2deg(out)

    def get_ra_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the right ascension (RA) value of each pixel in the image.
        """
        return self._get_radec_img()[:, :, 0]

    def get_ra_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of right ascension values as viewed by the
            observer. Locations which are not visible have a value of NaN.
        """
        return self._get_radec_map(degree_interval)[:, :, 0]

    def get_dec_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the declination (Dec) value of each pixel in the image.
        """
        return self._get_radec_img()[:, :, 1]

    def get_dec_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of declination values as viewed by the
            observer. Locations which are not visible have a value of NaN.
        """
        return self._get_radec_map(degree_interval)[:, :, 1]

    @_cache_clearable_result_with_args
    @progress_decorator
    def _get_xy_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 2)
        radec_map = self._get_radec_map(degree_interval)
        for a, b in self._iterate_image(out.shape, progress=True):
            ra, dec = radec_map[a, b]
            if not math.isnan(ra):
                x, y = self.radec2xy(ra, dec)
                if self._xy_in_image_frame(x, y):
                    out[a, b] = x, y
        return out

    def get_x_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the x pixel coordinate value of each pixel in the image.
        """
        out = self._make_empty_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = x
        return out

    def get_x_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the x pixel coordinates each location
            corresponds to in the observation. Locations which are not visible or are
            not in the image frame have a value of NaN.
        """
        return self._get_xy_map(degree_interval)[:, :, 0]

    def get_y_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the y pixel coordinate value of each pixel in the image.
        """
        out = self._make_empty_img()
        for y, x in self._iterate_image(out.shape):
            out[y, x] = y
        return out

    def get_y_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the y pixel coordinates each location
            corresponds to in the observation. Locations which are not visible or are
            not in the image frame have a value of NaN.
        """
        return self._get_xy_map(degree_interval)[:, :, 1]

    @_cache_clearable_result
    @progress_decorator
    def _get_illumination_gie_img(self) -> np.ndarray:
        out = self._make_empty_img(3)
        for y, x, targvec in self._enumerate_targvec_img(progress=True):
            out[y, x] = self._illumination_angles_from_targvec_radians(targvec)
        return np.rad2deg(out)

    @_cache_stable_result
    @progress_decorator
    def _get_illumf_map(self, degree_interval: float) -> np.ndarray:
        out = self._make_empty_map(degree_interval, 5)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            out[a, b] = self._illumf_from_targvec_radians(targvec)
        return np.rad2deg(out)

    def get_phase_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the phase angle value of each pixel in the image. Points
            off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 0]

    def get_phase_angle_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the phase angle value at each point on
            the target's surface.
        """
        return self._get_illumf_map(degree_interval)[:, :, 0]

    def get_incidence_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the incidence angle value of each pixel in the image.
            Points off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 1]

    def get_incidence_angle_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the incidence angle value at each point
            on the target's surface.
        """
        return self._get_illumf_map(degree_interval)[:, :, 1]

    def get_emission_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the emission angle value of each pixel in the image. Points
            off the disc have a value of NaN.
        """
        return self._get_illumination_gie_img()[:, :, 2]

    def get_emission_angle_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the emission angle value at each point
            on the target's surface.
        """
        return self._get_illumf_map(degree_interval)[:, :, 2]

    @_cache_clearable_result
    def get_azimuth_angle_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the azimuth angle value of each pixel in the image. Points
            off the disc have a value of NaN.
        """
        phase_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 0])
        incidence_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 1])
        emission_radians = np.deg2rad(self._get_illumination_gie_img()[:, :, 2])
        with warnings.catch_warnings():
            # TODO should we do this better?
            warnings.filterwarnings('ignore', 'divide by zero encountered in')
            warnings.filterwarnings('ignore', 'invalid value encountered in')
            azimuth_radians = self._azimuth_angle_from_gie_radians(
                phase_radians, incidence_radians, emission_radians
            )
        return np.rad2deg(azimuth_radians)

    @_cache_stable_result
    def get_azimuth_angle_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the azimuth angle value at each point
            on the target's surface.
        """
        phase_radians = np.deg2rad(self._get_illumf_map(degree_interval)[:, :, 0])
        incidence_radians = np.deg2rad(self._get_illumf_map(degree_interval)[:, :, 1])
        emission_radians = np.deg2rad(self._get_illumf_map(degree_interval)[:, :, 2])
        with warnings.catch_warnings():
            # TODO should we do this better?
            warnings.filterwarnings('ignore', 'divide by zero encountered in')
            warnings.filterwarnings('ignore', 'invalid value encountered in')
            azimuth_radians = self._azimuth_angle_from_gie_radians(
                phase_radians, incidence_radians, emission_radians
            )
        return np.rad2deg(azimuth_radians)

    @_cache_clearable_result
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
        return position_img, velocity_img, lt_img

    @_cache_stable_result
    @progress_decorator
    def _get_state_maps(
        self, degree_interval: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_map = self._make_empty_map(degree_interval, 3)
        velocity_map = self._make_empty_map(degree_interval, 3)
        lt_map = self._make_empty_map(degree_interval)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            (
                position_map[a, b],
                velocity_map[a, b],
                lt_map[a, b],
            ) = self._state_from_targvec(targvec)
        return position_map, velocity_map, lt_map

    def get_distance_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the observer-target distance in km of each pixel in the
            image. Points off the disc have a value of NaN.
        """
        position_img, velocity_img, lt_img = self._get_state_imgs()
        return lt_img * self.speed_of_light()

    def get_distance_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the observer-target distance in km of
            each point on the target's surface.
        """
        position_map, velocity_map, lt_map = self._get_state_maps(degree_interval)
        return lt_map * self.speed_of_light()

    @_cache_clearable_result
    @progress_decorator
    def get_radial_velocity_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the observer-target radial velocity in km/s of each pixel
            in the image. Velocities towards the observer are negative. Points off the
            disc have a value of NaN.
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
    def get_radial_velocity_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the observer-target radial velocity in
            km/s of each point on the target's surface.
        """
        out = self._make_empty_map(degree_interval)
        position_map, velocity_map, lt_map = self._get_state_maps(degree_interval)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            out[a, b] = self._radial_velocity_from_state(
                position_map[a, b], velocity_map[a, b]
            )
        return out

    def get_doppler_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the doppler factor for each pixel in the image, calculated
            using :func:`SpiceBase.calculate_doppler_factor` on velocities from
            :func:`get_radial_velocity_img`. Points off the disc have a value of NaN.
        """
        return self.calculate_doppler_factor(self.get_radial_velocity_img())

    def get_doppler_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the doppler factor of each point on the
            target's surface. This is calculated using
            :func:`SpiceBase.calculate_doppler_factor` on velocities from
            :func:`get_radial_velocity_map`.
        """
        return self.calculate_doppler_factor(
            self.get_radial_velocity_map(degree_interval)
        )

    @_cache_clearable_result
    @progress_decorator
    def _get_ring_plane_coordinate_imgs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_img = self._make_empty_img()
        long_img = self._make_empty_img()
        dist_img = self._make_empty_img()

        ra_img = self.get_ra_img()
        dec_img = self.get_dec_img()
        for y, x in self._iterate_image(radius_img.shape, progress=True):
            radius, long, dist = self.ring_plane_coordinates(
                ra_img[y, x], dec_img[y, x], only_visible=False
            )
            radius_img[y, x] = radius
            long_img[y, x] = long
            dist_img[y, x] = dist

        hidden_img = dist_img > self.get_distance_img()
        radius_img[hidden_img] = np.nan
        long_img[hidden_img] = np.nan
        dist_img[hidden_img] = np.nan
        return radius_img, long_img, dist_img

    @_cache_stable_result
    @progress_decorator
    def _get_ring_plane_coordinate_maps(
        self, degree_interval: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_map = self._make_empty_map(degree_interval)
        long_map = self._make_empty_map(degree_interval)
        dist_map = self._make_empty_map(degree_interval)

        visible = self._get_illumf_map(degree_interval)[:, :, 4]
        ra_map = self.get_ra_map(degree_interval)
        dec_map = self.get_dec_map(degree_interval)
        for a, b, targvec in self._enumerate_targvec_map(
            degree_interval, progress=True
        ):
            if visible[a, b]:
                radius, long, dist = self.ring_plane_coordinates(
                    ra_map[a, b], dec_map[a, b], only_visible=False
                )
                radius_map[a, b] = radius
                long_map[a, b] = long
                dist_map[a, b] = dist

        hidden_map = dist_map > self.get_distance_map(degree_interval)
        radius_map[hidden_map] = np.nan
        long_map[hidden_map] = np.nan
        dist_map[hidden_map] = np.nan
        return radius_map, long_map, dist_map

    def get_ring_plane_radius_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the ring plane radius in km for each pixel in the image,
            calculated using :func:`Body.ring_plane_coordinates`. Points of the ring
            plane obscured by the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[0]

    def get_ring_plane_radius_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the ring plane radius in km obscuring
            each point on the target's surface, calculated using
            :func:`Body.ring_plane_coordinates`. Points where the target body is
            unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(degree_interval)[0]

    def get_ring_plane_longitude_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the ring plane planetographic longitude in degrees for each
            pixel in the image, calculated using :func:`Body.ring_plane_coordinates`.
            Points of the ring plane obscured by the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[1]

    def get_ring_plane_longitude_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the ring plane planetographic longitude
            in degrees obscuring each point on the target's surface, calculated using
            :func:`Body.ring_plane_coordinates`. Points where the target body is
            unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(degree_interval)[1]

    def get_ring_plane_distance_img(self) -> np.ndarray:
        """
        See also :func:`get_backplane_img`.

        Returns:
            Array containing the ring plane distance from the observer in km for each
            pixel in the image, calculated using :func:`Body.ring_plane_coordinates`.
            Points of the ring plane obscured by the target body have a value of NaN.
        """
        return self._get_ring_plane_coordinate_imgs()[2]

    def get_ring_plane_distance_map(self, degree_interval: float = 1) -> np.ndarray:
        """
        See also :func:`get_backplane_map`.

        Args:
            degree_interval: Interval in degrees between points in the returned map.

        Returns:
            Array containing cylindrical map of the ring plane distance from the
            observer in km obscuring each point on the target's surface, calculated
            using  :func:`Body.ring_plane_coordinates`. Points where the target body is
            unobscured by the ring plane have a value of NaN.
        """
        return self._get_ring_plane_coordinate_maps(degree_interval)[2]

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
    return '\n'.join(msg)
