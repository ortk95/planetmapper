import datetime
import functools
import glob
import numbers
import os
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, cast

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import astropy.time
import numpy as np
import spiceypy as spice

from . import progress

DEFAULT_KERNEL_PATH = '~/spice_kernels/'

_KERNEL_DATA = {
    'kernel_path': None,
    'kernel_patterns': ('**/*.bsp', '**/*.tpc', '**/*.tls'),
    'kernels_loaded': False,
}

Numeric = TypeVar('Numeric', bound=float | np.ndarray)


T = TypeVar('T')
S = TypeVar('S')
P = ParamSpec('P')


def _cache_clearable_result(
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

    Note that any numpy arguments will be converted to (nested) tuples.
    """
    # pylint: disable=protected-access

    @functools.wraps(fn)
    def decorated(self, *args_in: P.args, **kwargs_in: P.kwargs):
        args, kwargs = _replace_np_arrr_args_with_tuples(args_in, kwargs_in)
        k = (fn.__name__, args, frozenset(kwargs.items()))
        if k not in self._cache:
            self._cache[k] = fn(self, *args, **kwargs)  # type: ignore
        return self._cache[k]

    return decorated


def _cache_stable_result(
    fn: Callable[Concatenate[S, P], T]
) -> Callable[Concatenate[S, P], T]:
    """
    Decorator to cache stable result

    Very roughly, this is a type-hinted version of `functools.lru_cache` that doesn't
    cache self.

    See _cache_clearable_result for more details.
    """

    # pylint: disable=protected-access
    @functools.wraps(fn)
    def decorated(self, *args_in: P.args, **kwargs_in: P.kwargs):
        args, kwargs = _replace_np_arrr_args_with_tuples(args_in, kwargs_in)
        k = (fn.__name__, args, frozenset(kwargs.items()))
        if k not in self._stable_cache:
            self._stable_cache[k] = fn(self, *args, **kwargs)  # type: ignore
        return self._stable_cache[k]

    return decorated


def _replace_np_arrr_args_with_tuples(args, kwargs) -> tuple[tuple, dict[str, Any]]:
    args = tuple(_maybe_np_arr_to_tuple(a) for a in args)
    kwargs = {k: _maybe_np_arr_to_tuple(v) for k, v in kwargs.items()}
    return args, kwargs


def _maybe_np_arr_to_tuple(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return _to_tuple(o)
    return o


def _to_tuple(arr: np.ndarray):
    if arr.ndim > 1:
        return tuple(_to_tuple(a) for a in arr)
    elif arr.ndim == 1:
        return tuple(arr)
    elif arr.ndim == 0:
        return float(arr)
    else:
        raise ValueError(f'Error converting arr {arr!r} to tuple')


class SpiceBase:
    """
    Class containing methods to interface with spice and manipulate coordinates.

    This is the base class for all the main classes used in planetmapper.

    Args:
        show_progress: Show progress bars for long running processes. This is mainly
            useful for complex functions in derived classes, such as backplane
            generation in :class:`BodyXY`. These progress bars can be quite messy, but
            can be useful to keep track of very long operations.
        optimize_speed: Toggle speed optimizations. For typical observations, the
            optimizations can make code significantly faster with no effect on accuracy,
            so should generally be left enabled.
        auto_load_kernels: Toggle automatic kernel loading with
            :func:`load_spice_kernels`.
        kernel_path: Passed to  :func:`load_spice_kernels` if `load_kernels` is True.
        manual_kernels: Passed to  :func:`load_spice_kernels` if `load_kernels` is True.
    """

    _DEFAULT_DTM_FORMAT_STRING = '%Y-%m-%dT%H:%M:%S.%f'

    def __init__(
        self,
        show_progress: bool = False,
        optimize_speed: bool = True,
        auto_load_kernels: bool = True,
        kernel_path: str | None = None,
        manual_kernels: None | list[str] = None,
    ) -> None:
        super().__init__()
        self._optimize_speed = optimize_speed

        self._cache = {}
        self._stable_cache = {}

        self._progress_hook: progress.ProgressHook | None = None
        self._progress_call_stack: list[str] = []

        if show_progress:
            self._set_progress_hook(progress.CLIProgressHook())

        if auto_load_kernels:
            self.load_spice_kernels(
                kernel_path=kernel_path, manual_kernels=manual_kernels
            )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SpiceBase)
            and self._get_equality_tuple() == other._get_equality_tuple()
        )

    def __hash__(self) -> int:
        return hash(self._get_equality_tuple())

    def _get_equality_tuple(self) -> tuple:
        """
        Tuple containing all the information needed to determine object equality.

        Subclasses should override this to include any additional information needed to
        determine equality e.g.

            return (self.a, self.b, super()._get_equality_tuple())

        Used by __eq__ and __hash__.
        """
        return (self._optimize_speed, repr(self))

    def _get_kwargs(self) -> dict[str, Any]:
        """
        Get kwargs used to __init__ a new object of this class.

        This is used by `copy` to copy the options of this object to a new object in
        conjunction with `_copy_options_to_other`.

        Subclasses should override this to include any additional information needed to
        build a new object e.g.

            return super()._get_kwargs() | dict(a=self.a, b=self.b)
        """
        return dict(optimize_speed=self._optimize_speed)

    def _copy_options_to_other(self, other: Self) -> None:
        """
        Copy customisable options, attributes etc. to another object.

        This is used by e.g. `copy` to copy the options of this object to a new
        object in conjunction with `_get_kwargs`.

        Subclasses should override this to include any additional information needed to
        build a new object e.g.

            super()._copy_options_to_other(other)
            other.c = self.c.copy()
        """

    def __copy__(self) -> Self:
        new = self.__class__(**self._get_kwargs())
        self._copy_options_to_other(new)
        return new

    def copy(self) -> Self:
        """
        Return a copy of this object.
        """
        return self.__copy__()

    def _clear_cache(self):
        """
        Clear cached results from `_cache_result`.
        """
        # TODO document cache clearing (incl stable cache)
        self._cache.clear()

    def standardise_body_name(self, name: str | int) -> str:
        """
        Return a standardised version of the name of a SPICE body.

        This converts the provided `name` into the SPICE ID code with `spice.bods2c`,
        then back into a string with `spice.bodc2s`. This standardises to the version of
        the name preferred by SPICE. For example, `'jupiter'`, `'JuPiTeR'`,
        `' Jupiter '`, `'599'` and `599` are all standardised to `'JUPITER'`

        Args:
            name: The name of a body (e.g. a planet). This can also be the numeric ID
                code of a body.

        Returns:
            Standardised version of the body's name preferred by SPICE.

        Raises:
            NotFoundError: If SPICE does not recognise the provided `name`
        """
        name = spice.bodc2s(spice.bods2c(str(name)))
        return name

    def et2dtm(self, et: float) -> datetime.datetime:
        """
        Convert ephemeris time to a Python datetime object.

        Args:
            et: Ephemeris time in seconds past J2000.

        Returns:
            Timezone aware (UTC) datetime corresponding to `et`.
        """
        s = spice.et2utc(et, 'ISOC', 6) + '+0000'
        # manually add '+0000' to string to make it timezone aware
        # i.e. this lets python know it is UTC
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')

    @staticmethod
    def mjd2dtm(mjd: float) -> datetime.datetime:
        """
        Convert Modified Julian Date into a python datetime object.

        Args:
            mjd: Float representing MJD.

        Returns:
            Python datetime object corresponding to `mjd`. This datetime is timezone
            aware and set to the UTC timezone.
        """
        dtm: datetime.datetime = astropy.time.Time(mjd, format='mjd').datetime
        return dtm.replace(tzinfo=datetime.timezone.utc)

    def speed_of_light(self) -> float:
        """
        Return the speed of light in km/s. This is a convenience function to call
        `spice.clight()`.

        Returns:
            Speed of light in km/s.
        """
        return spice.clight()

    def calculate_doppler_factor(self, radial_velocity: Numeric) -> Numeric:
        """
        Calculates the doppler factor caused by a target's radial velocity relative to
        the observer. This doppler factor, :math:`D` can be used to calculate the
        doppler shift caused by this velocity as :math:`\\lambda_r = \\lambda_e D`
        where :math:`\\lambda_r` is the wavelength received by the observer and
        :math:`\\lambda_e` is the wavelength emitted at the target.

        This doppler factor is calculated as
        :math:`D = \\sqrt{\\frac{1 + v/c}{1 - v/c}}` where :math:`v` is the input
        `radial_velocity` and :math:`c` is the speed of light.

        See also
        https://en.wikipedia.org/wiki/Relativistic_Doppler_effect#Relativistic_longitudinal_Doppler_effect

        Args:
            radial_velocity: Radial velocity in km/s with positive values corresponding
                to motion away from the observer. This can be a single float value or a
                numpy array containing multiple velocity values.

        Returns:
            Doppler factor calculated from input radial velocity. If the input
            `radial_velocity` is a single value, then a `float` is returned. If the
            input `radial_velocity` is a numpy array, then a numpy array of doppler
            factors is returned.
        """
        beta = radial_velocity / self.speed_of_light()
        return np.sqrt((1 + beta) / (1 - beta))  #  type: ignore

    @staticmethod
    def load_spice_kernels(
        kernel_path: str | None = None,
        manual_kernels: None | list[str] = None,
        only_if_needed: bool = True,
    ) -> None:
        """
        Attempt to intelligently SPICE kernels using `spice.furnsh`.

        If `manual_kernels` is `None` (the default), then all kernels in the directory
        given by `kernel_path` which match the following patterns are loaded:

        - `**/*.bsp`
        - `**/*.tpc`
        - `**/*.tls`

        Note that these patterns match an arbitrary number of nested directories (within
        `kernel_path`). If more control is required, you can instead specify a list of
        specific kernels to load with `manual_kernels`.

        .. hint::
            See the :ref:`SPICE kernel documentation <SPICE kernels>` for more detail
            about downloading SPICE kernels and the automatic kernel loading behaviour.

        Args:
            kernel_path: Path to directory where kernels are stored. If this is `None`
                (the default) then the result of :func:`get_kernel_path` is used. It is
                usually recommended to use one of the methods described in
                :ref:`the kernel directory documentation<kernel directory>` rather than
                using this `kernel_path` argument.
            manual_kernels: Optional manual list of paths to kernels to load instead of
                using `kernel_path`.
            only_if_needed: If this is `True`, kernels will only be loaded once per
                session.
        """
        if only_if_needed and _KERNEL_DATA['kernels_loaded']:
            return
        if manual_kernels:
            kernels = manual_kernels
        else:
            if kernel_path is None:
                kernel_path = get_kernel_path()
            kernel_path = os.path.expanduser(kernel_path)
            kernels = [
                os.path.join(kernel_path, pattern)
                for pattern in _KERNEL_DATA['kernel_patterns']
            ]

        kernel_paths = load_kernels(*kernels)
        if len(kernel_paths) == 0:
            print()
            print(f'WARNING: no SPICE kernels found in directory {kernel_path!r}')
            print(
                'Try running planetmapper.set_kernel_path to change where PlanetMapper looks for kernels'
            )
            print()
        else:
            _KERNEL_DATA['kernels_loaded'] = True

    @staticmethod
    def close_loop(arr: np.ndarray) -> np.ndarray:
        """
        Return copy of array with first element appended to the end.

        This is useful for cases like plotting the limb of a planet where the array of
        values forms a loop with the first and last values in `arr` adjacent to each
        other.

        Args:
            arr: Array of values of length :math:`n`.

        Returns:
            Array of values of length :math:`n + 1` where the final value is the same as
            the first value.
        """
        return np.append(arr, [arr[0]], axis=0)

    @staticmethod
    def unit_vector(v: np.ndarray) -> np.ndarray:
        """
        Return normalised copy of a vector.

        For an input vector :math:`\\vec{v}`, return the unit vector
        :math:`\\hat{v} = \\frac{\\vec{v}}{|\\vec{v}|}`.

        Args:
            v: Input vector to normalise.

        Returns:
            Normalised vector which is parallel to `v` and has a magnitude of 1.
        """
        # Profiling suggests this is the quickest method
        return v / (sum(v * v)) ** 0.5

    @staticmethod
    def vector_magnitude(v: np.ndarray) -> float:
        """
        Return the magnitude of a vector.

        For an input vector :math:`\\vec{v}`, return magnitude
        :math:`|\\vec{v}| = \\sqrt{\\sum{v_i^2}}`.

        Args:
            v: Input vector.

        Returns:
            Magnitude (length) of vector.
        """
        # Profiling suggests this is the quickest method
        return (sum(v * v)) ** 0.5

    def _encode_str(self, s: str) -> bytes | str:
        if self._optimize_speed:
            return s.encode('UTF-8')
        else:
            return s

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
    def _rotation_matrix_radians(theta: float) -> np.ndarray:
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    @staticmethod
    def angular_dist(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """
        Calculate the angular distance between two RA/Dec coordinate pairs.

        Args:
            ra1: RA of first point.
            dec1: Dec of first point.
            ra2: RA of second point
            dec2: Dec of second point.

        Returns:
            Angular distance in degrees between the two points.
        """
        return np.rad2deg(
            np.arccos(
                np.sin(np.deg2rad(dec1)) * np.sin(np.deg2rad(dec2))
                + np.cos(np.deg2rad(dec1))
                * np.cos(np.deg2rad(dec2))
                * np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))
            )
        )

    def _set_progress_hook(self, progress_hook: progress.ProgressHook) -> None:
        self._progress_hook = progress_hook
        self._progress_call_stack = []

    def _get_progress_hook(self) -> progress.ProgressHook | None:
        return self._progress_hook

    def _remove_progress_hook(self) -> None:
        self._progress_hook = None
        self._progress_call_stack = []

    def _update_progress_hook(self, progress_frac: float) -> None:
        """Update progress hook with progress of current function between 0 & 1"""
        if self._progress_hook is not None:
            self._progress_hook(progress_frac, self._progress_call_stack)


class BodyBase(SpiceBase):
    """
    Base class for :class:`planetmapper.Body` and :class:`planetmapper.BasicBody`.

    You are unlikely to need to use this class directly - use :class:`planetmapper.Body`
    or :class:`planetmapper.BasicBody` instead.
    """

    def __init__(
        self,
        *,
        target: str | int,
        utc: str | datetime.datetime | float | None,
        observer: str | int,
        aberration_correction: str,
        observer_frame: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Process inputs
        if isinstance(utc, (float, int, numbers.Number)):
            # Include a check for numbers.Number to allow other numeric types
            utc = self.mjd2dtm(utc)
        if utc is None:
            utc = datetime.datetime.now(datetime.timezone.utc)
        if isinstance(utc, datetime.datetime):
            # convert input datetime to UTC, then to a string compatible with spice
            if utc.tzinfo is None:
                # Default to UTC if no timezone is specified
                utc = utc.replace(tzinfo=datetime.timezone.utc)
            # Standardise to UTC
            utc = utc.astimezone(tz=datetime.timezone.utc)
            utc = utc.strftime(self._DEFAULT_DTM_FORMAT_STRING)

        self.target = self.standardise_body_name(target)
        self.observer = self.standardise_body_name(observer)
        self.observer_frame = observer_frame
        self.aberration_correction = aberration_correction

        self.et = spice.utc2et(utc)
        self.dtm: datetime.datetime = self.et2dtm(self.et)
        self.utc = self.dtm.strftime(self._DEFAULT_DTM_FORMAT_STRING)
        self.target_body_id: int = spice.bodn2c(self.target)

        # Encode strings which are regularly passed to spice (for speed)
        self._target_encoded = self._encode_str(self.target)
        self._observer_encoded = self._encode_str(self.observer)
        self._observer_frame_encoded = self._encode_str(self.observer_frame)
        self._aberration_correction_encoded = self._encode_str(
            self.aberration_correction
        )

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

    def __repr__(self) -> str:
        return f'BodyBase({self.target!r}, {self.utc!r})'

    def _get_equality_tuple(self) -> tuple:
        return (
            self.target,
            self.utc,
            self.observer,
            self.observer_frame,
            self.aberration_correction,
            super()._get_equality_tuple(),
        )

    def _get_kwargs(self) -> dict[str, Any]:
        return super()._get_kwargs() | dict(
            target=self.target,
            utc=self.utc,
            observer=self.observer,
            aberration_correction=self.aberration_correction,
            observer_frame=self.observer_frame,
        )

    def _obsvec2radec_radians(self, obsvec: np.ndarray) -> tuple[float, float]:
        """
        Transform rectangular vector in observer frame to observer ra/dec coordinates.
        """
        _, ra, dec = spice.recrad(obsvec)
        return ra, dec


def load_kernels(*paths, clear_before: bool = False) -> list[str]:
    """
    Load spice kernels defined by patterns

    Args:
        *paths: Paths to spice kernels, evaluated using `glob.glob` with
            `recursive=True`.
        clear_before: Clear kernel pool before loading new kernels.
    """
    if clear_before:
        spice.kclear()
    kernels = set()
    for pattern in paths:
        kernels.update(glob.glob(os.path.expanduser(pattern), recursive=True))
    for kernel in sorted(kernels):
        spice.furnsh(kernel)
    return list(kernels)


def _clear_kernels() -> None:
    """
    Clear spice kernel pool.

    This function calls `spice.kclear()`, and also indicates to PlanetMapper that
    kernels will need to be reloaded when a new object is created.
    """
    spice.kclear()
    _KERNEL_DATA['kernels_loaded'] = False


def set_kernel_path(path: str | None) -> None:
    """
    Set the path of the directory containing SPICE kernels. See
    :ref:`the kernel directory documentation<kernel directory>` for more detail.

    Args:
        path: Directory which PlanetMapper will search for SPICE kernels. If `None`,
            then the default value of `'~/spice_kernels/'` will be used.
    """
    _KERNEL_DATA['kernel_path'] = path


def get_kernel_path() -> str:
    """
    Get the path of the directory of SPICE kernels used in PlanetMapper.

    #. If a kernel path has been manually set using :func:`set_kernel_path`, then this
       path is used.

    #. Otherwise the value of the environment variable `PLANETMAPPER_KERNEL_PATH` is
       used.

    #. If `PLANETMAPPER_KERNEL_PATH` is not set, then the default value,
       `'~/spice_kernels/'` is used.
    """
    if _KERNEL_DATA['kernel_path'] is not None:
        return _KERNEL_DATA['kernel_path']

    try:
        path = os.environ['PLANETMAPPER_KERNEL_PATH']
        if path:
            return path
    except KeyError:
        pass

    return DEFAULT_KERNEL_PATH
