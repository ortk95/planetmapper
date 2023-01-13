import datetime
import glob
import os
from typing import TypeVar

import astropy.time
import numpy as np
import spiceypy as spice

from . import progress

DEFAULT_KERNEL_PATH = '~/spice_kernels/'

_KERNEL_DATA = {
    'kernel_path': None,
    'kernel_patterns': ('**/*.bsp', '**/*.tpc', '**/*.tls'),
}

Numeric = TypeVar('Numeric', bound=float | np.ndarray)


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
        load_kernels: Toggle automatic kernel loading with :func:`load_spice_kernels`.
        kernel_path: Passed to  :func:`load_spice_kernels` if `load_kernels` is True.
        manual_kernels: Passed to  :func:`load_spice_kernels` if `load_kernels` is True.
    """

    _DEFAULT_DTM_FORMAT_STRING = '%Y-%m-%dT%H:%M:%S.%f'
    _KERNELS_LOADED = False

    def __init__(
        self,
        show_progress: bool = False,
        optimize_speed: bool = True,
        load_kernels: bool = True,
        kernel_path: str | None = None,
        manual_kernels: None | list[str] = None,
    ) -> None:
        super().__init__()
        self._optimize_speed = optimize_speed

        self._progress_hook: progress.ProgressHook | None = None
        self._progress_call_stack: list[str] = []

        if show_progress:
            self._set_progress_hook(progress.CLIProgressHook())

        if load_kernels:
            self.load_spice_kernels(
                kernel_path=kernel_path, manual_kernels=manual_kernels
            )

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

    @classmethod
    def load_spice_kernels(
        cls,
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
        if only_if_needed and cls._KERNELS_LOADED:
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
            cls._KERNELS_LOADED = True

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

    def _update_progress_hook(self, progress: float) -> None:
        """Update progress hook with `progress` of current function between 0 & 1"""
        if self._progress_hook is not None:
            self._progress_hook(progress, self._progress_call_stack)


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


def set_kernel_path(path: str) -> None:
    """
    Set the path of the directory containing SPICE kernels. See
    :ref:`the kernel directory documentation<kernel directory>` for more detail.

    Args:
        path: Directory which PlanetMapper will search for SPICE kernels.
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
