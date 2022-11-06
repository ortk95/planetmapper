import datetime
import glob
import os
from typing import TypeVar

import numpy as np
import spiceypy as spice

KERNEL_PATH = '~/spice/naif/generic_kernels/'

Numeric = TypeVar('Numeric', bound=float | np.ndarray)


class SpiceBase:
    """
    Class containing methods to interface with spice and manipulate coordinates.

    This is the base class for all the main classes used in planetmapper.

    Args:
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
        optimize_speed: bool = True,
        load_kernels: bool = True,
        kernel_path: str = KERNEL_PATH,
        manual_kernels: None | list[str] = None,
    ) -> None:
        super().__init__()
        self._optimize_speed = optimize_speed

        if load_kernels:
            self.load_spice_kernels(
                kernel_path=kernel_path, manual_kernels=manual_kernels
            )

    def standardise_body_name(self, name: str | int) -> str:
        """
        Return a standardised version of the name of a SPICE body.

        This converts the provided `name` into the SPICE ID code, then back into a
        string, standardises to the version of the name preferred by SPICE. For example,
        `'jupiter'`, `'JuPiTeR'`, `' Jupiter '`, `'599'` and `599` are all standardised
        to `'JUPITER'`

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
        where :math:`\\lambda_r` is the wavelength recieved by the observer and
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
        kernel_path: str = KERNEL_PATH,
        manual_kernels: None | list[str] = None,
        only_if_needed: bool = True,
    ) -> None:
        """
        Attempt to intelligently SPICE kernels using `spice.furnsh`.

        Args:
            kernel_path: Path to directory where generic_kernels are stored.
            manual_kernels: Optional manual list of paths to kernels to load instead of
                using `kernel_path`.
            only_if_needed: If this is `True`, kernels will only be loaded once per
                session.
        """
        # TODO do this better - don't necessarily need to keep running it every time
        if only_if_needed and cls._KERNELS_LOADED:
            return
        if manual_kernels:
            kernels = manual_kernels
        else:
            kernel_path = os.path.expanduser(kernel_path)
            pcks = sorted(glob.glob(kernel_path + 'pck/*.tpc'))
            spks1 = sorted(glob.glob(kernel_path + 'spk/planets/de*.bsp'))
            spks2 = sorted(glob.glob(kernel_path + 'spk/satellites/*.bsp'))
            lsks = sorted(glob.glob(kernel_path + 'lsk/naif*.tls'))
            jwst = sorted(
                glob.glob(kernel_path + '../../jwst/**/*.bsp', recursive=True)
            )
            kernels = [pcks[-1], spks1[-1], *spks2, lsks[-1], *jwst]
        for kernel in kernels:
            spice.furnsh(kernel)
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
            Array of values of length :math:`n + 1` where the final value is the same as the
            first value.
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
