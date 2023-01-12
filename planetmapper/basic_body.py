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


class BasicBody(SpiceBase):
    """
    Class representing astronomical body which is treated as a point source.

    This is typically used for objects which have limited data in the SPICE kernels
    (e.g. minor satellites which do not have well known radii). Usually, you are
    unlikely to need to create :class:`BasicBody` instances directly, but they may be
    returned when using :func:`Body.create_other_body`.

    This is a very simplified version of :class:`Body`.

    Args:
        target: Name of target body (see :class:`Body` for more details).
        utc: Time of observation (see :class:`Body` for more details).
        observer: Name of observing body (see :class:`Body` for more details).
        **kwargs: See :class:`Body` for more details about additional arguments.
    """

    def __init__(
        self,
        target: str | int,
        utc: str | datetime.datetime | float | None = None,
        observer: str | int = 'EARTH',
        *,
        aberration_correction: str = 'CN',
        observer_frame: str = 'J2000',
        illumination_source=None,
        subpoint_method=None,
        surface_method=None,
        **kwargs,
    ) -> None:
        # some arguments are unused, but keep them so that the function has the same
        # signature as Body()
        super().__init__(**kwargs)

        # Document instance variables
        self.et: float
        """Ephemeris time of the observation corresponding to `utc`."""
        self.dtm: datetime.datetime
        """Python timezone aware datetime of the observation corresponding to `utc`."""
        self.target_body_id: int
        """SPICE numeric ID of the target body."""
        self.target_light_time: float
        """Light time from the target to the observer at the time of the observation."""
        self.target_distance: float
        """Distance from the target to the observer at the time of the observation."""
        self.target_ra: float
        """Right ascension (RA) of the target centre."""
        self.target_dec: float
        """Declination (Dec) of the target centre."""

        # Process inputs
        if isinstance(utc, float):
            utc = self.mjd2dtm(utc)
        if utc is None:
            utc = datetime.datetime.now(datetime.timezone.utc)
        if isinstance(utc, datetime.datetime):
            # convert input datetime to UTC, then to a string compatible with spice
            utc = utc.replace(tzinfo=datetime.timezone.utc)
            utc = utc.strftime(self._DEFAULT_DTM_FORMAT_STRING)
        self.utc = utc

        self.target = self.standardise_body_name(target)
        self.observer = self.standardise_body_name(observer)
        self.observer_frame = observer_frame
        self.aberration_correction = aberration_correction

        # Get target properties and state
        self.et = spice.utc2et(self.utc)
        self.dtm: datetime.datetime = self.et2dtm(self.et)
        self.target_body_id: int = spice.bodn2c(self.target)

        starg, lt = spice.spkezr(
            self.target,
            self.et,
            self.observer_frame,
            self.aberration_correction,
            self.observer,
        )
        self._target_obsvec = cast(np.ndarray, starg)[:3]
        self.target_light_time = cast(float, lt)
        # cast() calls are only here to make type checking play nicely with spice.spkezr
        self.target_distance = self.target_light_time * self.speed_of_light()
        _, self._target_ra_radians, self._target_dec_radians = spice.recrad(
            self._target_obsvec
        )
        self.target_ra, self.target_dec = self._radian_pair2degrees(
            self._target_ra_radians, self._target_dec_radians
        )

    def __repr__(self) -> str:
        return f'BasicBody({self.target!r}, {self.utc!r})'
