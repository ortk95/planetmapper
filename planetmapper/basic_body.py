import datetime
from typing import cast

import numpy as np
import spiceypy as spice

from .base import SpiceBase, _BodyBase


class BasicBody(_BodyBase):
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
        super().__init__(
            target=target,
            utc=utc,
            observer=observer,
            aberration_correction=aberration_correction,
            observer_frame=observer_frame,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f'BasicBody({self.target!r}, {self.utc!r})'

    def _get_equality_tuple(self) -> tuple:
        return (super()._get_equality_tuple(),)
