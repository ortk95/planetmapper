import datetime

from .base import BodyBase


class BasicBody(BodyBase):
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
        **kwargs,
    ) -> None:
        # some arguments are unused, but allow them so that the function has the same
        # signature as Body()
        for k in ('illumination_source', 'subpoint_method', 'surface_method'):
            kwargs.pop(k, None)
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
        self.target_light_time: float
        """Light time from the target to the observer at the time of the observation."""
        self.target_distance: float
        """Distance from the target to the observer at the time of the observation."""
        self.target_ra: float
        """Right ascension (RA) of the target centre."""
        self.target_dec: float
        """Declination (Dec) of the target centre."""

    def __repr__(self) -> str:
        return f'BasicBody({self.target!r}, {self.utc!r})'

    def _get_equality_tuple(self) -> tuple:
        return (super()._get_equality_tuple(),)
