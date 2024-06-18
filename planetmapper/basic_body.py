import datetime
from typing import Any

from .base import BodyBase, _add_help_note_to_spice_errors


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

    @_add_help_note_to_spice_errors
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
        return self._generate_repr('target', 'utc', kwarg_keys=['observer'])

    def _get_equality_tuple(self) -> tuple:
        return (super()._get_equality_tuple(),)

    @classmethod
    def _get_default_init_kwargs(cls) -> dict[str, Any]:
        return dict(
            observer='EARTH',
            aberration_correction='CN',
            observer_frame='J2000',
            **super()._get_default_init_kwargs(),
        )
