"""
Module containing any exceptions and warnings used by PlanetMapper.
"""

import os
import warnings


class PlanetmapperWarning(Warning):
    """
    Base class for all warnings raised by PlanetMapper.
    """


def warn(message: str, *, category: type[Warning] = PlanetmapperWarning) -> None:
    """
    Issue a warning, skipping stack frames within the planetmapper package. This means
    that the warning will to be attributed to the relevant line of user code.
    """
    warnings.warn(
        message,
        category=category,
        skip_file_prefixes=(os.path.dirname(__file__),),
    )
