"""
Module containing any exceptions and warnings used by PlanetMapper.
"""

import os
import sys
import warnings


class PlanetmapperWarning(Warning):
    """
    Base class for all warnings raised by PlanetMapper.
    """


def warn(message: str, *, category: type[Warning] = PlanetmapperWarning) -> None:
    """
    Emit a warning with the given message.
    """
    if sys.version_info >= (3, 12):
        # Skip stack frames within the planetmapper package, so that the warning is
        # attributed to the relevant line of user code.
        warnings.warn(
            message,
            category=category,
            skip_file_prefixes=(os.path.dirname(__file__),),
        )
    else:
        # skip_file_prefixes was added in Python 3.12, so for earlier versions, simply
        # skip this warn() function instead.
        warnings.warn(
            message,
            category=category,
            stacklevel=2,
        )
