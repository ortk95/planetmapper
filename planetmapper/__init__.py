"""
Coordinate systems:

- `xy` - image pixel coordinates
- `radec` - observer frame RA/Dec coordinates
- `obsvec` - observer frame (e.g. J2000) rectangular vector
- `obsvec_norm` - normalised observer frame rectangular vector
- `rayvec` - target frame rectangular vector from observer to point
- `targvec` - target frame rectangular vector
- `lonlat` - planetary coordinates on target

By default, all angles should be degrees unless using a function explicitly named with
`_radians`. Note that angles in SPICE are radians, so care should be taken converting
to/from SPICE values.

For more detail about SPICE, see:
https://spiceypy.readthedocs.io/en/main/documentation.html
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/

.. warning::

    This code is in active development, and may break/change/not work at all.

    Bugs can be reported at https://github.com/ortk95/planetmapper/issues/new
"""
print(
    'WARNING: planetmapper in active development, and may break/change/not work at all'
)
print('Bugs can be reported at https://github.com/ortk95/planetmapper/issues/new')
from .common import __version__, __author__, __url__
from .planet_mapper_tool import KERNEL_PATH, PlanetMapperTool
from .body import Body
from .body_xy import Backplane, BodyXY
from .observation import Observation
from . import gui
from . import utils

__all__ = [
    'KERNEL_PATH',
    'PlanetMapperTool',
    'Body',
    'Backplane',
    'BodyXY',
    'Observation',
    'gui',
    'utils',
]
