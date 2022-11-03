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
"""
from .common import __version__, __author__, __url__
from .mapper import KERNEL_PATH, MapperTool, Body, BodyXY, Observation, Backplane
from . import gui
from . import utils
