"""
.. warning::

    This code is in active development, and may break/change/not work at all.

    Bugs can be reported at https://github.com/ortk95/planetmapper/issues/new

The core logic of this code is based on conversion between different coordinate systems
of interest. The `xy` and `radec` coordinate systems define positions from the point of
view of the observer while the `lonlat` coordinate system defines locations on the
surface of the target body.

`xy` image pixel coordinates. These coordinates count the number of pixels in an
observed image with the bottom left pixel defined as `(0,0)`, and the `x` and `y`
coordinates defined as normal. Integer coordinates represent the middle of the
corresponding pixel, so `(0, 3)` covers `x` values from -0.5 to 0.5 and `y` values from
2.5 to 3.5.

`radec` observer frame RA/Dec sky coordinates. These are the right ascension and
declination which define the position in the sky of a point from the point of view of
the observer. These coordinates are expressed in degrees. See 
https://en.wikipedia.org/wiki/Equatorial_coordinate_system for more.

`lonlat` planetographic on target body. These are the planetographic longitude and
latitude coordinates of a point on the surface of the target body. These coordinates are 
expressed in degrees. See https://en.wikipedia.org/wiki/Planetary_coordinate_system for 
more.

================   =====
Dimension          Unit
================   =====
Angles             degrees  
Distances          km 
Time intervals     seconds
Speeds             km/s
Dates (timezone)   UTC   
================   =====

.. note::
    By default, all angles should be degrees unless using a function explicitly named 
    with `_radians`. Note that angles in SPICE are radians, so care should be taken 
    converting to/from SPICE values.

These additional coordinate systems are mainly used for internal calculations and to
interface with SPICE:

- `targvec` - target frame rectangular vector.
- `obsvec` - observer frame (e.g. J2000) rectangular vector.
- `obsvec_norm` - normalised observer frame rectangular vector.
- `rayvec` - target frame rectangular vector from observer to point.


For more detail about SPICE, see:
https://spiceypy.readthedocs.io/en/main/documentation.html
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/

"""
print(
    'WARNING: planetmapper in active development, and may break/change/not work at all'
)
print('Bugs can be reported at https://github.com/ortk95/planetmapper/issues/new')
from .common import __version__, __author__, __url__
from .base import SpiceBase
from .body import Body
from .body_xy import Backplane, BodyXY
from .observation import Observation
from . import gui
from . import utils

__all__ = [
    'SpiceBase',
    'Body',
    'Backplane',
    'BodyXY',
    'Observation',
    'gui',
    'utils',
]
