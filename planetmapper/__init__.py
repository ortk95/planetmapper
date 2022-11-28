"""
A Python module for navigating and mapping Solar System observations.

.. warning::

    This code is in active development, and may break/change/not work at all.

    Bugs can be reported at https://github.com/ortk95/planetmapper/issues/new

The core logic of this code is based on conversion between different coordinate systems
of interest. The `xy` and `radec` coordinate systems define positions from the point of
view of the observer while the `lonlat` coordinate system defines locations on the
surface of the target body:

`xy`: image pixel coordinates. These coordinates count the number of pixels in an
observed image with the bottom left pixel defined as `(0,0)`, and the `x` and `y`
coordinates defined as normal. Integer coordinates represent the middle of the
corresponding pixel, so `(0, 3)` covers `x` values from -0.5 to 0.5 and `y` values from
2.5 to 3.5.

`radec`: observer frame RA/Dec sky coordinates. These are the right ascension and
declination which define the position in the sky of a point from the point of view of
the observer. These coordinates are expressed in degrees. See 
https://en.wikipedia.org/wiki/Equatorial_coordinate_system for more.

`lonlat`: planetographic coordinates on target body. These are the planetographic 
longitude and latitude coordinates of a point on the surface of the target body. These 
coordinates are expressed in degrees. See 
https://en.wikipedia.org/wiki/Planetary_coordinate_system and 
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/recpgr_c.html#Particulars for
more.

`km`: defines the distance in the image plane from the centre of the target body in km
with the target's north pole pointing up. This coordinate system is similar to the
`radec` and `xy` coordinate systems, but has the image zoomed so that the planet's 
radius is fixed and rotated so that the north pole points up. It can therefore be useful
for comparing observations of the same target taken at different times.

=============================================   =====
Dimension                                       Unit
=============================================   =====
Angles (RA, Dec, longitude, latitude...)        degrees
Angles (target angular diameter, plate scale)   arcseconds [#arcsec]_
Distances                                       km
Time intervals                                  seconds
Speeds                                          km/s
Dates (timezone)                                UTC   
=============================================   =====

.. [#arcsec] 3600 arcseconds = 1 degree

.. note::
    By default, all angles should be degrees unless using a function/value explicitly 
    named with `_arcsec` or `_radians`. Note that angles in SPICE are radians, so extra
    care should be taken converting to/from SPICE values.

These additional coordinate systems are mainly used for internal calculations and to
interface with SPICE:

- `targvec` - target frame rectangular vector.
- `obsvec` - observer frame (e.g. J2000) rectangular vector.
- `obsvec_norm` - normalised observer frame rectangular vector.
- `rayvec` - target frame rectangular vector from observer to point.


This code makes extensive use of the the `spiceypy` package which provides a Python
wrapper around NASA's `cspice` toolkit. For more details, see:

- https://spiceypy.readthedocs.io/en/main/documentation.html
- https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/
"""
from .common import __version__, __author__, __url__
from .base import SpiceBase
from .body import Body
from .body_xy import Backplane, BodyXY
from .observation import Observation
from . import gui
from . import utils
from . import kernel_downloader
from . import data_loader

__all__ = [
    'SpiceBase',
    'Body',
    'Backplane',
    'BodyXY',
    'Observation',
    'gui',
    'utils',
    'kernel_downloader',
]


def main():
    """:meta private:"""
    gui._main()
