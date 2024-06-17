"""
PlanetMapper: A Python package for visualising, navigating and mapping Solar System
observations.

..
    See https://planetmapper.readthedocs.io for full documentation.

The core logic of this code is based on conversion between different coordinate systems
of interest. The `xy` and `radec` coordinate systems define positions from the point of
view of the observer while the `lonlat` coordinate system defines locations on the
surface of the target body:

`xy`: image pixel coordinates. These coordinates count the number of pixels in an
observed image with the bottom left pixel defined as `(0, 0)`, and the `x` and `y`
coordinates defined as normal. Integer coordinates represent the middle of the
corresponding pixel, so `(0, 3)` covers `x` values from -0.5 to 0.5 and `y` values from
2.5 to 3.5.

`radec`: observer frame RA/Dec sky coordinates. These are the right ascension and
declination which define the position in the sky of a point from the point of view of
the observer. These coordinates are expressed in degrees. See 
`Wikipedia <https://en.wikipedia.org/wiki/Equatorial_coordinate_system>`__ for more 
details.

`lonlat`: planetographic coordinates on target body. These are the planetographic 
longitude and latitude coordinates of a point on the surface of the target body. These 
coordinates are expressed in degrees. See 
`Wikipedia <https://en.wikipedia.org/wiki/Planetary_coordinate_system>`__ and 
`the SPICE documentation <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/recpgr_c.html#Particulars>`__
for more details. The surface altitude can also be customised with the `alt` parameter,
for example, `body.lonlat2radec(12, 34, alt=1000)` will calculate the RA/Dec coordinates
of the point at planetographic coordinates (12, 34) and with an altitude of 1000 km. If
planetocentric coordinates are desired, then functions
:func:`Body.graphic2centric_lonlat` and :func:`Body.centric2graphic_lonlat` can be used
to convert between planetographic and planetocentric coordinates.

`km`: defines the distance in the image plane from the centre of the target body in km
with the target's north pole pointing up. This coordinate system is similar to the
`radec` and `xy` coordinate systems, but has the image zoomed so that the planet's 
radius is fixed and rotated so that the north pole points up. It can therefore be useful
for comparing observations of the same target taken at different times.

`angular`: relative angular coordinates in arcseconds. By default, the angular
coordinates are centred on the target body, with the same rotation as the `radec`
coordinates, meaning the angular coordinates define the distance in arcseconds from the
centre of the target body. However, the origin and rotation of the angular coordinates
can also be customised to measure the angular distance in arcseconds relative to an
arbitrary point in the sky. See :func:`Body.radec2angular` for more details on
customising the `angular` coordinates. Similarly to the `km` coordinate system, this
can be useful for comparing observations of the same target taken at different times. It
also can be used to minimise the distortion present when plotting `radec` coordinates
for targets located near the celestial pole.

=====================================================   =====
Dimension                                               Unit
=====================================================   =====
Angles (RA, Dec, longitude, latitude...)                degrees
Angles (relative angular coordinates, plate scale...)   arcseconds [#arcsec]_
Distances                                               km
Time intervals                                          seconds
Speeds                                                  km/s
Dates (timezone)                                        UTC   
=====================================================   =====

.. [#arcsec] 3600 arcseconds = 1 degree

.. note::
    By default, all angles should be degrees unless using a function/value explicitly 
    named with `_arcsec` or `_radians`, or using the relative `angular` coordinate 
    system. Note that angles in SPICE are radians, so extra care should be taken 
    converting to/from SPICE values.

These additional coordinate systems are mainly used for internal calculations and to
interface with SPICE:

- `targvec` - target frame rectangular vector.
- `obsvec` - observer frame (e.g. J2000) rectangular vector.
- `obsvec_norm` - normalised observer frame rectangular vector.
- `rayvec` - target frame rectangular vector from observer to point.


This code makes extensive use of the the `spiceypy` package which provides a Python
wrapper around NASA's `cspice` toolkit. See the 
`spiceypy documentation <https://spiceypy.readthedocs.io/en/main/documentation.html>`__
and the 
`SPICE documentation <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/>`__
for more information.


If you use PlanetMapper in your research, please :ref:`cite the following paper
<citation>`:

   King et al., (2023). PlanetMapper: A Python package for visualising, navigating and
   mapping Solar System observations. Journal of Open Source Software, 8(90), 5728,
   https://doi.org/10.21105/joss.05728


.. warning::

    This code is in active development, so may contain bugs! Any issues, bugs and 
    suggestions can be 
    `reported on GitHub <https://github.com/ortk95/planetmapper/issues/new>`__.

..
    MIT License

    Copyright (c) 2022 Oliver King

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from . import base, data_loader, gui, kernel_downloader, utils
from .base import SpiceBase, get_kernel_path, set_kernel_path
from .basic_body import BasicBody
from .body import (
    DEFAULT_WIREFRAME_FORMATTING,
    AngularCoordinateKwargs,
    Body,
    WireframeComponent,
    WireframeKwargs,
)
from .body_xy import Backplane, BodyXY, MapKwargs
from .common import __author__, __description__, __license__, __url__, __version__
from .observation import Observation

__all__ = [
    'set_kernel_path',
    'get_kernel_path',
    'SpiceBase',
    'Body',
    'Backplane',
    'BodyXY',
    'Observation',
    'BasicBody',
    'AngularCoordinateKwargs',
    'WireframeKwargs',
    'WireframeComponent',
    'DEFAULT_WIREFRAME_FORMATTING',
    'MapKwargs',
    'base',
    'gui',
    'utils',
    'kernel_downloader',
    'data_loader',
]
