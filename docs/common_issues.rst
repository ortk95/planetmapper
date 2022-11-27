Common issues
*************

.. note::
    If you find any bugs/errors that you cannot solve, please open an issue at https://github.com/ortk95/planetmapper/issues/new

SPICE Errors
============
If you have any errors caused reported by the SPICE system, it is likely that it doesn't have the correct SPICE kernels loaded. Therefore, make sure you have the :ref:`appropriate SPICE kernels downloaded <SPICE kernels>` to your computer and that they are in the `~/spice_kernels` directory.

Planets appear are in the wrong position
========================================
- Make sure you are using the correct observer - e.g. a planet will appear in a different position from Earth and from JWST.
- Make sure you are using the correct observation time - times in PlanetMapper default to UTC, so make sure there are no time zone conversions needed.
- Make sure you are using the correct aberration correction.