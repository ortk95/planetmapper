.. _common issues:

Common issues
*************

.. note::
    If you find any bugs/errors that you cannot solve, please open an issue at https://github.com/ortk95/planetmapper/issues/new

SPICE Errors
============
If you have any errors caused reported by the SPICE system, it is likely that it doesn't have the correct SPICE kernels loaded. Therefore, make sure you have the :ref:`appropriate SPICE kernels downloaded <SPICE kernels>` to your computer and that you have set :ref:`the kernel directory<kernel directory>` correctly.

SPICE `NOLEAPSECONDS` Error
---------------------------
This error usually occurs when SPICE has not loaded *any* of your desired kernels. This may be because PlanetMapper is not looking in the correct directory for your kernels, so make sure you have set :ref:`the kernel directory<kernel directory>` correctly.


SPICE `SPKINSUFFDATA` Error
---------------------------
This error usually occurs when SPICE has successfully loaded some kernels, but is missing ephemeris data for either the target body or the observer body. This may be because you have not downloaded the appropriate kernel containing the ephemeris data for this body. 

The error message should tell you which body is missing, and you can then identify the correct kernel to download by searching the `NAIF database <https://naif.jpl.nasa.gov/pub/naif/>`_. For example, if you are missing data for a planetary body, you can search the `generic_kernels/spk/satellites/aa_summaries.txt <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/aa_summaries.txt>`_ file to identify which kernel need downloading.


Planets appear are in the wrong position
========================================
- Make sure you are using the correct observer - e.g. a planet will appear in a different position from Earth and from JWST.
- Make sure you are using the correct observation time - times in PlanetMapper default to UTC, so make sure there are no time zone conversions needed.
- Make sure you are using the correct aberration correction.