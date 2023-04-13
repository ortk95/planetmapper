.. _common issues:

Common issues & solutions
*************************

.. note::
    If you find any bugs/errors that you cannot solve, please `open an issue at on GitHub <https://github.com/ortk95/planetmapper/issues/new>`__


General solutions
=================
Run `pip install planetmapper --upgrade` in a terminal to make sure you are running the latest version of PlanetMapper. It is possible that your bug may have been fixed in a recent update (you can also check the release notes for each version `on GitHub <https://github.com/ortk95/planetmapper/releases>`__).


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
This is likely to be due to an issue with your SPICE kernels or settings, possible fixes include...

- Make sure you are using the correct observer - e.g. a planet will appear in a different position from Earth and from JWST.
- Make sure you are using the correct observation time - times in PlanetMapper default to UTC, so make sure there are no time zone conversions needed.
- Make sure you have the latest version of any SPICE kernels, especially for any observers like HST or JWST which have have locations which are difficult to predict accurately.
- Make sure you are using the correct aberration correction.
- If you are using WCS information saved in the FITS header to automatically set the disc position, note that telescope pointing information (i.e. the WCS information) is never pefect. For example, due to the errors in guide star tracking, JWST pointing is only accurate to ~0.5".