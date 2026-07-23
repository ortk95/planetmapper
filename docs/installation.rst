.. _installation:

Installation
************

Installing PlanetMapper
=======================

PlanetMapper can easily be `installed from PyPI <https://pypi.org/project/planetmapper/>`_ using pip by running: ::
    
    pip install planetmapper

or with `conda <https://anaconda.org/channels/conda-forge/packages/planetmapper/overview>`_ by running: ::

    conda install -c conda-forge planetmapper

This will automatically install PlanetMapper, along with any dependencies (e.g. NumPy and Astropy) that you do not already have installed. Note that PlanetMapper requires a minimum Python version of 3.10.

.. _updating_planetmapper:

Updating PlanetMapper
=====================

If you installed PlanetMapper with pip, it can be upgraded to the latest version by running: ::

    pip install planetmapper --upgrade

or, if you installed PlanetMapper with conda, run: ::

    conda update planetmapper

Note that it can sometimes take a few days for the latest version to appear on conda after it has been released on PyPI, so try again later if you aren't getting the very latest version.

.. image:: https://img.shields.io/pypi/v/planetmapper?label=PyPI
    :target: https://pypi.org/project/planetmapper/
    :alt: PyPI Version
.. image:: https://img.shields.io/conda/vn/conda-forge/planetmapper?label=conda
    :target: https://anaconda.org/conda-forge/planetmapper
    :alt: Conda Version

The release notes for each version can be `found on GitHub <https://github.com/ortk95/planetmapper/releases>`__, and you can check what version of PlanetMapper you have installed by running: ::

    import planetmapper
    print(planetmapper.__version__)

First steps
===========

The core logic of PlanetMapper uses a series of files called ':ref:`SPICE kernels`' which contain the information about the positions and properties of Solar System bodies. Therefore, once you have PlanetMapper installed, you will need to :ref:`download the appropriate kernels <SPICE kernels>` before you can properly use PlanetMapper.

Once you have the SPICE kernels downloaded, you can type `planetmapper` in the command line to open an interactive window, or `import planetmapper` in a Python script to get the full functionality.

.. seealso::
    Check the :ref:`list of common issues<common issues>` if you encounter any problems when using PlanetMapper
