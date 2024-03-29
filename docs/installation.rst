.. _installation:

Installation
************

Installing PlanetMapper
=======================

PlanetMapper can easily be `installed from PyPI <https://pypi.org/project/planetmapper/>`_ using pip by running: ::
    
    pip install planetmapper

or with conda by running: ::

    conda install -c conda-forge planetmapper

This will automatically install PlanetMapper, along with any dependencies (e.g. NumPy and Astropy) which you do not already have installed. Note that PlanetMapper requires a minimum Python version of 3.10.

Updating PlanetMapper
=====================
.. _updating_planetmapper:

To upgrade an existing PlanetMapper installation to the latest version, run: ::

    pip install planetmapper --upgrade

if you installed PlanetMapper with pip, or: ::

    conda update planetmapper

if you installed PlanetMapper with conda.

The release notes for each version can be `found on GitHub <https://github.com/ortk95/planetmapper/releases>`__, and you can check what version of PlanetMapper you have installed by running: ::

    import planetmapper
    print(planetmapper.__version__)

First steps
===========

The core logic of PlanetMapper uses a series of files called ':ref:`SPICE kernels`' which contain the information about the positions and properties of Solar System bodies. Therefore, once you have PlanetMapper installed, you will need to :ref:`download the appropriate kernels <SPICE kernels>` before you can properly use PlanetMapper.

Once you have the SPICE kernels downloaded, you can type `planetmapper` in the command line to open an interactive window, or `import planetmapper` in a Python script to get the full functionality.

.. hint::
    Check the :ref:`list of common issues<common issues>` if you encounter any problems when using PlanetMapper