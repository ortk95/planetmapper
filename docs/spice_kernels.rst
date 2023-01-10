.. _SPICE kernels:

SPICE kernels
*************

Introduction
============
The core logic of PlanetMapper uses the SPICE system, which was developed by NASA's `Navigation and Ancillary Information Facility <https://naif.jpl.nasa.gov/naif/>`_ to provide detailed and accurate information about the positions and properties of Solar System bodies and spacecraft. This SPICE database is stored in a series of files called 'SPICE kernels' which must be downloaded for PlanetMapper to function.

Most useful SPICE kernels can be found at https://naif.jpl.nasa.gov/pub/naif/. Each individual SPICE kernel typically contains information about a specific object or set of objects (e.g. one kernel file may contain information about Jupiter and its moons, while another may contain information about a specific spacecraft). Therefore, you only need to download a small subset of the SPICE kernels.


Downloading SPICE kernels
=========================
To aid in downloading appropriate SPICE kernels, PlanetMapper contains a series of useful functions such as :func:`planetmapper.kernel_downloader.download_urls` to download kernels from the `NAIF database <https://naif.jpl.nasa.gov/pub/naif/>`_. These functions will automatically download the SPICE kernels to your computer where they can be used by PlanetMapper, so you only need to worry about downloading the kernels once, then PlanetMapper will deal with everything.

.. hint::
    By default, PlanetMapper will downloaded and search for kernels in a directory named `spice_kernels` within your user directory. If you would like to customise this location (e.g. if you already have kernels saved elsewhere), see the :ref:`section on customising the kernel directory<kernel directory>` below.

Required kernels
----------------
To download kernels which are required for virtually every SPICE calculation, run the following commands in Python: ::

    from planetmapper.kernel_downloader import download_urls
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/')
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/')

This will automatically download a series of 'leap second kernels' and 'planetary constant kernels' to your computer. The function :func:`planetmapper.kernel_downloader.download_urls` effectively replicates the file structure from the `NAIF database <https://naif.jpl.nasa.gov/pub/naif/>`_ to your local system. Therefore, the leap second kernels will be automatically downloaded to `~/spice_kernels/naif/generic_kernels/lsk/` and the planetary constant kernels will be downloaded to `~/spice_kernels/naif/generic_kernels/pck/`.

If you don't want to use :func:`planetmapper.kernel_downloader.download_urls`, you can instead manually browse and download files from https://naif.jpl.nasa.gov/pub/naif/ yourself.

These required kernels are relatively small (~50KB for `.../lsk` and ~50MB for `.../pck`), so downloading them should be relatively fast. Once you have downloaded these required kernels, you will also need to download some of the ephemeris kernels described below.


Planetary ephemeris kernels
---------------------------
.. warning::
    Some SPICE ephemeris kernels can be very large (>1GB), so make sure you have enough free disk space when downloading.

The positions of solar system bodies (e.g. planets and moons) are contained in ephemeris kernels. The kernels are required for each body which you are observing, e.g. if you are using observations of Jupiter you will need to have a Jupiter kernel downloaded.

If you have enough disk space, you can easily download all the planet and moon kernels using: ::

    from planetmapper.kernel_downloader import download_urls
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/')
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/')

These `spk` kernels add up to ~30GB, so if you have limited disk space, you may want to instead download the specific kernels for the bodies you are interested in. Look at the `summaries` and `readme` text files at the top of the `planets <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>`_ and `satellites <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/>`_ archive pages to see which specific files you want to download. For example, if you are only interested in Jupiter and its moons, you could instead use: ::

    from planetmapper.kernel_downloader import download_urls
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp')
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup344.bsp')
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp')
    # The exact URLs given here may not work if new kernel versions are published


Spacecraft kernels
------------------
If you are using observations from a spacecraft, you will also need to download the ephemeris kernels describing the spacecraft's position over time. For example, if you are using observations from the Hubble Space Telescope, you should run: ::

    from planetmapper.kernel_downloader import download_urls
    download_urls('https://naif.jpl.nasa.gov/pub/naif/HST/kernels/spk/')

The directory name for different missions can be found by searching the `NAIF archive <https://naif.jpl.nasa.gov/pub/naif/>`_.

Other kernels
-------------
In some cases, you may require other kernels in addition to those listed above. You should be able to identify the kernels required by searching the `NAIF archive <https://naif.jpl.nasa.gov/pub/naif/>`_. For example, if you are observing comets, you can download comet ephemerides using ::

    from planetmapper.kernel_downloader import download_urls
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/comets/')


.. _kernel directory:

Customising the kernel directory
================================
By default, PlanetMapper downloads and looks for spice kernels in the `~/spice_kernels` directory. However, if needed (e.g. if you already have kernels saved elsewhere), this directory can be customised using the different methods described below. The environment variable method is usually the simplest and easiest.


Method 1: Environment variable
------------------------------
The easiest way to customise the directory is to set the environment variable `PLANETMAPPER_KERNEL_PATH` to point to your desired path. For example, on a Unix-like system, you can add a line to to your `.bash_profile` file to automatically set this environment variable: ::

    export PLANETMAPPER_KERNEL_PATH="/path/where/you/save/your/spice/kernels"


Method 2: Using `set_kernel_path`
---------------------------------
The function :func:`planetmapper.set_kernel_path` can be used to set the kernel path for a single script. This function *must* be called before using any other `planetmapper` functionality, so it is easiest to run :func:`planetmapper.set_kernel_path` immediately after importing `planetmapper`: ::

    import planetmapper
    planetmapper.set_kernel_path('/path/where/you/save/your/spice/kernels')

This path should also be set before downloading any SPICE kernels, otherwise they will be downloaded to the incorrect directory: ::

    import planetmapper
    from planetmapper.kernel_downloader import download_urls
    planetmapper.set_kernel_path('/path/where/you/save/your/spice/kernels')

    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/')
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/')


Automatic kernel loading
========================
PlanetMapper will automatically load SPICE kernels the first time any object inheriting from :class:`planetmapper.SpiceBase` (e.g. :class:`planetmapper.Body`) is created. All kernels in the directory returned by :func:`planetmapper.get_kernel_path` which match any of the patterns `**/*.bsp`, `**/*.tpc` or `**/*.tls` are loaded by default. 

If you would like finer control over kernel loading, you can manually specify a list of kernel paths to load by specifying `manual_kernels=[...]` when e.g. creating a new :class:`planetmapper.Body` object. Alternatively, you can manually load kernels yourself using `spiceypy.furnsh` and then set `load_kernels=False` which will disable automatic kernel loading completely. 

See :class:`planetmapper.SpiceBase` and :func:`planetmapper.SpiceBase.load_spice_kernels` for more detail about controlling automatic kernel loading.
