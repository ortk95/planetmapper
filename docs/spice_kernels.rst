.. _SPICE kernels:

SPICE kernels
*************

Introduction
============
The core logic of PlanetMapper uses the SPICE system, which was developed by NASA's `Navigation and Ancillary Information Facility <https://naif.jpl.nasa.gov/naif/>`_ to provide detailed and accurate information about the positions and properties of Solar System bodies and spacecraft. This SPICE database is stored in a series of files called 'SPICE kernels' which must be downloaded for PlanetMapper to function.

Most useful SPICE kernels can be found at https://naif.jpl.nasa.gov/pub/naif/. Each individual SPICE kernel typically contains information about a specific object or set of objects (e.g. one kernel file may contain information about Jupiter and its moons, while another may contain information about a specific spacecraft). Therefore, you only need to download a small subset of the SPICE kernels.


Downloading SPICE kernels
=========================
To aid in downloading appropriate SPICE kernels, PlanetMapper contains a series of useful functions such as :func:`planetmapper.kernel_downloader.download_urls` to download kernels from the `NAIF database <https://naif.jpl.nasa.gov/pub/naif/>`_. These functions will automatically download the SPICE kernels to a `spice_kernels` directory in your user directory. PlanetMapper automatically looks for kernels in `~/spice_kernels`, so once you only need to worry about downloading the kernels once, then PlanetMapper should deal with everything.

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


Using pre-existing kernels
==========================
If you already have the required kernels on your computer, you can use them instead of re-downloading them all. If your kernels are not `~/spice_kernels`, they will not be loaded by PlanetMapper's automatic kernel loading, so you will need to load the kernels manually. See the automatic kernel loading section below for more details.


Automatic kernel loading
========================
PlanetMapper will automatically load SPICE kernels the first time any object inheriting from :class:`planetmapper.SpiceBase` (e.g. :class:`planetmapper.Body`) is created. All kernels in `~/spice_kernels` which match any of the patterns `**/*.bsp`, `**/*.tpc` or `**/*.tls` are loaded by default. 

If you would like finer control over kernel loading, you can manually specify a list of kernel paths to load by specifying `manual_kernels=[...]` when e.g. creating a new :class:`planetmapper.Body` object. Alternatively, you can manually load kernels yourself using `spiceypy.furnsh` and then set `load_kernels=False` which will disable automatic kernel loading completely. 

See :class:`planetmapper.SpiceBase` and :func:`planetmapper.SpiceBase.load_spice_kernels` for more detail about controlling automatic kernel loading.
