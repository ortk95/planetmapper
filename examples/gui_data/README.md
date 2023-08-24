# Example dataset for use with the PlanetMapper GUI

This directory contains an example dataset which can be used to test the PlanetMapper GUI. The file [`europa.fits`](europa.fits) contains an observation of Europa taken by the VLT/SPHERE IRDIS instrument ([King et al., 2022](https://doi.org/10.3847/PSJ/ac596d)).

For more details about the PlanetMapper GUI, see the [documentation page](https://planetmapper.readthedocs.io/en/latest/user_interface.html).

## Required kernels

This example Europa observation requires general SPICE kernels to be downloaded (`lsk` and `pck`), along with the `de430.bsp` and `jup365.bsp` kernels. The following code snippet can be used to download these kernels:

```python
from planetmapper.kernel_downloader import download_urls

# General kernels
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/')
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/')

# Locations of planetary system barycentres
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp')

# Locations of Jupiter and its major satellites
download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp')
```

Note, the exact URLs in this example may not work if the `de430.bsp` and `jup365.bsp` kernels are superseded by newer versions. If you have any issues, the latest versions of the files can be found at:
- https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets
- https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites

For more details about SPICE kernels and PlanetMapper, see the [documentation page](https://planetmapper.readthedocs.io/en/latest/spice_kernels.html).
