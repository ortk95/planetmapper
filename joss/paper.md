---
title: 'PlanetMapper: A Python package for visualising, navigating and mapping Solar System observations'
tags:
  - Python
  - astronomy
authors:
  - name: Oliver R.T. King
    orcid: 0000-0002-6271-0062
    corresponding: true
    affiliation: 1
  - name: Leigh N. Fletcher
    orcid: 0000-0001-5834-9588
    affiliation: 1
affiliations:
 - name: School of Physics and Astronomy, University of Leicester, University Road, Leicester, LE1 7RH, United Kingdom
   index: 1
date: 12 June 2023
bibliography: paper.bib
---

# Summary
<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->
`PlanetMapper` is an open source Python package that is used visualise, process and understand astronomical observations of Solar System objects, such as planets, moons and rings. Astronomers can use `PlanetMapper` to 'navigate' observations by calculating coordinate values (such as latitude and longitude) for each pixel in an observed image, and can map observations by projecting the observed data onto a map of the target body. Calculated values are exportable and directly accessible through a well documented API, allowing `PlanetMapper` to be used for custom analysis and processing. `PlanetMapper` can also be used to help generate publication quality figures, and has a Graphical User Interface to significantly simplify the processing of astronomical data. `PlanetMapper` can be applied to a wide range of datasets, including both amateur and professional ground-based observations, and data from space telescopes like Hubble and JWST.

# Statement of need
<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->
In order to accurately interpret astronomical observations of objects in the Solar System, it is crucial to understand the exact geometry and illumination conditions of the observation. Some of the first questions in analysing a new dataset are working out exactly what you are looking at, for example:

- How is the planet oriented in the image? 
- What are the latitude and longitude coordinates for each pixel? 
- How is the planet's surface illuminated?
- Are points of light in the background sky moons of the target planet, or background stars?
- What is the line-of-sight velocity of the target's surface, and what is the associated doppler correction?

Without answering these kinds of questions, it is often challenging to accurately interpret the data. However, calculating the appearance of a target body is a complex problem, as it requires accurate knowledge of both the target and observers position and orientation in space at specific times. To add to the complexity, non-trivial effects, such as the light travel time from the target to the observer and stellar aberration must also be accounted for.

The NAIF SPICE Toolkit [@Acton2018] was developed by NASA to provide a standardised set of 'SPICE kernels', datasets containing the positions of Solar System objects, and a set of tools to interface with these kernels. This toolkit provides low level functions which can be combined to solve the problem of calculating the appearance of a target body. `PlanetMapper` is designed to significantly simplify the use of SPICE for planetary astronomers, effectively providing a high level interface to the toolkit. For example, the conversion between right ascension/declination coordinates (in the sky of the observer) to latitude/longitude coordinates (on the target body) requires calling ~10 SPICE functions, but can be done in a single function call with `PlanetMapper`. `PlanetMapper` makes use of the `SpiceyPy` package [@Annex2020] which provides a Python interface to the low level SPICE toolkit functions.

Planetary astronomers have developed toolkits for image navigation, such as the `IDL` language based `DRM` [@Fletcher2009] and the `WinJUPOS` windows application for analysing Jupiter observations ([jupos.privat.t-online.de](http://jupos.privat.t-online.de/index.htm)). The USGS `ISIS` software [@ISIS] also provides a comprehensive set of tools for processing and navigating data from specific instruments on some spacecraft missions, but does not support generalised datasets or ground-based observations. `PlanetMapper` is the first general purpose Python package for navigating and mapping astronomical observations, and is designed to be used with any form of imaging data observing any Solar System object which has a SPICE kernel available.

# Functionality

Publication quality figures can be created with `PlanetMapper` and the `matplotlib` package [@Hunter2007] - for example, \autoref{fig:wireframe} shows a visualisation of the appearance of Saturn at a specific time. These plots can be used to help visualise and plan observing campaigns, and to help interpret observations by providing geometric context for data. Information about the observer-target geometry can also be generated, including data such as calculating the apparent size of the target and testing if a moon is in eclipse or occultation.

Astronomers can use `PlanetMapper` to calculate the geometry of an astronomical observation, and generate a series of 'backplanes' which contain the coordinates (latitude/longitude, illumination angles, ring plane coordinates, velocities etc.) for each pixel in the observation. These backplanes can be saved to FITS data files for future use, or used directly in Python code. `PlanetMapper` contains functions to project observed data to a map (\autoref{fig:jupiter-mapped}), and to export FITS files containing this mapped data.

The `PlanetMapper` Graphical User Interface (GUI), shown in \autoref{fig:gui}, allows users to interactively fit, navigate and save observations with no coding required. This GUI can also be invoked from within Python code, allowing users to easily fit observations within their own data reduction and processing workflow.

`PlanetMapper` is actively used in the processing and  analysis of observations in JWST Giant Planets programmes [@King2023], and has been used to help create data visualisations and figures. It can work with a wide range of datasets, including those from ground (e.g. VLT) and space based (e.g. JWST) telescopes, spacecraft missions and amateur observations. The ability to generate generalised data and plots, such as \autoref{fig:wireframe}, means that `PlanetMapper` can also be useful for more general research purposes, even if the user is not specifically working with astronomical images.

`PlanetMapper` is tested with both unit and integration tests which run automatically using GitHub's continuous integration service. The package is well documented, with all public methods and functions containing detailed docstrings, and documentation automatically built at [planetmapper.readthedocs.io](https://planetmapper.readthedocs.io). `PlanetMapper` is distributed on PyPI and the code is licensed under the MIT license. 

# Figures

![Saturn 'wireframe' plot generated with `PlanetMapper`, visualising the appearance of Saturn from the Earth on 1 January 2020. This plot was created with a single function call, and all elements are fully customisable.\label{fig:wireframe}](../docs/images/saturn_wireframe_radec.png)

![More complex example of `PlanetMapper`'s functionality. The navigated Jupiter observation (top left) was mapped (top right) using `PlanetMapper`. The emission angle backplanes generated with `PlanetMapper` are shown in the bottom panels. Jupiter image credit: NASA, ESA, STScI, A. Simon (Goddard Space Flight Center), and M.H. Wong (University of California, Berkeley) and the OPAL team.\label{fig:jupiter-mapped}](../docs/images/jupiter_mapped.png)

![Screenshot of the `PlanetMapper` graphical user interface being used to fit a ground-based VLT observation of Europa [@King2022]. The user can adjust the location of Europa's fitted disc (the white circle) until it matches Europa's observed disc. If the observation has embedded WCS information (about the approximate telescope pointing), the disc position, rotation and size is initialised with the position derived from the WCS, so often only small manual adjustments to the disc position are needed.\label{fig:gui}](../docs/images/gui_fitting.png)


# Acknowledgements
`PlanetMapper` was developed with support from a European Research Council Consolidator Grant (under the European Union’s Horizon 2020 research and innovation programme, grant agreement No 723890). Thanks to Mike Roman and Naomi Rowe-Gurney for their suggestions, beta testing and feedback.

# References