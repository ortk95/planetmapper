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
`PlanetMapper` is an open source Python package which is used visualise, navigate and map astronomical observations of Solar System objects like planets and moons. Astronomers can use `PlanetMapper` to fit and navigate observations, ...



# Statement of need
<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->
In order to accurately interpret astronomical observations of objects in the solar system, it is crucial to understand the exact geometry of the observation. Some of the first questions in analysing a new dataset is working out exactly what you are looking at, for example:

- How is the planet oriented in the image? 
- What are the latitude and longitude coordinates for each pixel? 
- How is the planet's surface illuminated?
- Are points of light in the background sky moons of the target planet, or background stars?
- What is the line-of-sight velocity of the target's surface, and what is the associated doppler correction?

Without answering these kinds of questions, it is often impossible to accurately interpret the data. However, calculating the appearance of a target body is a complex problem, as it requires accurate knowledge of both the target and observers position and orientation in space at specific times. To add to the complexity, non-trivial effects, such as the light travel time from the target to the observer and stellar aberration must also be accounted for.

The NAIF SPICE Toolkit [@Acton2018] was developed by NASA to provide a standardised set of 'SPICE kernels', datasets containing the positions of solar system objects, and a set of tools to interface with these kernels. This toolkit ...

`PlanetMapper` effectively functions as a high level interface to the NAIF SPICE toolkit [@Acton2018], and makes use of the the `SpiceyPy` Python package [@Annex2020].

# Figures

![Saturn 'wireframe' plot generated with `PlanetMapper`, visualising the appearance of Saturn from the Earth on 1 January 2020.\label{fig:wireframe}](../docs/images/saturn_wireframe_radec.png)

![Screenshot of the `PlanetMapper` graphical user interface being used to fit an astronomical observation.\label{fig:gui}](../docs/images/gui_fitting.png)

![More complex example of `PlanetMapper`'s functionality. The navigated Jupiter observation (top left) was mapped (top right) using `PlanetMapper`. The emission angle backplanes generated with `PlanetMapper` are shown in the bottom panels. Jupiter image credit: NASA, ESA, STScI, A. Simon (Goddard Space Flight Center), and M.H. Wong (University of California, Berkeley) and the OPAL team.\label{fig:jupiter-mapped}](../docs/images/jupiter_mapped.png)

# Acknowledgements
`PlanetMapper` was developed with support from a European Research Council Consolidator Grant (under the European Union’s Horizon 2020 research and innovation programme, grant agreement No 723890). Thanks to Mike Roman and Naomi Rowe-Gurney for their suggestions, beta testing and feedback.

# References