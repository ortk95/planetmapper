# ![PlanetMapper logo](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/logo_wide_transparent.png)

[![PyPI](https://img.shields.io/pypi/v/planetmapper?label=PyPi&logo=python&logoColor=silver)](https://pypi.org/project/planetmapper/)
[![Publish](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml)
[![Checks](https://github.com/ortk95/planetmapper/actions/workflows/checks.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/checks.yml)
[![Coverage Status](https://img.shields.io/coverallsCoverage/github/ortk95/planetmapper)](https://coveralls.io/github/ortk95/planetmapper)
[![Documentation Status](https://readthedocs.org/projects/planetmapper/badge/?version=latest)](https://planetmapper.readthedocs.io/en/latest/?badge=latest)

PlanetMapper is an open source Python module for visualising, navigating and mapping Solar System observations.

## [Documentation](https://planetmapper.readthedocs.io)
For full documentation and [API reference](https://planetmapper.readthedocs.io/en/latest/documentation.html), visit [planetmapper.readthedocs.io](https://planetmapper.readthedocs.io/en/latest/index.html)


## [Installation](https://planetmapper.readthedocs.io/en/latest/installation.html)
```
pip install planetmapper --upgrade
```

_Requires Python 3.10+_

## Key features
### [Fit and map astronomical observations using a full featured user interface](https://planetmapper.readthedocs.io/en/latest/user_interface.html)
[![Screenshot of the PlanetMapper graphical user interface showing an observation of Europa being navigated](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/gui_fitting.png)](https://planetmapper.readthedocs.io/en/latest/user_interface.html)

### [Easily visualise solar system observations with just a few lines of code](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

```python
body = planetmapper.Body('saturn', '2020-01-01')
body.plot_wireframe_radec()
plt.show()
```

[![Image of Saturn generated with PlanetMapper showing the orientation of Saturn and its rings](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/saturn_wireframe_radec.png)](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

### [Convert coordinates, generate backplanes and project maps of telescope observations](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)
[![Plot of a mapped Jupiter observation, generated with PlanetMapper, showing observed and mapped versions of the Jupiter data](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/jupiter_mapped.png)](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)


## Contributing

If you spot a bug, have a suggestion, or want contribute code to PlanetMapper, check out the [contributing guidelines](https://github.com/ortk95/planetmapper/blob/main/CONTRIBUTING.md)!