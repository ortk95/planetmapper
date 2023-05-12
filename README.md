# PlanetMapper

[![PyPI](https://img.shields.io/pypi/v/planetmapper?label=PyPi&logo=python&logoColor=silver)](https://pypi.org/project/planetmapper/)
[![Upload Package](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml)
[![Pylint](https://github.com/ortk95/planetmapper/actions/workflows/pylint.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/pylint.yml)
[![Format](https://github.com/ortk95/planetmapper/actions/workflows/format.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/format.yml)
[![Tests](https://github.com/ortk95/planetmapper/actions/workflows/test.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/test.yml)
[![Coverage Status](https://img.shields.io/coverallsCoverage/github/ortk95/planetmapper)](https://coveralls.io/github/ortk95/planetmapper)
[![Documentation Status](https://readthedocs.org/projects/planetmapper/badge/?version=latest)](https://planetmapper.readthedocs.io/en/latest/?badge=latest)


A Python module for visualising, navigating and mapping Solar System observations.

## [Documentation](https://planetmapper.readthedocs.io)
For full documentation and [API reference](https://planetmapper.readthedocs.io/en/latest/documentation.html), visit [planetmapper.readthedocs.io](https://planetmapper.readthedocs.io/en/latest/index.html)


## [Installation](https://planetmapper.readthedocs.io/en/latest/installation.html)
```
pip install planetmapper --upgrade
```

_Requires Python 3.10+_

## Key features
### [Fit and map astronomical observations using a full featured user interface](https://planetmapper.readthedocs.io/en/latest/user_interface.html)
![PlanetMapper graphical user interface](https://github.com/ortk95/planetmapper/blob/main/docs/images/gui_fitting.png?raw=true)

### [Easily visualise solar system observations with just a few lines of code](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

```python
body = planetmapper.Body('saturn', '2020-01-01')
body.plot_wireframe_radec()
plt.show()
```

![Image of Saturn generated with PlanetMapper](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/saturn_wireframe_radec.png)

### [Convert coordinates, generate backplanes and project maps of telescope observations](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)
![Plot of a mapped Jupiter observation, generated with PlanetMapper](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/jupiter_mapped.png)
