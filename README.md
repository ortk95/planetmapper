# PlanetMapper

[![PyPI](https://img.shields.io/pypi/v/planetmapper?logo=python&logoColor=silver)](https://pypi.org/project/planetmapper/)
[![Documentation Status](https://readthedocs.org/projects/planetmapper/badge/?version=latest)](https://planetmapper.readthedocs.io/en/latest/?badge=latest)
[![Upload Python Package](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml)


A Python module for visualising, navigating and mapping Solar System observations.

## [Documentation](https://planetmapper.readthedocs.io)
For full documentation and [API reference](https://planetmapper.readthedocs.io/en/latest/documentation.html), visit [planetmapper.readthedocs.io](https://planetmapper.readthedocs.io/en/latest/index.html)


## [Installation](https://planetmapper.readthedocs.io/en/latest/installation.html)
```
pip install planetmapper
```

## Key features
### [Fit and map observations using a full featured user interface](https://planetmapper.readthedocs.io/en/latest/user_interface.html)
![PlanetMapper graphical user interface](https://github.com/ortk95/planetmapper/blob/main/docs/images/gui_fitting.png?raw=true)

### [Easily visualise solar system observations with just a few lines of code](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

```python
body = planetmapper.Body('saturn', '2020-01-01')
body.plot_wireframe_radec()
plt.show()
```

![Image of Saturn generated with PlanetMapper](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/saturn_wireframe_radec.png)

### [Convert coordinates, generate backplanes and project maps of telescope observations](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)
![Image of Europa generated with PlanetMapper](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/europa_backplane.png)
