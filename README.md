# ![PlanetMapper logo](docs/images/logo_wide_transparent.png)

[![PyPI Version](https://img.shields.io/pypi/v/planetmapper?label=PyPI)](https://pypi.org/project/planetmapper/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/planetmapper?label=conda)](https://anaconda.org/conda-forge/planetmapper)
[![Publish Workflow Status](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ortk95/planetmapper/actions/workflows/python-publish.yml)
[![Checks Workflow Status](https://github.com/ortk95/planetmapper/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/ortk95/planetmapper/actions/workflows/checks.yml)
[![Documentation Status](https://readthedocs.org/projects/planetmapper/badge/?version=latest)](https://planetmapper.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/ortk95/planetmapper/badge.svg?branch=main)](https://coveralls.io/github/ortk95/planetmapper?branch=main)
[![JOSS Paper DOI](https://joss.theoj.org/papers/10.21105/joss.05728/status.svg)](https://doi.org/10.21105/joss.05728)

PlanetMapper is an open source Python package for visualising, navigating and mapping Solar System observations.

## [Documentation](https://planetmapper.readthedocs.io)
For full documentation and [API reference](https://planetmapper.readthedocs.io/en/latest/documentation.html), visit [planetmapper.readthedocs.io](https://planetmapper.readthedocs.io/en/latest/index.html).


## [Installation](https://planetmapper.readthedocs.io/en/latest/installation.html)
```bash
pip install planetmapper --upgrade
```

```bash
conda install -c conda-forge planetmapper
```

_Requires Python 3.10+_


## Citing PlanetMapper
If you use PlanetMapper in your research, please cite the following paper:

> King et al., (2023). PlanetMapper: A Python package for visualising, navigating and mapping Solar System observations. Journal of Open Source Software, 8(90), 5728, https://doi.org/10.21105/joss.05728

<details>
<summary>Citation BibTeX entry</summary>

```bibtex
@article{king_2023_planetmapper,
  author  = {King, Oliver R. T. and Fletcher, Leigh N.},
  doi     = {10.21105/joss.05728},
  journal = {Journal of Open Source Software},
  month   = oct,
  number  = {90},
  pages   = {5728},
  title   = {{PlanetMapper: A Python package for visualising, navigating and mapping Solar System observations}},
  url     = {https://joss.theoj.org/papers/10.21105/joss.05728},
  volume  = {8},
  year    = {2023}
}
```

</details>

Each PlanetMapper version is also archived on Zenodo at [doi.org/10.5281/zenodo.7963121](https://doi.org/10.5281/zenodo.7963121).


## Key features
### [Fit and map astronomical observations using a full featured user interface](https://planetmapper.readthedocs.io/en/latest/user_interface.html)
[![Screenshot of the PlanetMapper graphical user interface showing an observation of Europa being navigated](docs/images/gui_fitting.png)](https://planetmapper.readthedocs.io/en/latest/user_interface.html)

### [Easily visualise solar system observations with just a few lines of code](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

```python
body = planetmapper.Body('saturn', '2020-01-01')
body.plot_wireframe_radec()
plt.show()
```

[![Image of Saturn generated with PlanetMapper showing the orientation of Saturn and its rings](docs/images/saturn_wireframe_radec.png)](https://planetmapper.readthedocs.io/en/latest/general_python_api.html#wireframe-plots)

### [Convert coordinates, generate backplanes and project maps of telescope observations](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)
[![Plot of a mapped Jupiter observation, generated with PlanetMapper, showing observed and mapped versions of the Jupiter data](docs/images/jupiter_mapped.png)](https://planetmapper.readthedocs.io/en/latest/general_python_api.html)


## Contributing

If you spot a bug, or want contribute code to PlanetMapper, check out the [contributing guidelines](https://github.com/ortk95/planetmapper/blob/main/CONTRIBUTING.md).

## Help and support

If you have any questions, suggestions or feedback, please [visit our support page and get in touch](https://planetmapper.readthedocs.io/en/latest/help.html)!