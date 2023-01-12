# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
import sphinx_rtd_theme

sys.path.insert(0, os.path.join(os.path.split(__file__)[0], '..'))
import planetmapper
from planetmapper.common import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PlanetMapper'
copyright = '2022, Oliver King'
author = 'Oliver King'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

default_role = 'code'
master_doc = 'index'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'spiceypy': ('https://spiceypy.readthedocs.io/en/latest', None),
}

# Autodoc
autodoc_member_order = 'bysource'
# autoclass_content = 'both'
# autodoc_typehints = 'both'
# autodoc_typehints_description_target = 'documented_params'
autodoc_inherit_docstrings = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Dynamically generate backplane documentation
from planetmapper.body_xy import _make_backplane_documentation_str

p = os.path.join(os.path.split(__file__)[0], 'default_backplanes.rst')
with open(p, 'w') as f:
    print('Updating default backplanes')
    s = _make_backplane_documentation_str()
    f.write(s)
