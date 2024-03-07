import os
import sys

import setuptools

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root, 'planetmapper'))
# pylint: disable-next=import-error
import common  # type: ignore

with open(os.path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Replace relative image links with absolute links to GitHub hosted images so that
# images display properly on PyPI
long_description = long_description.replace(
    '](docs/images/',
    '](https://raw.githubusercontent.com/ortk95/planetmapper/main/docs/images/',
)

setuptools.setup(
    name='planetmapper',
    version=common.__version__,
    author=common.__author__,
    author_email='oliver.king95@gmail.com',
    description=common.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=common.__url__,
    license=common.__license__,
    download_url='https://pypi.org/project/planetmapper/',
    packages=['planetmapper'],
    package_dir={'planetmapper': 'planetmapper'},
    package_data={'planetmapper': ['data/*.json', 'py.typed']},
    include_package_data=True,
    project_urls={
        'Documentation': 'https://planetmapper.readthedocs.io/',
        'GitHub': common.__url__,
        'Paper': 'https://doi.org/10.21105/joss.05728',
        'conda-forge': 'https://anaconda.org/conda-forge/planetmapper',
    },
    entry_points={
        # Copy any changes here to the conda-forge recipe (meta.yaml)
        # https://github.com/conda-forge/planetmapper-feedstock/
        'console_scripts': ['planetmapper=planetmapper.cli:main'],
    },
    python_requires='>=3.10.0',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy<2.0',
        'Pillow',
        'spiceypy',
        'scipy',
        'photutils',
        'tqdm',
        'pyproj',
        'typing-extensions',
    ],
    keywords=[
        'planetmapper',
        'astronomy',
        'space',
        'science',
        'spice',
        'ephemeris',
        'planetary-science',
        'geometry',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Physics',
        'Framework :: Matplotlib',
    ],
)
