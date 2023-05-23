import os
import sys

import setuptools

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root, 'planetmapper'))
# pylint: disable-next=import-error
import common  # type: ignore

with open(os.path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='planetmapper',
    version=common.__version__,
    author=common.__author__,
    author_email='oliver.king95@gmail.com',
    description='A Python module for visualising, navigating and mapping Solar System observations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=common.__url__,
    download_url='https://pypi.org/project/planetmapper/',
    packages=['planetmapper'],
    package_dir={'planetmapper': 'planetmapper'},
    package_data={'planetmapper': ['data/*.json']},
    include_package_data=True,
    project_urls={
        'Documentation': 'https://planetmapper.readthedocs.io/',
        'GitHub': common.__url__,
    },
    entry_points={'console_scripts': ['planetmapper=planetmapper:main']},
    python_requires='>=3.10.0',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
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
    ],
)
