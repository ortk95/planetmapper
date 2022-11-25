import os
import setuptools
import planetmapper.common

root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'r') as f:
    long_description = f.read()

with open(os.path.join(root, 'requirements.txt'), 'r') as f:
    requirements = [l.strip() for l in f.readlines()]
    requirements = [l for l in requirements if l]

setuptools.setup(
    name='planetmapper',
    version=planetmapper.common.__version__,
    author=planetmapper.common.__author__,
    author_email='oliver.king95@gmail.com',
    description='A Python module for navigating and mapping Solar System observations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=planetmapper.common.__url__,
    download_url='https://pypi.org/project/planetmapper/',
    install_requires=requirements,
    python_requires='>=3.10.0',
    packages=['planetmapper'],
    package_dir={'planetmapper': 'planetmapper'},
    package_data={'planetmapper': ['data/*.json']},
    include_package_data=True,
    project_urls={
        'Documentation': 'https://planetmapper.readthedocs.io/',
        'GitHub': planetmapper.common.__url__,
    },
)
