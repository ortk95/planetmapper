#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to help downloading spice kernels.

Will download local copy of kernels with same directory structure as on 
https://naif.jpl.nasa.gov/. Use :func:`planetmapper.set_kernel_path` to choose the
location that the kernels are downloaded to.

These functions can be used to download a set of URLS. For example: ::

    from planetmapper.kernel_downloader import download_urls

    # Download all kernel files in generic_kernels/pck
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/')

    # Download specific kernel file
    download_urls('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls')

    # Download multiple sets of kernel files
    download_urls(
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/',
    )

"""
import os
import urllib.parse
import urllib.request

import tqdm

from . import utils
from .base import get_kernel_path

URL_ROOT = 'https://naif.jpl.nasa.gov/pub/'


def download_urls(*urls: str) -> None:
    """
    Download data from naif.jpl.nasa.gov and save locally.

    urls can either be a the url of a single kernel, or the index page containing
    multiple kernels.

    If a single kernel, download the kernel using download_kernel().

    If an index page, download all first-level files using
    :func:`download_kernels_from_webpage`.

    Args:
        urls: kernel URL on naif.jpl.nasa.gov.
    """
    for url in urls:
        # look for '.' in filename part of url to identify if a file/directory
        path = urllib.parse.urlsplit(url).path
        if '.' in os.path.split(path)[1]:
            download_kernel(url)
        else:
            download_kernels_from_webpage(url)


def download_kernels_from_webpage(index_url: str) -> None:
    """
    Download all first-level kernels listed in the page given by index_url.

    URL must be on https://naif.jpl.nasa.gov/pub/. This will break if JPL changes the
    format of the webpage.

    .. warning ::

        This function will only download kernels found immediately on `index_url`.
        Kernels in nested folders must therefore be downloaded manually.

    Args:
        index_url: URL of index page on naif.jpl.nasa.gov.
    """
    urls = get_kernel_paths_from_webpage(index_url)
    print(f'{len(urls)} to download from {index_url}')
    for idx, url in enumerate(urls):
        download_kernel(url, note=f'[{idx+1}/{len(urls)}] ')
    print(f'All kernels downloaded from {index_url}')
    print()


def download_kernel(url: str, force_download: bool = False, note: str = '') -> None:
    """
    Download single kernel given by url.

    URL must be on https://naif.jpl.nasa.gov/pub/. By default will only download file if
    if does not already exist locally. Set `force_download=True` to override this check
    and download the file even if it already exists locally.

    Args:
        url: URL of kernel on naif.jpl.nasa.gov.
        force_download: toggle overwriting already downloaded kernels.
        note: string to include in progress message.
    """
    kp = _get_kernel_path(url)
    print(f'{note}Checking {kp}')
    if _check_kernel_exists_locally(url):
        if force_download:
            print('  Kernel already exists, downloading anyway')
        else:
            print('  OK - Kernel already exists locally')
            return
    local_path = _convert_url_to_local_path(url)
    print(f'  Downloading to {local_path}')
    download_file(url, local_path)
    print('    Done')


def get_kernel_paths_from_webpage(index_url: str) -> list[str]:
    """
    Get list of kernel urls from an index page on https://naif.jpl.nasa.gov/pub/.

    This is a bit of a hack and will break if JPL changes the format of the webpage.

    Args:
        index_url: URL of webpage.

    Returns:
        List of URL strings corresponding to kernels on the webpage.
    """
    # pylint: disable=consider-using-with
    assert index_url.startswith(URL_ROOT), f'URL must begin with {URL_ROOT}'
    webpage = urllib.request.urlopen(index_url).read().decode()
    data = webpage.split('<!--start data_content-->')[1].split('</table>')[0]
    lines = data.splitlines()  # get lines from table
    paths = []
    for l in lines:
        if not l.startswith('<img src="/icons/'):
            continue  # ignore irrelevant lines from table
        href = l.split('<a href="')[1].split('"')[0]  # find links from table
        if '.' in href:
            p = index_url + '/' + href  # create url from link
            paths.append(p)
    return paths


def _check_kernel_exists_locally(url: str) -> bool:
    """Test if kernel file already exists on local filesystem."""
    local_path = _convert_url_to_local_path(url)
    return os.path.exists(local_path)


def _convert_url_to_local_path(url: str) -> str:
    """Convert a url on https://naif.jpl.nasa.gov to the equavilent local path."""
    assert url.startswith(URL_ROOT), f'URL must begin with {URL_ROOT}'
    kp = _get_kernel_path(url)
    return _kernel_path_to_local_path(kp)


def _standardise_path(p: str) -> str:
    """Make a standardised version of path."""
    return os.path.normpath(os.path.expanduser(p))


def _get_kernel_path(p: str) -> str:
    """
    Get the useful part of the path from a URL/local filepath.

    For example both
    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/'
    and '~/spice/naif/generic_kernels/spk/satellites/' are converted into
    'naif/generic_kernels/spk/satellites'
    """
    p = _standardise_path(p)
    for prefix in (URL_ROOT, get_kernel_path()):
        prefix = _standardise_path(prefix)
        if p.startswith(prefix):
            return _standardise_path(os.path.relpath(p, prefix))
    raise ValueError('Cannot get kernel path from "{}"'.format(p))


def _kernel_path_to_url(kp: str) -> str:
    """Create URL from a kernel path"""
    return URL_ROOT + os.path.sep + kp


def _kernel_path_to_local_path(kp: str) -> str:
    """Create a local path from a kernel path"""
    return _standardise_path(get_kernel_path() + os.path.sep + kp)


def download_file(url: str, local_path: str) -> None:
    """
    Download kernel file to local system.

    Args:
        url: URL of kernel file.
        local_path: File path to save kernel file on local system.
    """
    utils.check_path(local_path)

    # download to temp file so don't get issues from partial downloads being killed
    temp_path = local_path + '.temp'
    urllib.request.urlretrieve(url, temp_path, reporthook=_DownloadProgressBar())

    # once fully downloaded, we can safely move the temp file to the desired path
    os.replace(temp_path, local_path)


class _DownloadProgressBar:
    """
    Shows download progress with tqdm
    """

    def __init__(self):
        self.pbar = None
        self.previous_downloaded = 0

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm.tqdm(
                total=total_size, unit_scale=True, unit='B', unit_divisor=1024
            )
        downloaded = block_num * block_size
        change = downloaded - self.previous_downloaded
        self.previous_downloaded = downloaded
        self.pbar.update(change)
