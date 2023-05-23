import copy
import functools
import json
import os


def make_data_path(filename: str) -> str:
    """
    Generates a path to a static data file.

    Args:
        filename: Filename of the data file stored in `planetmapper/data`

    Returns:
        Absolute path to the data file.
    """
    data_dir = os.path.join(os.path.split(__file__)[0], 'data')
    return os.path.join(data_dir, filename)


def get_ring_radii() -> dict[str, dict[str, list[float]]]:
    """
    Load planetary ring radii from data file.

    These ring radii values are sourced from
    https://nssdc.gsfc.nasa.gov/planetary/planetfact.html.

    Returns:
        Dictionary where the keys are planet names and the values are dictionaries
        containing ring data. These ring data dictionaries have keys corresponding to
        the names of the rings, and values with a list of ring radii in km. If the
        length of this list is 2, then the values give the inner and outer radii of the
        ring respectively. Otherwise, the length should be 1, meaning the ring has a
        single radius.
    """
    return copy.deepcopy(_get_ring_radii_data())


@functools.cache
def _get_ring_radii_data() -> dict[str, dict[str, list[float]]]:
    with open(make_data_path('rings.json'), encoding='utf-8') as f:
        return json.load(f)


def get_ring_aliases() -> dict[str, str]:
    """
    Load ring aliases from data file.

    These are used to allow pure ASCII ring names to be used in functions such as
    :func:`planetmapper.Body.add_named_rings`.

    Returns:
        Dictionary where the keys are variants of ring names (e.g. `liberte`) and the
        values are the ring names (e.g. `liberté`) in a format consistent with the
        ring names in :func:`get_ring_radii`. Note that the keys and values are all in
        lower case.
    """
    return copy.deepcopy(_get_ring_aliases_data())


@functools.cache
def _get_ring_aliases_data() -> dict[str, str]:
    with open(make_data_path('ring_aliases.json'), encoding='utf-8') as f:
        return json.load(f)
