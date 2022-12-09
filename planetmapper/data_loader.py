import os
import json
import functools


def make_data_path(filename: str) -> str:
    """
    Generates a path to a static data file.

    Args:
        filename: Filename of the data file stored in planetmapper/data

    Returns:
        Absolute path to the data file.
    """
    data_dir = os.path.join(os.path.split(__file__)[0], 'data')
    return os.path.join(data_dir, filename)


@functools.lru_cache
def get_ring_radii() -> dict[str, dict[str, list[float]]]:
    """
    Load planetary ring radii from data file.

    These ring radii values are sourced from
    https://nssdc.gsfc.nasa.gov/planetary/planetfact.html.

    Returns:
        Dictionary where the keys are planet names and the values are dictionaries
        contianing ring data. These ring data dictionaries have keys corresponding to
        the names of the rings, and values with a list of ring radii in km. If the
        length of this list is 2, then the values give the inner and outer radii of the
        ring respectively. Otherwise, the length should be 1, meaning the ring has a 
        single radius.
    """
    with open(make_data_path('rings.json'), 'r') as f:
        return json.load(f)
