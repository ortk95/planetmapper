import os
import json
import functools


def make_data_path(filename: str) -> str:
    data_dir = os.path.join(os.path.split(__file__)[0], 'data')
    return os.path.join(data_dir, filename)


@functools.lru_cache
def get_ring_radii():
    with open(make_data_path('rings.json'), 'r') as f:
        return json.load(f)
