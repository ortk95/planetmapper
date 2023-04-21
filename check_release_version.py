#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check GitHub tag version is equal to planetmapper.__version__.
"""
import os
import sys
root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root, 'planetmapper'))
import common  #  type: ignore


def main(tag_version: str):
    planetmapper_version = 'v' + common.__version__

    print('tag_version = {!r}'.format(tag_version))
    print(
        'planetmapper.common.__version__ = {!r} ({!r})'.format(
            common.__version__, planetmapper_version
        )
    )
    if tag_version == planetmapper_version:
        print('Versions match')
    else:
        print(
            'Version mismatch: {!r} != {!r}'.format(tag_version, planetmapper_version)
        )
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1])
