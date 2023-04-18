#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check GitHub tag version is equal to planetmapper.__version__.
"""
import sys
import planetmapper


def main(tag_version: str):
    planetmapper_version = 'v' + planetmapper.__version__

    print('tag_version = {!r}'.format(tag_version))
    print(
        'planetmapper.__version__ = {!r} ({!r})'.format(
            planetmapper.__version__, planetmapper_version
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
