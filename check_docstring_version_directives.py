#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check docstrings for any version directives that have a missing version.

This allows to e.g. add `.. versionadded:: ?.?.?` to the docstring of a new function
when it is written, even if the next version number is currently unknown. This script
will then catch the missing version directive and remind the developer to update the
version directive before releasing a new version.
"""

import sys
from pathlib import Path
from typing import Sequence

from packaging.version import InvalidVersion, Version

VERSION_DIRECTIVES: tuple[str, ...] = (
    'versionadded',
    'versionchanged',
    'versionremoved',
    'deprecated',
)
PLANETMAPPER_ROOT = Path(__file__).parent / 'planetmapper'


def main() -> None:
    found_errors = check_planetmapper_source()
    if found_errors > 0:
        sys.exit(1)


def check_planetmapper_source() -> int:
    paths = sorted(PLANETMAPPER_ROOT.rglob('*.py'))
    print(f'Checking {len(paths)} files for invalid versions in {PLANETMAPPER_ROOT}')
    found_errors = 0
    for path in paths:
        found_errors += check_docstring_version_directives(path)
    if found_errors:
        print()
        print(f'Found {found_errors} invalid version directives')
    else:
        print('No invalid version directives found')
    return found_errors


def check_docstring_version_directives(path: Path) -> int:
    found_errors = 0
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        maybe_version = maybe_get_directive_version(line)
        if maybe_version is not None:
            try:
                Version(maybe_version)
            except InvalidVersion:
                found_errors += 1
                print_error(path, i, lines, maybe_version)
    return found_errors


def maybe_get_directive_version(line: str) -> str | None:
    line = line.strip()
    for directive in VERSION_DIRECTIVES:
        if line.startswith(f'.. {directive}::'):
            return line.split('::', 1)[1].strip()
    return None


def print_error(
    path: Path,
    i: int,
    lines: Sequence[str],
    maybe_version: str,
    *,
    context_lines: int = 2,
) -> None:
    print()
    print(f'{path}:{i + 1} InvalidVersion: {maybe_version!r}')
    for j in range(i - context_lines, i + context_lines + 1):
        if 0 <= j < len(lines):
            print(f'{j + 1:4d} {lines[j].rstrip()}')


if __name__ == '__main__':
    main(*sys.argv[1:])
