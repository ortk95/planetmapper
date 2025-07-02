#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Python files for TODO comments.

This checks for TODO comments in the planetmapper source code and tests, then prints
the found comments.
"""
import sys
from pathlib import Path
from typing import Sequence

MESSAGES: tuple[str, ...] = (
    '# TODO',
    '# FIXME',
    '# XXX',
)
ROOT_DIRECTORIES: list[Path] = [
    Path(__file__).parent / 'planetmapper',
    Path(__file__).parent / 'tests',
]


def main() -> None:
    found_todos = check_files()
    if found_todos > 0:
        sys.exit(1)


def check_files() -> int:
    paths = get_file_paths()
    print(
        f'Checking {len(paths)} files for TODO comments in {", ".join(str(root) for root in ROOT_DIRECTORIES)}',
    )
    found_todos = 0
    for path in paths:
        found_todos += check_file_for_todos(path)
    if found_todos:
        print(f'Found {found_todos} TODO comments')
    else:
        print('No TODO comments found')
    return found_todos


def get_file_paths() -> Sequence[Path]:
    """Get all Python file paths in the specified root directories."""
    file_paths = []
    for root in ROOT_DIRECTORIES:
        file_paths.extend(root.rglob('*.py'))
    return sorted(file_paths)


def check_file_for_todos(path: Path) -> int:
    found_todos = 0
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if check_line_for_todos(line):
            found_todos += 1
            print_todo(path, i, lines)
    return found_todos


def check_line_for_todos(line: str) -> bool:
    line = line.strip()
    return any(line.startswith(message) for message in MESSAGES)


def print_todo(
    path: Path,
    i: int,
    lines: Sequence[str],
) -> None:
    print(f'{path}:{i + 1}: {lines[i].strip()}')


if __name__ == '__main__':
    main(*sys.argv[1:])
