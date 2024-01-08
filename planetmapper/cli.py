"""
This module is the entry point for the PlanetMapper Command Line Interface (CLI).

The command line interface is generally used to launch the PlanetMapper Graphical User
Interface (GUI) without needing to write any Python code. For example, simply running
`planetmapper` in the command line will launch the GUI.

For a list of available command line options, run `planetmapper --help` in the command
line.
"""

import argparse

# Defer main planetmapper imports until they are needed to improve speed of CLI.
# e.g. `planetmapper --version` should print the version number without importing
# the rest of the package, and the 'Launching PlanetMapper ...' message should be
# printed before the main imports start (so the user gets some immediate feedback).
# TODO this doesn't seem to actually work

# XXX change entry points (PyPI & conda-forge) to use this function & test
# XXX add tests


def main() -> None:
    """
    Entry point for CLI.

    :meta private:
    """

    args = _get_parser().parse_args()
    _run_gui(args.file_path)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='planetmapper',
        description="""
PlanetMapper: An open source Python module for visualising, navigating and mapping Solar
System observations. See https://planetmapper.readthedocs.io for full documentation.
""",
        epilog='If no arguments are provided, the PlanetMapper GUI will be launched.',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        'file_path',
        nargs='?',
        type=str,
        help='launch the PlanetMapper GUI with the specified FITS file open',
        default=None,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'PlanetMapper {_get_version()}',
        help='print the version number and exit',
    )
    return parser


def _run_gui(file_path: str | None) -> None:
    print(f'Launching PlanetMapper {_get_version()}', flush=True)

    from . import gui

    gui._run_gui_from_cli(file_path)


def _get_version() -> str:
    from . import common

    return common.__version__
