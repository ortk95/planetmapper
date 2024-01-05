"""
TODO
"""

import argparse

import common

# XXX add docstring
# XXX change entry points (PyPI & conda-forge) to use this function & test
# XXX add tests


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
        version=f'PlanetMapper {common.__version__}',
        help='print the version number and exit',
    )
    return parser


def _run_gui(file_path: str | None) -> None:
    print(f'Launching PlanetMapper {common.__version__}', flush=True)

    # Defer importing main planetmapper module until here to improve apparent speed
    # of startup (i.e. the 'Launching PlanetMapper ...' message should be printed before
    # the relatively slow importing of the various modules).
    from . import gui

    gui._run_gui_from_cli(file_path, printed_launching_message=True)


def main() -> None:
    """:meta private:"""
    args = _get_parser().parse_args()
    _run_gui(args.file_path)


main()  # XXX remove this
