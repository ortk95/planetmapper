# pylint: disable=unused-argument

from typing import NoReturn

ERROR_MESSAGE = (
    'The "tkinter" package is not included in your Python installation, so PlanetMapper cannot create a graphical user interface. '
    'See https://docs.python.org/3/library/tkinter.html for more information.'
)
PARENT_EXCEPTION: ImportError = ImportError()


def raise_tkinter_import_error(parent_exception: ImportError | None = None) -> NoReturn:
    """
    Inform the user that tkinter is not available when they try to use the GUI.
    """
    parent_exception = (
        PARENT_EXCEPTION if parent_exception is None else parent_exception
    )
    if PARENT_EXCEPTION.name is not None and 'tkinter' in PARENT_EXCEPTION.name:
        raise ModuleNotFoundError(ERROR_MESSAGE, name='tkinter') from PARENT_EXCEPTION
    raise PARENT_EXCEPTION  # pragma: no cover


class mock_gui_module_class:
    """
    A basic mock class to use in replacing the planetmapper.gui module when tkinter is
    not available. This class will raise an ImportError when any of its attributes are
    accessed, with a message explaining that tkinter is not available. This allows
    users to still import the full planetmapper package, and use 90% of its
    functionality, and then get a useful error message if they try and use any GUI
    functionality.
    """

    def __getattr__(self, name) -> NoReturn:
        raise_tkinter_import_error()


def run_gui(*args, **kwargs) -> NoReturn:
    """Basic copy of planetmapper.gui.run_gui that will raise an error when used."""
    raise_tkinter_import_error()
