from typing import NoReturn

_ERROR_MESSAGE = (
    'The "tkinter" package is not included in your Python installation, so PlanetMapper cannot create any graphical user interfaces. '
    'See https://docs.python.org/3/library/tkinter.html for more information.'
)
_PARENT_EXCEPTION: Exception | None = None


def raise_tkinter_import_error() -> NoReturn:
    """
    Inform the user that tkinter is not available when they try to use the GUI.s
    """
    raise ImportError(_ERROR_MESSAGE, name='tkinter') from _PARENT_EXCEPTION


class _mock_gui_module_class:
    """
    A basic mock class to use in replacing the planermapper.gui module when tkinter is
    not available. This class will raise an ImportError when any of its attributes are
    accessed, with a message explaining that tkinter is not available. This allows
    users to still import the full planetmapper package, and use 90% of its
    functionality, and then get a useful error message if they try and use any GUI
    functionality.
    """

    def __getattr__(self, name):
        raise_tkinter_import_error()


def run_gui(*args, **kwargs):
    """Basic copy of planetmapper.gui.run_gui that will raise an error when used."""
    raise_tkinter_import_error()
