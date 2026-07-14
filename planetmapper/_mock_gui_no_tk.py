# pylint: disable=unused-argument

from typing import Callable, NoReturn

ERROR_MESSAGE = (
    'The "tkinter" package is not included in your Python installation, so PlanetMapper cannot create a graphical user interface. '
    'See https://docs.python.org/3/library/tkinter.html for more information.'
)


def raise_tkinter_import_error(parent_exception: ImportError) -> NoReturn:
    """
    Inform the user that tkinter is not available when they try to use the GUI.
    """
    if (
        isinstance(parent_exception, ImportError)
        and parent_exception.name is not None
        and 'tkinter' in parent_exception.name
    ):
        raise ModuleNotFoundError(ERROR_MESSAGE, name='tkinter') from parent_exception
    raise parent_exception


class _MockGUIModuleClass:
    """
    A basic mock class to use in replacing the planetmapper.gui module when tkinter is
    not available. This class will raise an ImportError when any of its attributes are
    accessed, with a message explaining that tkinter is not available. This allows
    users to still import the full planetmapper package, and use 90% of its
    functionality, and then get a useful error message if they try and use any GUI
    functionality.
    """

    def __init__(self, parent_exception: ImportError) -> None:
        self._parent_exception = parent_exception

    def __getattr__(self, name) -> NoReturn:
        raise_tkinter_import_error(self._parent_exception)


def get_mocks(
    parent_exception: ImportError,
) -> tuple[_MockGUIModuleClass, Callable[..., NoReturn]]:
    """
    Return a tuple of the mock gui module and the run_gui function. This is used in
    planetmapper/__init__.py to set the gui and run_gui attributes when tkinter is not
    available.
    """

    def run_gui(*args, **kwargs) -> NoReturn:
        """Basic copy of planetmapper.gui.run_gui that will raise an error when used."""
        raise_tkinter_import_error(parent_exception)

    return _MockGUIModuleClass(parent_exception), run_gui
