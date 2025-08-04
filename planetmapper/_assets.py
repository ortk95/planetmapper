import os


def make_asset_path(filename: str) -> str:
    """
    Generates a path to a static asset file.

    Args:
        filename: Filename of the asset file stored in `planetmapper/assets`

    Returns:
        Absolute path to the asset file.
    """
    asset_dir = os.path.join(os.path.split(__file__)[0], 'assets')
    return os.path.join(asset_dir, filename)


def get_gui_icon_path() -> str:
    """
    Returns the path to the GUI icon image.

    Returns:
        Absolute path to the GUI icon image.
    """
    return make_asset_path('gui_icon.png')
