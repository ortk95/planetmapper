import datetime
import glob
import math
import os
import sys
from typing import Callable, Iterable, TypeVar, ParamSpec, NamedTuple, cast, Any
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.patches
from matplotlib.transforms import Transform
from matplotlib.axes import Axes
import numpy as np
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError
from functools import wraps
import PIL.Image
from astropy.io import fits
from . import utils
from . import common
from .body_xy import BodyXY


class Observation(BodyXY):
    """
    Class representing an actual observation of an astronomical body at a specific time.

    This is a subclass of :class:`BodyXY`, with additional methods to interact with the
    observed data, such as by saving a FITS file containing calculated backplane data.
    All methods described in :class:`BodyXY`, :class:`Body` and :class:`PlanetMapperTool` are
    therefore available in instances of this class.

    This class can be created by either providing a `path` to a data file to be loaded,
    or by directly providing the `data` itself  (and optionally a FITS `header`). The
    `nx` and `ny` values for :class:`BodyXY` will automatically be calculated from the
    input data.

    If the input data is a FITS file (or a `header` is specified), then this class will
    attempt to generate appropriate parameters (e.g. `target`, `utc`) by using the
    values in the header. This allows an instance of this class to be created with a
    single argument specifying the `path` to the FITS file e.g.
    `Observation('path/to/file.fits')`. Manually specified parameters will take
    precedence, so `Observation('path/to/file.fits', target='JUPITER')` will have
    Jupiter as a target, regardless of any values saying otherwise in the FITS header.

    If a FITS header is not provided (e.g. if the input path corresponds to an image
    file), then at least the `target` and `utc` parameters need to be specified.

    Args:
        path: Path to data file to load. If this is `None` then `data` must be specified
            instead.
        data: Array containing observation data to use instead of loading the data from
            `path`. This should only be provided if `path` is None.
        header: FITS header which corresponds to the provided `data`. This is optional
            and should only be provided if `path` is None.
        target: Name of target body, passed to :class:`Body`. If this is unspecified,
            then the target will be derived from the values in the FITS header.
        utc: Time of observation, passed to :class:`Body`. If this is unspecified,
            then the time will be derived from the values in the FITS header.
        **kwargs: Additional parameters are passed to :class:`BodyXY`. These can be used
            to specify additional parameters such as`observer`.
    """

    FITS_FILE_EXTENSIONS = ('.fits', '.fits.gz')
    """File extensions which will be read as FITS files."""
    IMAGE_FILE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    """File extensions which will be read as image files."""
    FITS_KEYWORD = 'PLANMAP'
    """FITS keyword used in metadata added to header of output FITS files."""

    def __init__(
        self,
        path: str | None = None,
        *,
        data: np.ndarray | None = None,
        header: fits.Header | None = None,
        target: str | None = None,
        utc: str | datetime.datetime | None = None,
        **kwargs,
    ) -> None:
        # Add docstrings
        self.data: np.ndarray
        """Observed data."""
        self.header: fits.Header
        """
        FITS header containing data about the observation. If this is not provided, then
        a basic header will be produced containing data derived from the `target` and 
        `utc` parameters.
        """

        self.path = path
        self.header = None  # type: ignore

        # Add optional kw to kwargs if specified
        if target is not None:
            kwargs['target'] = target
        if utc is not None:
            kwargs['utc'] = utc

        # TODO add warning about header being modified in place? Or copy header?
        if self.path is None:
            if data is None:
                raise ValueError('Either `path` or `data` must be provided')
            self.data = data
        else:
            # TODO should we have a way to provide both path and header for e.g. JPEGS?
            if data is not None:
                raise ValueError('`path` and `data` are mutually exclusive')
            if header is not None:
                raise ValueError('`path` and `header` are mutually exclusive')
            self._load_data_from_path()

        # TODO validate/standardise shape of data here (cube etc.)
        self.data = np.asarray(self.data)
        if self.header is not None:
            # use values from header to fill in arguments (e.g. target) which aren't
            # specified by the user
            self._add_kw_from_header(kwargs)
        super().__init__(nx=self.data.shape[2], ny=self.data.shape[1], **kwargs)

        if self.header is None:
            self.header = fits.Header(
                {
                    'OBJECT': self.target,
                    'DATE-OBS': self.utc,
                }
            )
        self.centre_disc()

    def _load_data_from_path(self):
        assert self.path is not None
        if any(self.path.endswith(ext) for ext in self.FITS_FILE_EXTENSIONS):
            self._load_fits_data()
        elif any(self.path.endswith(ext) for ext in self.IMAGE_FILE_EXTENSIONS):
            self._load_image_data()
        else:
            raise ValueError(f'Unexpected file type for {self.path!r}')

    def _load_fits_data(self):
        assert self.path is not None
        self.data, self.header = fits.getdata(self.path, header=True)  #  type: ignore
        # TODO add check data is a cube

    def _load_image_data(self):
        assert self.path is not None
        image = np.array(PIL.Image.open(self.path))

        if len(image.shape) == 2:
            # If greyscale image, add another dimension so that it is an image cube with
            # a single frame. This will ensure that data will always be a cube.
            image = np.array([image])
        else:
            # If RGB image, change the axis order so wavelength is the first axis (i.e.
            # consistent with FITS)
            image = np.moveaxis(image, 2, 0)
        self.data = image

    def _add_kw_from_header(self, kw: dict):
        # fill in kwargs with values from header (if they aren't specified by the user)
        # TODO deal with more FITS files (e.g. DATE-OBS doesn't work for JWST)
        # TODO deal with missing values
        kw.setdefault('target', self.header['OBJECT'])
        kw.setdefault('utc', self.header['DATE-OBS'])

    def __repr__(self) -> str:
        return f'Observation({self.path!r})'  # TODO make more explicit?

    # Auto disc id
    def centre_disc(self) -> None:
        """
        Centre the target's planetary disc and make it fill ~90% of the observation.

        This adjusts `x0` and `y0` so that they lie in the centre of the image, and `r0`
        is adjusted so that the disc fills 90% of the shortest side of the image. For
        example, if `nx = 20` and `ny = 30`, then `x0` will be set to 10, `y0` will be
        set to 15 and `r0` will be set to 9. The rotation of the disc is unchanged.
        """
        self.set_x0(self._nx / 2)
        self.set_y0(self._ny / 2)
        self.set_r0(0.9 * (min(self.get_x0(), self.get_y0())))
        self.set_disc_method('centre_disc')

    # Output
    def append_to_header(
        self,
        keyword: str,
        value: str | float | bool | complex,
        comment: str | None = None,
        hierarch_keyword: bool = True,
    ):
        """
        Add a card to the FITS :attr:`header`. This is mainly used to record metadata
        which is then saved in the header of any FITS files saved by :func:`save`. By
        default, the keyword is modified to provide a consistent keyword prefix for all
        header cards added by this routine.

        Args:
            keyword: Card keyword.
            value: Card value.
            comment: Card comment. If unspecified not comment will be added.
            hierarch_keyword: Toggle adding the keyword prefix from :attr:`FITS_KEYWORD`
                to the keyword.
        """
        if hierarch_keyword:
            keyword = f'HIERARCH {self.FITS_KEYWORD} {keyword}'
        with utils.filter_fits_comment_warning():
            self.header.append(fits.Card(keyword=keyword, value=value, comment=comment))

    def add_header_metadata(self):
        """
        Add automatically generated metadata to :attr:`header`. This is automatically
        called by :func:`save` so `add_header_metadata` does not normally need to be
        called manually.
        """
        self.append_to_header('VERSION', common.__version__, 'Planet Mapper version.')
        self.append_to_header('URL', common.__url__, 'Webpage.')
        self.append_to_header(
            'DATE',
            datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'File generation datetime.',
        )
        if self.path is not None:
            self.append_to_header(
                'INFILE',
                os.path.split(self.path)[1],
                'Input file name.',
            )
        self.append_to_header(
            'DISC X0', self.get_x0(), '[pixels] x coordinate of disc centre.'
        )
        self.append_to_header(
            'DISC Y0', self.get_y0(), '[pixels] y coordinate of disc centre.'
        )
        self.append_to_header(
            'DISC R0', self.get_r0(), '[pixels] equatorial radius of disc.'
        )
        self.append_to_header(
            'DISC ROT', self.get_rotation(), '[degrees] rotation of disc.'
        )
        self.append_to_header(
            'DISC METHOD', self.get_disc_method(), 'Method used to find disc.'
        )
        self.append_to_header(
            'ET-OBS', self.et, 'J2000 ephemeris seconds of observation.'
        )
        self.append_to_header(
            'TARGET',
            self.target,
            'Target body name used in SPICE.',
        )
        self.append_to_header(
            'TARGET-ID', self.target_body_id, 'Target body ID from SPICE.'
        )
        self.append_to_header(
            'R EQ', self.r_eq, '[km] Target equatorial radius from SPICE.'
        )
        self.append_to_header(
            'R POLAR', self.r_polar, '[km] Target polar radius from SPICE.'
        )
        self.append_to_header(
            'LIGHT-TIME',
            self.target_light_time,
            '[seconds] Light time to target from SPICE.',
        )
        self.append_to_header(
            'DISTANCE', self.target_distance, '[km] Distance to target from SPICE.'
        )
        self.append_to_header(
            'OBSERVER',
            self.observer,
            'Observer name used in SPICE.',
        )
        self.append_to_header(
            'TARGET-FRAME',
            self.target_frame,
            'Target frame used in SPICE.',
        )
        self.append_to_header(
            'OBSERVER-FRAME',
            self.observer_frame,
            'Observer frame used in SPICE.',
        )
        self.append_to_header(
            'ILLUMINATION',
            self.illumination_source,
            'Illumination source used in SPICE.',
        )
        self.append_to_header(
            'ABCORR', self.aberration_correction, 'Aberration correction used in SPICE.'
        )
        self.append_to_header(
            'SUBPOINT-METHOD', self.subpoint_method, 'Subpoint method used in SPICE.'
        )
        self.append_to_header(
            'SURFACE-METHOD',
            self.surface_method,
            'Surface intercept method used in SPICE.',
        )
        self.append_to_header(
            'OPTIMIZATION-USED', self._optimize_speed, 'Speed optimizations used.'
        )

    def save(self, path: str) -> None:
        """
        Save a FITS file containing the observed data and generated backplanes.

        The primary HDU in the FITS file will be the :attr:`data` and :attr:`header`
        of the observed data, with appropriate metadata automatically added to the
        header by :func:`add_header_metadata`. The backplanes are generated from all the
        registered backplanes in :attr:`BodyXY.backplanes` and are saved as additional
        HDUs in the FITS file.

        For larger image sizes, the backplane generation can be slow, so this function
        may take some time to complete.

        Args:
            path: Filepath of output file.
        """
        with utils.filter_fits_comment_warning():
            self.add_header_metadata()

            hdu = fits.PrimaryHDU(data=self.data, header=self.header)
            hdul = fits.HDUList([hdu])

            for name, backplane in self.backplanes.items():
                utils.print_progress(name)
                img = backplane.fn()
                header = fits.Header([('ABOUT', backplane.description)])
                header.add_comment('Backplane generated by Planet Mapper software.')
                hdu = fits.ImageHDU(data=img, header=header, name=name)
                hdul.append(hdu)
            hdul.writeto(path, overwrite=True)

    def make_filename(self, extension='.fits') -> str:
        """
        Automatically generates a useful filename from the target name and date of the
        observation, e.g. `'JUPITER_2000-01-01T123456.fits.gz'`.

        Args:
            extension: Optionally specify the file extension to add to the filename.

        Returns:
            Filename built from the target name and observation date.

        """
        return '{target}_{date}{extension}'.format(
            target=self.target,
            date=self.dtm.strftime('%Y-%m-%dT%H%M%S'),
            extension=extension,
        )
