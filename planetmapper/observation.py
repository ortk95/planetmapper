import datetime
import os
import warnings
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import astropy.wcs
import numpy as np
import photutils.aperture
import PIL.Image
import scipy.ndimage
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from . import common, utils
from .base import _cache_clearable_result
from .body_xy import BodyXY, Unpack, _MapKwargs
from .progress import SaveMapProgressHookCLI, SaveNavProgressHookCLI, progress_decorator

T = TypeVar('T')
S = TypeVar('S')
P = ParamSpec('P')


class Observation(BodyXY):
    """
    Class representing an actual observation of an astronomical body at a specific time.

    This is a subclass of :class:`BodyXY`, with additional methods to interact with the
    observed data, such as by saving a FITS file containing calculated backplane data.
    All methods described in :class:`BodyXY`, :class:`Body` and :class:`SpiceBase` are
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

    When an `Observation` object is created, the disc parameters
    `(x0, y0, r0, rotation)` initialised to the most useful values possible:

    1. If the input file has previously been fit by PlanetMapper, the previous parameter
       values saved in the FITS header are loaded using :func:`disc_from_header`.
    2. Otherwise, if there is WCS information in the FITS header, this is loaded with
       :func:`disc_from_wcs`.
    3. Finally, if there is no useful information in the FITS header (or no header is
       provided), the disc parameters are initialised using :func:`centre_disc`.

    Args:
        path: Path to data file to load. If this is `None` then `data` must be specified
            instead. Any user (`~`) and shell variables (e.g. `$var`) in the path are
            automatically expanded if possible.
        data: Array containing observation data to use instead of loading the data from
            `path`. This should only be provided if `path` is None.
        header: FITS header which corresponds to the provided `data`. This is optional
            and should only be provided if `path` is None.
        target: Name of target body, passed to :class:`Body`. If this is unspecified,
            then the target will be derived from the values in the FITS header.
        utc: Time of observation, passed to :class:`Body`. If this is unspecified, then
            the time will be derived from the values in the FITS header.
        **kwargs: Additional parameters are passed to :class:`BodyXY`. These can be used
            to specify additional parameters such as`observer`.
    """

    FITS_FILE_EXTENSIONS = ('.fits', '.fits.gz')
    """
    File extensions which will be read as FITS files. All other file extensions will be
    assumed to be images.
    """
    FITS_KEYWORD = 'PLANMAP'
    """FITS keyword used in metadata added to header of output FITS files."""

    def __init__(
        self,
        path: str | None = None,
        *,
        data: np.ndarray | None = None,
        header: fits.Header | None = None,
        **kwargs,
    ) -> None:
        self._path_arg = path
        self._data_arg = data
        self._header_arg = header

        # Add docstrings
        self.data: np.ndarray
        """Observed data."""
        self.header: fits.Header
        """
        FITS header containing data about the observation. If this is not provided, then
        a basic header will be produced containing data derived from the `target` and 
        `utc` parameters.
        """
        if path is not None:
            path = os.path.expandvars(os.path.expanduser(path))

        self.path = path
        self.header = None  # type: ignore

        # TODO add warning about header being modified in place? Or copy header?
        if self.path is None:
            if data is None:
                raise ValueError('Either `path` or `data` must be provided')
            self.data = data
            if header is not None:
                self.header = header
        else:
            # TODO should we have a way to provide both path and header for e.g. JPEGS?
            if data is not None:
                raise ValueError('`path` and `data` are mutually exclusive')
            if header is not None:
                raise ValueError('`path` and `header` are mutually exclusive')
            self._load_data_from_path()

        # TODO validate/standardise shape of data here (cube etc.)
        self.data = np.asarray(self.data)
        if len(self.data.shape) == 2:
            # Turn data into cube for consistency
            self.data = self.data[np.newaxis, ...]
        if self.header is not None:
            # use values from header to fill in arguments (e.g. target) which aren't
            # specified by the user
            self._add_kw_from_header(kwargs, self.header)
        super().__init__(nx=self.data.shape[-1], ny=self.data.shape[-2], **kwargs)

        if self.header is None:
            self.header = fits.Header(
                {
                    'OBJECT': self.target,
                    'DATE-OBS': self.utc,
                }
            )

        try:
            self.disc_from_header()
        except ValueError:
            try:
                self.disc_from_wcs(suppress_warnings=True)
            except ValueError:
                self.centre_disc()

    def __repr__(self) -> str:
        return f'Observation({self.path!r})'

    def _get_equality_tuple(self) -> tuple:
        # Use nan_to_num to convert NaNs to zeros, so that NaNs in the data don't
        # cause the equality check to fail. Then use isnan to compare the NaN masks
        # to ensure e.g. [1, NaN] != [1, 0]. Compare .data to get booleans rather
        # than numpy's array of booleans.
        return (
            self.path,
            np.nan_to_num(self.data).data,
            np.isnan(self.data).data,
            self.header,
            super()._get_equality_tuple(),
        )

    def _get_kwargs(self) -> dict[str, Any]:
        kw = super()._get_kwargs() | dict(
            path=self._path_arg,
            data=self._data_arg,
            header=self._header_arg,
        )
        kw.pop('nx')
        kw.pop('ny')
        return kw

    def _load_data_from_path(self):
        assert self.path is not None
        if any(self.path.endswith(ext) for ext in self.FITS_FILE_EXTENSIONS):
            self._load_fits_data()
        else:
            self._load_image_data()

    def _load_fits_data(self):
        assert self.path is not None
        with fits.open(self.path, memmap=False) as hdul:
            for idx, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    if idx:
                        # pylint: disable-next=no-member
                        header = hdul[0].header.copy()  # type: ignore
                        header.update(hdu.header.copy())
                    else:
                        header = hdu.header.copy()
                    break
            else:
                raise ValueError('No data found in provided FITS file')

        if len(data.shape) == 2:
            # If greyscale image, add another dimension so that it is an image cube with
            # a single frame. This will ensure that data will always be a cube.
            data = np.array([data])
        self.data = data
        self.header = header

    def _load_image_data(self):
        assert self.path is not None
        image = np.flipud(np.array(PIL.Image.open(self.path)))

        if len(image.shape) == 2:
            # If greyscale image, add another dimension so that it is an image cube with
            # a single frame. This will ensure that data will always be a cube.
            image = np.array([image])
        else:
            # If RGB image, change the axis order so wavelength is the first axis (i.e.
            # consistent with FITS)
            image = np.moveaxis(image, 2, 0)
        self.data = image

    @classmethod
    def _add_kw_from_header(cls, kw: dict, header: fits.Header):
        # fill in kwargs with values from header (if they aren't specified by the user)

        _try_get_header_value(
            kw,
            header,
            'target',
            [cls._make_fits_kw('TARGET'), 'OBJECT', 'TARGET', 'TARGNAME'],
        )
        _try_get_header_value(
            kw,
            header,
            'observer',
            [cls._make_fits_kw('OBSERVER'), 'TELESCOP'],
            value_fn=lambda v: 'EARTH' if str(v).startswith('ESO-') else v,
        )

        _try_get_header_value(
            kw,
            header,
            'utc',
            [cls._make_fits_kw('UTC-OBS'), 'MJD-AVG', 'EXPMID', 'DATE-AVG'],
        )
        if 'utc' not in kw:
            try:
                # If the header has a MJD value for the start and end of the
                # observation, average them and use astropy to convert this
                # mid-observation MJD into a fits format time string
                beg = float(header['MJD-BEG'])  #  type: ignore
                end = float(header['MJD-END'])  #  type: ignore
                mjd = (beg + end) / 2
                kw['utc'] = mjd
            except (KeyError, TypeError, ValueError):
                pass
            if 'utc' not in kw:
                try:
                    kw['utc'] = header['DATE-OBS'] + ' ' + header['TIME-OBS']  # type: ignore
                except KeyError:
                    pass
            _try_get_header_value(
                kw,
                header,
                'utc',
                ['DATE-OBS', 'DATE-BEG', 'DATE-END', 'MJD-BEG', 'MJD-END'],
            )

        _try_get_header_value(
            kw, header, 'observer_frame', [cls._make_fits_kw('OBSERVER-FRAME')]
        )
        _try_get_header_value(
            kw, header, 'illumination_source', [cls._make_fits_kw('ILLUMINATION')]
        )
        _try_get_header_value(
            kw, header, 'aberration_correction', [cls._make_fits_kw('ABCORR')]
        )
        _try_get_header_value(
            kw, header, 'subpoint_method', [cls._make_fits_kw('SUBPOINT-METHOD')]
        )
        _try_get_header_value(
            kw, header, 'surface_method', [cls._make_fits_kw('SURFACE-METHOD')]
        )

    # Auto disc id
    def disc_from_header(self) -> None:
        """
        Sets the target's planetary disc data in the FITS header generated by previous
        runs of `planetmapper`.

        This uses values such as `HIERARCH PLANMAP DISC X0` to set the disc location to
        be the same as the previous run.

        Raises:
            ValueError: if the header does not contain appropriate metadata values. This
                is likely because the file was not created by `planetmapper`.
        """
        if (
            self._make_fits_kw('MAP PROJECTION') in self.header
            or self._make_fits_kw('DEGREE-INTERVAL') in self.header
        ):
            raise ValueError('FITS header refers to mapped data')
        try:
            self.set_disc_params(
                x0=self.header[self._make_fits_kw('DISC X0')],  # type: ignore
                y0=self.header[self._make_fits_kw('DISC Y0')],  # type: ignore
                r0=self.header[self._make_fits_kw('DISC R0')],  # type: ignore
                rotation=self.header[self._make_fits_kw('DISC ROT')],  # type: ignore
            )
        except KeyError as exc:
            raise ValueError('No disc parameters found in FITS header') from exc

    def _get_wcs_from_header(self, suppress_warnings: bool = False) -> astropy.wcs.WCS:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter('ignore', category=AstropyWarning)
            return astropy.wcs.WCS(self.header).celestial

    def _get_disc_params_from_wcs(
        self,
        suppress_warnings: bool = False,
        validate: bool = True,
    ) -> tuple[float, float, float, float]:
        wcs = self._get_wcs_from_header(suppress_warnings=suppress_warnings)

        if wcs.naxis == 0:
            raise ValueError('No WCS information found in FITS header')

        if validate:
            if not all(u == 'deg' for u in wcs.world_axis_units):
                raise ValueError('WCS coordinates are not in degrees')
            if not wcs.world_axis_physical_types == ['pos.eq.ra', 'pos.eq.dec']:
                raise ValueError('WCS axes are not RA/Dec coordinates')
            if wcs.has_distortion:
                raise ValueError('WCS conversion contains distortion terms')

        x0, y0 = wcs.world_to_pixel_values(self.target_ra, self.target_dec)

        b1, b2 = wcs.pixel_to_world_values(x0, y0 + 1)
        c1, c2 = wcs.pixel_to_world_values(x0, y0)

        rotation = np.rad2deg(np.arctan2(b1 - c1, b2 - c2))

        s = self.angular_dist(b1, b2, c1, c2)
        arcsec_per_px = s * 60 * 60  # s = degrees/px
        r0 = self.target_diameter_arcsec / (2 * arcsec_per_px)
        x0, y0 = wcs.world_to_pixel_values(self.target_ra, self.target_dec)

        return x0, y0, r0, rotation

    def disc_from_wcs(
        self, suppress_warnings: bool = False, validate: bool = True
    ) -> None:
        """
        Set disc parameters using WCS information in the observation's FITS header.

        See also :func:`rotation_from_wcs` and :func:`plate_scale_from_wcs`.

        .. note::

            There may be very slight differences between the coordinates converted
            directly from the WCS information, and the coordinates converted by
            PlanetMapper

        Args:
            suppress_warnings: Hide warnings produced by astropy when calculating WCS
                conversions.
            validate: Run checks to ensure the WCS conversion has appropriate RA/Dec
                coordinate dimensions.
        Raises:
            ValueError: if no WCS information is found in the FITS header, or validation
                fails.
        """
        x0, y0, r0, rotation = self._get_disc_params_from_wcs(
            suppress_warnings, validate
        )
        self.set_x0(x0)
        self.set_y0(y0)
        self.set_r0(r0)
        self.set_rotation(rotation)
        self.set_disc_method('wcs')

    def position_from_wcs(
        self, suppress_warnings: bool = False, validate: bool = True
    ) -> None:
        """
        Set disc position `(x0, y0)` using WCS information in the observation's FITS
        header.

        See also :func:`disc_from_wcs`.

        Args:
            suppress_warnings: Hide warnings produced by astropy when calculating WCS
                conversions.
            validate: Run checks to ensure the WCS conversion has appropriate RA/Dec
                coordinate dimensions.
        Raises:
            ValueError: if no WCS information is found in the FITS header, or validation
                fails.
        """
        x0, y0, r0, rotation = self._get_disc_params_from_wcs(
            suppress_warnings, validate
        )
        self.set_x0(x0)
        self.set_y0(y0)
        self.set_disc_method('wcs_position')

    def rotation_from_wcs(
        self, suppress_warnings: bool = False, validate: bool = True
    ) -> None:
        """
        Set disc rotation using WCS information in the observation's FITS header.

        See also :func:`disc_from_wcs`.

        Args:
            suppress_warnings: Hide warnings produced by astropy when calculating WCS
                conversions.
            validate: Run checks to ensure the WCS conversion has appropriate RA/Dec
                coordinate dimensions.
        Raises:
            ValueError: if no WCS information is found in the FITS header, or validation
                fails.
        """
        x0, y0, r0, rotation = self._get_disc_params_from_wcs(
            suppress_warnings, validate
        )
        self.set_rotation(rotation)
        self.set_disc_method('wcs_rotation')

    def plate_scale_from_wcs(
        self, suppress_warnings: bool = False, validate: bool = True
    ) -> None:
        """
        Set plate scale (i.e. `r0`) using WCS information in the observation's FITS
        header.

        See also :func:`disc_from_wcs`.

        Args:
            suppress_warnings: Hide warnings produced by astropy when calculating WCS
                conversions.
            validate: Run checks to ensure the WCS conversion has appropriate RA/Dec
                coordinate dimensions.
        Raises:
            ValueError: if no WCS information is found in the FITS header, or validation
                fails.
        """
        x0, y0, r0, rotation = self._get_disc_params_from_wcs(
            suppress_warnings, validate
        )
        self.set_r0(r0)
        self.set_disc_method('wcs_plate_scale')

    def _get_img_for_fitting(self) -> np.ndarray:
        img = np.nansum(self.data, axis=0)
        mask_img = np.isnan(img)
        img[mask_img] = np.nanmin(img)  # Mask nan values for com calculation etc.
        return img

    def fit_disc_position(self) -> None:
        """
        Automatically find and set `x0` and `y0` so that the planet's disc is fit to the
        brightest part of the data.
        """
        threshold_img = self._get_img_for_fitting()
        threshold = 0.5 * sum(
            [  # type: ignore
                np.percentile(threshold_img, 5),
                np.percentile(threshold_img, 95),
            ]
        )
        threshold_img[np.where(threshold_img <= threshold)] = 0
        threshold_img[np.where(threshold_img > threshold)] = 1
        x0, y0 = np.array(scipy.ndimage.center_of_mass(threshold_img))[::-1]

        self.set_x0(x0)
        self.set_y0(y0)
        self.set_disc_method('fit_position')

    def fit_disc_radius(self) -> None:
        """
        Automatically find and set `r0` using aperture photometry.

        This routine calculates the brightness in concentric annular apertures around
        `(x0, y0)` and sets `r0` as the radius where the brightness decreases the
        fastest. Note that this uses circular apertures, so will be less reliable for
        targets with greater flattening and may not work well for targets which are not
        entirely in the image frame.

        Raises:
            ValueError: if `x0` or `y0` are not within the image frame.
        """
        if not self._xy_in_image_frame(self.get_x0(), self.get_y0()):
            raise ValueError(
                'x0 and y0 must be within the image frame to fit the radius'
            )

        img = self._get_img_for_fitting()
        centroid = np.array([self.get_x0(), self.get_y0()])

        r_ceil = max(int(min(*centroid, *(img.shape - centroid))), 2)
        if r_ceil > 100:
            r_list = np.linspace(1, r_ceil + 1, 100)
        else:
            r_list = np.array(range(1, r_ceil + 1))
        apertures = [photutils.aperture.CircularAperture(centroid, r) for r in r_list]

        val_list = []
        for aperture in apertures:
            table = photutils.aperture.aperture_photometry(img, aperture)
            aperture_sum = float(table['aperture_sum'])  # type: ignore
            val_list.append(aperture_sum / aperture.area)
        val_list = np.array(val_list)

        r_list = r_list[1:] - 0.5 * (
            r_list[1] - r_list[0]
        )  # Get radii corresponding to dv
        dv_list = np.diff(val_list)
        r0 = r_list[dv_list.argmin()]
        self.set_r0(r0)
        self.set_disc_method('fit_r0')

    # Mapping
    def get_mapped_data(
        self,
        interpolation: Literal['nearest', 'linear', 'quadratic', 'cubic'] = 'linear',
        **map_kwargs: Unpack[_MapKwargs],
    ) -> np.ndarray:
        """
        Projects the observed :attr:`data` onto a map. See
        :func:`BodyXY.generate_map_coordinates` for details about customising the
        projection used.

        For larger datasets, it can take some time to map every wavelength. Therefore,
        the mapped data is automatically cached (in a similar way to backplanes - see
        :class:`BodyXY`) so that subsequent calls to this function do not have to
        recompute the mapped data. As with cached backplanes, the cached mapped data is
        automatically cleared if any disc parameters are changed (i.e. you shouldn't
        need to worry about the cache, it all happens 'magically' behind the scenes).

        Args:
            degree_interval: Interval in degrees between the longitude/latitude points.
                Passed to :func:`BodyXY.map_img`.
            interpolation: Interpolation used when mapping. This can either any of
                `'nearest'`, `'linear'`, `'quadratic'` or `'cubic'`. Passed to
                :func:`BodyXY.map_img`.
            **map_kwargs: Additional arguments are passed to
                :func:`BodyXY.generate_map_coordinates` to specify and customise the map
                projection.
        Returns:
            Array containing cube of maped of the values in `img` at each location on
            the surface of the target body. Locations which are not visible or outside
            the projection domain have a value of NaN.
        """
        # Return a copy so that the cached value isn't tainted by any modifications
        return self._get_mapped_data(interpolation=interpolation, **map_kwargs).copy()

    @_cache_clearable_result
    @progress_decorator
    def _get_mapped_data(
        self,
        interpolation: Literal['nearest', 'linear', 'quadratic', 'cubic'] = 'linear',
        **map_kwargs: Unpack[_MapKwargs],
    ):
        projected = []
        if interpolation == 'linear' and np.any(np.isnan(self.data)):
            data = np.nan_to_num(self.data)
            print(
                'Warning, data contains NaN values which will be set to 0 before interpolating'
            )
        else:
            data = self.data
        for idx, img in enumerate(data):
            self._update_progress_hook(idx / len(data))
            projected.append(
                self.map_img(
                    img,
                    interpolation=interpolation,
                    **map_kwargs,
                )
            )
        return np.array(projected)

    # Output
    def append_to_header(
        self,
        keyword: str,
        value: str | float | bool | complex,
        comment: str | None = None,
        hierarch_keyword: bool = True,
        header: fits.Header | None = None,
        truncate_strings: bool = True,
        remove_existing: bool = True,
    ):
        """
        Add a card to a FITS header. If a `header` is not specified, then :attr:`header`
        is modified.

        By default, the keyword is modified to provide a consistent keyword prefix for
        all header cards added by this routine.

        Args:
            keyword: Card keyword.
            value: Card value.
            comment: Card comment. If unspecified not comment will be added.
            hierarch_keyword: Toggle adding the keyword prefix from :attr:`FITS_KEYWORD`
                to the keyword.
            header: FITS Header which the card will be added to in-place. If `header` is
                `None`, then :attr:`header` will be modified.
            truncate_strings: Allow string values to be truncated if they will create
                a card longer than 80 characters.
            remove_existing: Remove any existing cards with the same key before adding
                the new card.
        """
        if header is None:
            header = self.header
        if hierarch_keyword:
            keyword = self._make_fits_kw(keyword)
        if truncate_strings and isinstance(value, str):
            if len(keyword) + len(value) + 4 > 80:
                # +4 accounts for space, equals and two quotes around the value
                n = 80 - len(keyword) - 4 - 3
                value = value[:n] + '...'
        if remove_existing:
            header.remove(keyword, ignore_missing=True, remove_all=True)
        with utils.filter_fits_comment_warning():
            header.append(fits.Card(keyword=keyword, value=value, comment=comment))

    @classmethod
    def _make_fits_kw(cls, keyword: str) -> str:
        return f'HIERARCH {cls.FITS_KEYWORD} {keyword}'

    def add_header_metadata(self, header: fits.Header | None = None):
        """
        Add automatically generated metadata a FITS header. This is automatically
        called by :func:`save` so `add_header_metadata` does not normally need to be
        called manually.

        Args:
            header: FITS Header which the metadata will be added to in-place. If
                `header` is `None`, then :attr:`header` will be modified.
        """
        self.append_to_header(
            'VERSION', common.__version__, 'PlanetMapper version.', header=header
        )
        self.append_to_header('URL', common.__url__, 'Webpage.', header=header)
        self.append_to_header(
            'DATE',
            datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'File generation datetime.',
            header=header,
        )
        if self.path is not None:
            self.append_to_header(
                'INFILE', os.path.split(self.path)[1], 'Input file name.', header=header
            )
        self.append_to_header(
            'DISC X0',
            self.get_x0(),
            '[pixels] x coordinate of disc centre.',
            header=header,
        )
        self.append_to_header(
            'DISC Y0',
            self.get_y0(),
            '[pixels] y coordinate of disc centre.',
            header=header,
        )
        self.append_to_header(
            'DISC R0',
            self.get_r0(),
            '[pixels] equatorial radius of disc.',
            header=header,
        )
        self.append_to_header(
            'DISC ROT',
            self.get_rotation(),
            '[degrees] rotation of image.',
            header=header,
        )
        self.append_to_header(
            'DISC METHOD',
            self.get_disc_method(),
            'Method used to find disc.',
            header=header,
        )
        self.append_to_header(
            'UTC-OBS', self.utc, 'UTC date of observation', header=header
        )
        self.append_to_header(
            'ET-OBS', self.et, 'J2000 ephemeris seconds of observation.', header=header
        )
        self.append_to_header(
            'TARGET', self.target, 'Target body name used in SPICE.', header=header
        )
        self.append_to_header(
            'TARGET-ID',
            self.target_body_id,
            'Target body ID from SPICE.',
            header=header,
        )
        self.append_to_header(
            'SUBPOINT LAT',
            self.subpoint_lat,
            '[degrees] Sub-observer pgr latitude.',
            header=header,
        )
        self.append_to_header(
            'SUBPOINT LON',
            self.subpoint_lon,
            '[degrees] Sub-observer pgr longitude.',
            header=header,
        )
        self.append_to_header(
            'LON-DIRECTION',
            self.positive_longitude_direction,
            'Positive pgr longitude direction.',
            header=header,
        )
        self.append_to_header(
            'NP-ANGLE',
            self.north_pole_angle(),
            '[degrees] North pole angle.',
            header=header,
        )
        self.append_to_header(
            'TARGET RA',
            self.target_ra,
            '[degrees] RA of target centre.',
            header=header,
        )
        self.append_to_header(
            'TARGET DEC',
            self.target_dec,
            '[degrees] Dec of target centre.',
            header=header,
        )
        self.append_to_header(
            'TARGET DIAMETER',
            self.target_diameter_arcsec,
            '[arcsec] Equatorial angular diameter of target.',
            header=header,
        )
        self.append_to_header(
            'R EQ',
            self.r_eq,
            '[km] Target equatorial radius from SPICE.',
            header=header,
        )
        self.append_to_header(
            'R POLAR',
            self.r_polar,
            '[km] Target polar radius from SPICE.',
            header=header,
        )
        self.append_to_header(
            'FLATTENING',
            self.flattening,
            'Flattening of target body.',
            header=header,
        )
        self.append_to_header(
            'LIGHT-TIME',
            self.target_light_time,
            '[seconds] Light time to target from SPICE.',
            header=header,
        )
        self.append_to_header(
            'DISTANCE',
            self.target_distance,
            '[km] Distance to target from SPICE.',
            header=header,
        )
        self.append_to_header(
            'OBSERVER', self.observer, 'Observer name used in SPICE.', header=header
        )
        self.append_to_header(
            'TARGET-FRAME',
            self.target_frame,
            'Target frame used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'OBSERVER-FRAME',
            self.observer_frame,
            'Observer frame used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'ILLUMINATION',
            self.illumination_source,
            'Illumination source used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'ABCORR',
            self.aberration_correction,
            'Aberration correction used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'SUBPOINT-METHOD',
            self.subpoint_method,
            'Subpoint method used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'SURFACE-METHOD',
            self.surface_method,
            'Surface intercept method used in SPICE.',
            header=header,
        )
        self.append_to_header(
            'OPTIMIZATION-USED',
            self._optimize_speed,
            'Speed optimizations used.',
            header=header,
        )

    def make_filename(
        self, extension: str = '.fits', prefix: str = '', suffix: str = ''
    ) -> str:
        """
        Automatically generates a useful filename from the target name and date of the
        observation, e.g. `'JUPITER_2000-01-01T123456.fits'`.

        Args:
            extension: Optionally specify the file extension to add to the filename.
            prefix: Optionally specify filename prefix.
            suffix: Optionally specify filename suffix.

        Returns:
            Filename built from the target name and observation date.
        """
        return '{prefix}{target}_{date}{suffix}{extension}'.format(
            prefix=prefix,
            target=self.target,
            date=self.dtm.strftime('%Y-%m-%dT%H%M%S'),
            extension=extension,
            suffix=suffix,
        )

    @progress_decorator
    def save_observation(
        self,
        path: str,
        *,
        include_wireframe: bool = True,
        wireframe_kwargs: dict[str, Any] | None = None,
        show_progress: bool = False,
        print_info: bool = True,
    ) -> None:
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
            include_wireframe: Toggle generating and saving wireframe overlay image as
                an additional backplane of the output FITS file. The wireframe is
                generated by :func:`BodyXY.get_wireframe_overlay_img`.
            wireframe_kwargs: Dictionary of keyword arguments passed to
                :func:`BodyXY.get_wireframe_overlay_img` to customise the wireframe
                overlay.
            show_progress: Display a progress bar rather than printing progress info.
                This does not have an effect if `show_progress=True` was set when
                creating this `Observation`.
            print_info: Toggle printing of progress information (defaults to `True`).
        """
        if show_progress and self._get_progress_hook() is None:
            print_info = False
            self._set_progress_hook(SaveNavProgressHookCLI())
        else:
            show_progress = False

        if print_info:
            print('Saving observation to', path)

        progress_max = 10 + len(self.backplanes)
        with utils.filter_fits_comment_warning():
            data = self.data
            header = self.header.copy()

            self._update_progress_hook(1 / progress_max)

            self.add_header_metadata(header)
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            for bp_idx, (name, backplane) in enumerate(self.backplanes.items()):
                self._update_progress_hook((bp_idx + 1) / progress_max)
                if print_info:
                    print(' Creating backplane:', name)
                img = backplane.get_img()
                header = fits.Header([('ABOUT', backplane.description)])
                header.add_comment('Backplane generated by PlanetMapper software.')
                hdu = fits.ImageHDU(data=img, header=header, name=name)
                hdul.append(hdu)

            if include_wireframe:
                if print_info:
                    print(' Creating wireframe...')
                wireframe = self.get_wireframe_overlay_img(**wireframe_kwargs or {})
                header = fits.Header([('ABOUT', 'Wireframe image overlay')])
                header.add_comment(
                    'Wireframe overlay generated by PlanetMapper software.'
                )
                hdu = fits.ImageHDU(data=wireframe, header=header, name='WIREFRAME')
                hdul.append(hdu)

            if print_info:
                print(' Saving file...')
            utils.check_path(path)
            hdul.writeto(path, overwrite=True, output_verify='warn')
        if print_info:
            print('File saved')

        if show_progress:
            self._update_progress_hook(1)
            self._remove_progress_hook()

    @progress_decorator
    def save_mapped_observation(
        self,
        path: str,
        *,
        interpolation: Literal['nearest', 'linear', 'quadratic', 'cubic'] = 'linear',
        include_backplanes: bool = True,
        include_wireframe: bool = True,
        wireframe_kwargs: dict[str, Any] | None = None,
        show_progress: bool = False,
        print_info: bool = True,
        **map_kwargs: Unpack[_MapKwargs],
    ) -> None:
        """
        Save a FITS file containing the mapped observation in a cylindrical projection.

        The mapped data is generated using :func:`get_mapped_data`, and mapped backplane
        data is saved by default.

        For larger image sizes, the map projection and backplane generation can be slow,
        so this function may take some time to complete.

        Args:
            path: Filepath of output file.
            interpolation: Interpolation used when mapping. This can either any of
                `'nearest'`, `'linear'`, `'quadratic'` or `'cubic'`. Passed to
                :func:`BodyXY.map_img`.
            include_backplanes: Toggle generating and saving backplanes to output FITS
                file.
            include_wireframe: Toggle generating and saving wireframe overlay map as an
                additional backplane of the output FITS file. The wireframe is generated
                by :func:`BodyXY.get_wireframe_overlay_map`.
            wireframe_kwargs: Dictionary of keyword arguments passed to
                :func:`BodyXY.get_wireframe_overlay_map` to customise the wireframe
                overlay.
            show_progress: Display a progress bar rather than printing progress info.
                This does not have an effect if `show_progress=True` was set when
                creating this `Observation`.
            print_info: Toggle printing of progress information (defaults to `True`).
            **map_kwargs: Additional arguments are passed to
                :func:`BodyXY.generate_map_coordinates` to specify and customise the map
                projection.
        """
        if show_progress and self._get_progress_hook() is None:
            print_info = False
            self._set_progress_hook(SaveMapProgressHookCLI(len(self.data)))
        else:
            show_progress = False

        if print_info:
            print('Saving map to', path)

        progress_max = 15 + (len(self.backplanes) if include_backplanes else 0)
        with utils.filter_fits_comment_warning():
            if print_info:
                print(' Projecting mapped data...')
            data = self.get_mapped_data(interpolation=interpolation, **map_kwargs)
            header = self.header.copy()

            self._update_progress_hook(1 / progress_max)

            self.add_header_metadata(header)
            self._add_map_header_metadata(header, interpolation, **map_kwargs)
            self._add_map_wcs_to_header(header, **map_kwargs)

            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            if include_backplanes:
                for bp_idx, (name, backplane) in enumerate(self.backplanes.items()):
                    self._update_progress_hook((bp_idx + 1) / progress_max)
                    if print_info:
                        print(' Creating backplane:', name)
                    img = backplane.get_map(**map_kwargs)
                    header = fits.Header([('ABOUT', backplane.description)])
                    header.add_comment('Backplane generated by PlanetMapper software.')
                    self._add_map_wcs_to_header(header, **map_kwargs)

                    hdu = fits.ImageHDU(data=img, header=header, name=name)
                    hdul.append(hdu)

            if include_wireframe:
                if print_info:
                    print(' Creating wireframe...')
                wireframe = self.get_wireframe_overlay_map(
                    **wireframe_kwargs or {}, **map_kwargs
                )
                header = fits.Header([('ABOUT', 'Wireframe map overlay')])
                header.add_comment(
                    'Wireframe overlay generated by PlanetMapper software.'
                )
                hdu = fits.ImageHDU(data=wireframe, header=header, name='WIREFRAME')
                hdul.append(hdu)

            if print_info:
                print(' Saving file...')
            utils.check_path(path)
            hdul.writeto(path, overwrite=True, output_verify='warn')
        if print_info:
            print('File saved')

        if show_progress:
            self._update_progress_hook(1)
            self._remove_progress_hook()

    def _add_map_header_metadata(
        self,
        header: fits.Header,
        interpolation: str,
        **map_kwargs: Unpack[_MapKwargs],
    ):
        lons, lats, xx, yy, transformer, info = self.generate_map_coordinates(
            **map_kwargs
        )
        self.append_to_header(
            'MAP INTERPOLATION',
            interpolation,
            'Interpolation method used in mapping.',
            header=header,
        )
        self.append_to_header(
            'MAP PROJECTION',
            info['projection'],
            'Projection used for mapping.',
            header=header,
        )

        try:
            self.append_to_header(
                'MAP DEGREE-INTERVAL',
                info['degree_interval'],
                '[deg] Degree interval in output map.',
                header=header,
            )
        except KeyError:
            pass

        try:
            self.append_to_header(
                'MAP LON',
                info['lon'],
                'Central longitude of map projection.',
                header=header,
            )
        except KeyError:
            pass

        try:
            self.append_to_header(
                'MAP LAT',
                info['lat'],
                'Central latitude of map projection.',
                header=header,
            )
        except KeyError:
            pass

        try:
            self.append_to_header(
                'MAP SIZE',
                info['size'],
                'Size of output map.',
                header=header,
            )
        except KeyError:
            pass

    def _add_map_wcs_to_header(
        self,
        header: fits.Header,
        **map_kwargs: Unpack[_MapKwargs],
    ) -> None:
        lons, lats, xx, yy, transformer, info = self.generate_map_coordinates(
            **map_kwargs
        )
        if info['projection'] == 'rectangular':
            # Add new values
            header['CTYPE1'] = 'Planetographic longitude, positive {}'.format(
                self.positive_longitude_direction
            )
            header['CUNIT1'] = 'deg'
            header['CRPIX1'] = 1
            header['CRVAL1'] = lons[0][0]
            header['CDELT1'] = lons[0][1] - lons[0][0]

            header['CTYPE2'] = 'Planetographic latitude'
            header['CUNIT2'] = 'deg'
            header['CRPIX2'] = 1
            header['CRVAL2'] = lats[0][0]
            header['CDELT2'] = lats[1][0] - lats[0][0]
        else:
            # Remove values which correspond to previous projection
            for n in ['1', '2']:
                for key in [
                    f'CTYPE{n}',
                    f'CUNIT{n}',
                    f'CRPIX{n}',
                    f'CRVAL{n}',
                    f'CDELT{n}',
                ]:
                    header.remove(key, ignore_missing=True, remove_all=True)

        # Remove values which correspond to previous projection
        for a in ['1', '2']:
            for b in ['1', '2', '3']:
                for key in [f'PC{a}_{b}', f'PC{b}_{a}', f'CD{a}_{b}', f'CD{b}_{a}']:
                    header.remove(key, ignore_missing=True, remove_all=True)

    def run_gui(self) -> list[tuple[float, float]]:
        """
        Run an interactive GUI to display and adjust the fitted observation.

        This modifies the `Observation` object in-place, so can be used within a script
        to e.g. interactively fit the planet's disc. Simply run the GUI, adjust the
        parameters until the disc is fit, then close the GUI and the `Observation`
        object will have your new values: ::

            # Load in some data
            observation = planetmapper.Observation('exciting_data.fits')

            # Use the GUI to manually fit the disc and set the x0,y0,r0,rotation values
            observation.run_gui()

            # At this point, you can use the manually fitted observation
            observation.plot_wireframe_xy()

        .. hint ::

            Once you have manually fitted the disc, you can simply close the user
            interface window and the disc parameters will be updated to the new values.
            This means that you don't need to click the `Save...` button unless you
            specifically want to save a navigated file to disk.


        The return value can also be used to interactively select a locations:::

            observation = planetmapper.Observation('exciting_data.fits')
            clicks = observation.run_gui()
            ax = observation.plot_wireframe_radec()
            for x, y in clicks:
                ra, dec = observation.xy2radec()
                ax.scatter(ra, dec)

        .. note ::

            The `Open...` button is disabled for user interfaces created by this method
            to ensure that only one :class:`Observation` object is modified by the user
            interface.

            If you want the full user interface functionality instead, then call
            `planetmapper` from the command line or create and run a user interface
            manually using :func:`planetmapper.gui.GUI.run`.

        Returns:
            List of `(x, y)` pixel coordinate tuples corresponding to where the user
            clicked on the plot window to mark a location.
        """
        # pylint: disable=cyclic-import
        from .gui import GUI  # Prevent circular imports

        gui = GUI(allow_open=False)
        gui.set_observation(self)
        gui.run()
        return gui.click_locations


def _try_get_header_value(
    kw: dict,
    header: fits.Header,
    kw_key: str,
    header_keys: list[str],
    value_fn: Callable[[Any], Any] | None = None,
) -> bool:
    if value_fn is None:
        value_fn = lambda x: x
    if kw_key not in kw:
        for hk in header_keys:
            try:
                kw[kw_key] = value_fn(header[hk])
                return True
            except KeyError:
                pass
    return False
