import datetime
import os
import warnings
import math
from functools import wraps, lru_cache, partial
from typing import Any, Callable, Iterable, NamedTuple, ParamSpec, TypeVar, Concatenate

import numpy as np
import PIL.Image
from astropy.io import fits
import astropy.wcs
from astropy.utils.exceptions import AstropyWarning
import scipy.ndimage
import photutils.aperture
from . import common, utils
from .body_xy import BodyXY, _cache_clearable_result, _cache_clearable_result_with_args

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
        if self.header is not None:
            # use values from header to fill in arguments (e.g. target) which aren't
            # specified by the user
            self._add_kw_from_header(kwargs, self.header)
        super().__init__(nx=self.data.shape[2], ny=self.data.shape[1], **kwargs)

        if self.header is None:
            self.header = fits.Header(
                {
                    'OBJECT': self.target,
                    'DATE-OBS': self.utc,
                }
            )

        self._use_wcs: bool = False
        self._dx: float = 0
        self._dy: float = 0
        self._dra: float = 0
        self._dec: float = 0
        self._drotation: float = 0

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
        # TODO generally do this better
        # TODO add check data is a cube
        with fits.open(self.path) as hdul:
            for idx, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    if idx:
                        header = hdul[0].header.copy()  # type: ignore
                        header.update(hdu.header.copy())
                    else:
                        header = hdu.header.copy()
                    break
            else:
                raise ValueError('No data found in provided FITS file')
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

    @staticmethod
    def _add_kw_from_header(kw: dict, header: fits.Header):
        # fill in kwargs with values from header (if they aren't specified by the user)

        if 'target' not in kw:
            for k in ['OBJECT', 'TARGET', 'TARGNAME']:
                try:
                    kw.setdefault('target', header[k])
                    break
                except KeyError:
                    continue

        if 'observer' not in kw:
            for k in ['TELESCOP']:
                try:
                    value = header[k]
                    if value.startswith('ESO-'):
                        value = 'EARTH'
                    kw.setdefault('observer', 'EARTH')
                    break
                except KeyError:
                    continue

        if 'utc' not in kw:
            for k in [
                'MJD-AVG',
                'EXPMID',
                'DATE-AVG',
            ]:
                try:
                    kw.setdefault('utc', header[k])
                    break
                except KeyError:
                    continue

            try:
                kw.setdefault('utc', header['MJD-AVG'])
            except KeyError:
                pass

            try:
                # If the header has a MJD value for the start and end of the
                # observation, average them and use astropy to convert this
                # mid-observation MJD into a fits format time string
                beg = float(header['MJD-BEG'])  #  type: ignore
                end = float(header['MJD-END'])  #  type: ignore
                mjd = (beg + end) / 2
                kw.setdefault('utc', mjd)
            except:
                pass

            for k in [
                'DATE-AVG',
                'DATE-OBS',
                'DATE-BEG',
                'DATE-END',
            ]:
                try:
                    kw.setdefault('utc', header[k])
                    break
                except KeyError:
                    continue

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

    def _get_wcs_from_header(self, supress_warnings: bool = False) -> astropy.wcs.WCS:
        with warnings.catch_warnings():
            if supress_warnings:
                warnings.simplefilter('ignore', category=AstropyWarning)
            return astropy.wcs.WCS(self.header).celestial

    def disc_from_wcs(
        self, supress_warnings: bool = False, validate: bool = True
    ) -> None:
        """
        Set disc parameters using WCS information in the observation's FITS header.

        .. warning::

            This WCS transform is not perfect

        Args:
            supress_warnings: Hide warnings produced by astropy when calculating WCS
                conversions.
            validate: Run checks to ensure the derived coordinate conversion is
                consistent with the WCS conversion. This checks that the conversions are
                consistent (to within 0.1") and that the input WCS data has appropriate
                units.
        """
        wcs = self._get_wcs_from_header(supress_warnings=supress_warnings)

        if wcs.naxis == 0:
            raise ValueError('No WCS information found in FITS header')

        if validate:
            print('WARNING: this WCS transformation is only approximate')
            # TODO do these checks better
            assert not wcs.has_distortion
            assert all(u == 'deg' for u in wcs.world_axis_units)
            assert wcs.world_axis_physical_types == ['pos.eq.ra', 'pos.eq.dec']

        # a1, a2 = wcs.pixel_to_world_values(1, 0)
        b1, b2 = wcs.pixel_to_world_values(0, 1)
        c1, c2 = wcs.pixel_to_world_values(0, 0)

        s = np.sqrt((b1 - c1) ** 2 + (b2 - c2) ** 2)

        theta_degrees = np.rad2deg(np.arctan2(b1 - c1, b2 - c2))
        arcsec_per_px = s * 60 * 60  # s = degrees/px
        x0, y0 = wcs.world_to_pixel_values(self.target_ra, self.target_dec)

        self.set_x0(x0)
        self.set_y0(y0)
        self.set_plate_scale_arcsec(arcsec_per_px)
        self.set_rotation(theta_degrees)
        self.set_disc_method('wcs')

        if validate:
            # Run checks on a few coordinates to ensure our transformation is consistent
            # with the results from WCS
            coords = [
                (0, 0),
                (self._nx, 0),
                (self._nx, self._ny),
                (0, self._ny),
                (x0, y0),
                (10, -5),
            ]
            for x, y in coords:
                ra_wcs, dec_wcs = wcs.pixel_to_world_values(x, y)
                ra, dec = self.xy2radec(x, y)
                # Do checks with -180 and %360 so that e.g. 359.99999 becomes -0.00001
                assert (ra_wcs - ra - 180) % 360 - 180 < 0.1 / 3600
                assert (dec_wcs - dec - 180) % 360 - 180 < 0.1 / 3600

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
            [np.percentile(threshold_img, 5), np.percentile(threshold_img, 95)]
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

        This routine calculates the brighntess in concentric annular apertures around
        `(x0, y0)` and sets `r0` as the radius where the brightness decreases the
        fastest. Note that this uses circular apertures, so will be less reliable for
        targets with greater flatttening.
        """
        img = self._get_img_for_fitting()
        centroid = np.array([self.get_x0(), self.get_y0()])

        r_ceil = int(min(*centroid, *(img.shape - centroid)))
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
    @_cache_clearable_result_with_args
    def mapped_data(self, degree_interval: float = 1) -> np.ndarray:
        """
        Projects the observed :attr:`data` onto a lon/lat grid using
        :func:`BodyXY.map_img`.

        Args:
            degree_interval: Interval in degrees between the longitude/latitude points.
                Passed to :func:`BodyXY.map_img`.

        Returns:
            Array containing a cube of cylindrical map of the values in :attr:`data` at
            each location on the surface of the target body. Locations which are not
            visible have a value of NaN.
        """
        projected = []
        for img in self.data:
            projected.append(self.map_img(img, degree_interval=degree_interval))
        return np.array(projected)

    # Output
    def append_to_header(
        self,
        keyword: str,
        value: str | float | bool | complex,
        comment: str | None = None,
        hierarch_keyword: bool = True,
        header: fits.Header | None = None,
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
        """
        if header is None:
            header = self.header
        if hierarch_keyword:
            keyword = f'HIERARCH {self.FITS_KEYWORD} {keyword}'
        with utils.filter_fits_comment_warning():
            header.append(fits.Card(keyword=keyword, value=value, comment=comment))

    def add_header_metadata(self, header: fits.Header | None = None):
        """
        Add automatically generated metadata a FIFTS header. This is automatically
        called by :func:`save` so `add_header_metadata` does not normally need to be
        called manually.

        Args:
            header: FITS Header which the metadata will be added to in-place. If
                `header` is `None`, then :attr:`header` will be modified.
        """
        self.append_to_header(
            'VERSION', common.__version__, 'Planet Mapper version.', header=header
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
            '[degrees] rotation of disc.',
            header=header,
        )
        self.append_to_header(
            'DISC METHOD',
            self.get_disc_method(),
            'Method used to find disc.',
            header=header,
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
        observation, e.g. `'JUPITER_2000-01-01T123456.fits.gz'`.

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

    def save_observation(self, path: str) -> None:
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
            data = self.data
            header = self.header.copy()

            self.add_header_metadata(header)
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            for name, backplane in self.backplanes.items():
                utils.print_progress(name)
                img = backplane.get_img()
                header = fits.Header([('ABOUT', backplane.description)])
                header.add_comment('Backplane generated by Planet Mapper software.')
                hdu = fits.ImageHDU(data=img, header=header, name=name)
                hdul.append(hdu)
            hdul.writeto(path, overwrite=True)

    def save_mapped_observation(
        self, path: str, include_backplanes: bool = True, degree_interval: float = 1
    ) -> None:
        """
        Save a FITS file containing the mapped observation in a cylindrical projection.

        The mapped data is generated using :func:`mapped_data`, and mapped backplane
        data is saved by default.

        For larger image sizes, the map projection and backplane generation can be slow,
        so this function may take some time to complete.

        Args:
            path: Filepath of output file.
            include_backplanes: Toggle generating and saving backplanes to output FITS
                file.
            degree_interval: Interval in degrees between the longitude/latitude points.
        """
        with utils.filter_fits_comment_warning():
            data = self.mapped_data(degree_interval)
            header = self.header.copy()

            self.add_header_metadata(header)
            self.append_to_header(
                'DEGREE-INTERVAL',
                degree_interval,
                '[deg] Degree interval in output map.',
                header=header,
            )
            self._add_map_wcs_to_header(header, degree_interval)

            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            if include_backplanes:
                for name, backplane in self.backplanes.items():
                    utils.print_progress(name)
                    img = backplane.get_map(degree_interval)
                    header = fits.Header([('ABOUT', backplane.description)])
                    header.add_comment('Backplane generated by Planet Mapper software.')
                    self._add_map_wcs_to_header(header, degree_interval)

                    hdu = fits.ImageHDU(data=img, header=header, name=name)
                    hdul.append(hdu)
            hdul.writeto(path, overwrite=True)

    def _add_map_wcs_to_header(
        self, header: fits.Header, degree_interval: float
    ) -> None:
        lons, lats = self._make_map_lonlat_arrays(degree_interval)

        # Add new values
        header['CTYPE1'] = 'Planetographic longitude, positive {}'.format(
            self.positive_longitude_direction
        )
        header['CUNIT1'] = 'deg'
        header['CRPIX1'] = 1
        header['CRVAL1'] = lons[0]
        header['CDELT1'] = lons[1] - lons[0]

        header['CTYPE2'] = 'Planetographic latitude'
        header['CUNIT2'] = 'deg'
        header['CRPIX2'] = 1
        header['CRVAL2'] = lats[0]
        header['CDELT2'] = lats[1] - lats[0]

        # Remove values which correspond to previous projection
        for a in ['1', '2']:
            for b in ['1', '2', '3']:
                for key in [f'PC{a}_{b}', f'PC{b}_{a}', f'CD{a}_{b}', f'CD{b}_{a}']:
                    header.remove(key, ignore_missing=True, remove_all=True)

    def run_gui(self) -> None:
        """
        Run an interactive GUI to display and adjust the fitted observation.

        This modifies the `Observation` object in-place, so can be used within a script
        to e.g. interactively fit the planet's disc ::

            # Loaad in some datta
            observation = planetmapper.Observation('exciting_data.fits')

            # Use the GUI to manually fit the disc and set the x0,y0,r0,rotation values
            observation.run_gui()

            # At this point, you can use the manually fitted observation
            observation.plot_wireframe_xy()
        """
        from .gui import GUI
        gui = GUI(allow_open=False)
        gui.set_observation(self)
        gui.run()
