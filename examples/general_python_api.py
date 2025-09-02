#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to generate example plots on
https://planetmapper.readthedocs.io/en/latest/general_python_api.html

Run download_spice_kernels() to download the kernels required for these examples.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..')  # Use local dev version of planetmapper
import planetmapper
from planetmapper.kernel_downloader import download_urls

PLOT_DIRECTORY = '../docs/images'


def main():
    pass
    # download_spice_kernels()
    # plot_saturn_wireframe()
    # plot_neptune_wireframe()
    # plot_jupiter_wireframe()
    # plot_uranus_wireframe_comparison()
    plot_saturn_wireframe_formatting()
    # plot_europa_backplane()
    # plot_jupiter_backplane()
    # plot_jupiter_mapped()


def download_spice_kernels():
    # This command will download ~2GB of data
    # Note, the exact URLs in this example may not work if new kernel versions are published
    download_urls(
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/ura111.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep097.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/HST/kernels/spk/',
    )


def plot_saturn_wireframe():
    body = planetmapper.Body('saturn', '2020-01-01')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    body.plot_wireframe_radec(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'saturn_wireframe_radec.png'))
    plt.close(fig)


def plot_neptune_wireframe():
    body = planetmapper.Body('neptune', '2020-01-01')

    # Add Triton to any wireframe plots
    body.add_other_bodies_of_interest('triton')

    # Mark this specific coordinate (if visible) on any wireframe plots
    body.coordinates_of_interest_lonlat.append((360, -45))

    # Add some rings to the plot
    rings = planetmapper.data_loader.get_ring_radii()['NEPTUNE']
    for radii in rings.values():
        body.ring_radii.update(radii)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    body.plot_wireframe_radec(ax)

    # Manually add some text to the plot
    ax.text(
        body.target_ra, body.target_dec + 2 / 60 / 60, 'NEPTUNE', color='b', ha='center'
    )
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'neptune_wireframe_radec.png'))
    plt.close(fig)


def plot_jupiter_wireframe():
    fig, [ax_radec, ax_km] = plt.subplots(nrows=2, figsize=(6, 8), dpi=200)

    dates = ['2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00']
    colors = ['r', 'g', 'b']

    for date, c in zip(dates, colors):
        body = planetmapper.Body('jupiter', date)
        body.add_other_bodies_of_interest('Io')
        body.plot_wireframe_radec(ax_radec, color=c)
        body.plot_wireframe_km(ax_km, color=c)

        # Plot some blank data with the correct colour to go on the legend
        ax_radec.scatter(np.nan, np.nan, color=c, label=date)

    ax_radec.legend(loc='upper left')

    ax_radec.set_title('Position in the sky')
    ax_km.set_title('Position relative to Jupiter')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'jupiter_wireframes.png'))
    plt.close(fig)


def plot_uranus_wireframe_comparison():
    body = planetmapper.Body('uranus', '2020-01-01')
    body.add_named_rings()
    body.add_other_bodies_of_interest('miranda')

    fig, ((ax_radec, ax_km), (ax_angular1, ax_angular2)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        dpi=200,
        gridspec_kw=dict(hspace=0.3, wspace=0.3),
    )

    body.plot_wireframe_radec(ax_radec)
    ax_radec.set_title('plot_wireframe_radec()')

    body.plot_wireframe_km(ax_km)
    ax_km.set_title('plot_wireframe_km()')

    body.plot_wireframe_angular(ax_angular1)
    ax_angular1.set_title('plot_wireframe_angular()')

    miranda = body.create_other_body('miranda')
    ra, dec = miranda.target_ra, miranda.target_dec
    # angular plot centred on custom RA/Dec and with a custom rotation
    body.plot_wireframe_angular(
        ax_angular2,
        origin_ra=ra,
        origin_dec=dec,
        coordinate_rotation=-45,
    )
    ax_angular2.set_title('plot_wireframe_angular(...)')

    fig.savefig(
        os.path.join(PLOT_DIRECTORY, 'uranus_wireframe_comparison.png'),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_saturn_wireframe_formatting():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    body = planetmapper.Body('saturn', '2020-02-08', observer='iapetus')
    body.add_other_bodies_of_interest('dione', 'methone')
    body.plot_wireframe_km(
        ax,
        scale_factor=1 / body.r_eq,  # use units of Saturn radii rather than km
        add_title=False,
        label_poles=False,
        indicate_equator=True,
        indicate_prime_meridian=True,
        grid_interval=15,
        grid_lat_limit=75,
        formatting={
            'grid': {'linestyle': '-', 'linewidth': 0.5, 'alpha': 0.3},
            'prime_meridian': {'linewidth': 1, 'color': 'r'},
            'equator': {'linewidth': 1, 'color': 'r'},
            'terminator': {'color': 'b'},
            'limb_illuminated': {'color': 'b'},
            'ring': {'color': 'g', 'linestyle': ':'},
            'other_body_of_interest_marker': {'marker': '*'},
            'other_body_of_interest_label': {'color': 'c', 'rotation': 30, 'alpha': 1},
        },
    )
    ax.set_xlabel('Distance in Saturn radii')
    ax.set_ylabel('Distance in Saturn radii')
    ax.annotate(
        body.get_description(),
        (0.01, 0.02),
        xycoords='axes fraction',
        color='0.5',
        size='small',
    )

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'saturn_wireframe_formatting.png'))
    plt.close(fig)


def plot_europa_backplane():
    observation = planetmapper.Observation('gui_data/europa.fits')

    # Set the disc position
    observation.set_plate_scale_arcsec(12.25e-3)
    observation.set_disc_params(x0=110, y0=104)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    observation.plot_backplane_img('LON-GRAPHIC', ax=ax)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'europa_backplane.png'))
    plt.close(fig)


def plot_jupiter_backplane():
    # Create an object representing how Jupiter would appear in a 50x50 pixel image
    # taken from Earth at a specific time
    body = planetmapper.BodyXY('jupiter', utc='2030-01-01', observer='Earth', sz=50)
    body.set_disc_params(x0=25, y0=25, r0=20)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    body.plot_backplane_img('RADIAL-VELOCITY', ax=ax)

    radial_velocities = body.get_backplane_img('RADIAL-VELOCITY')
    print(f'Average radial velocity: {np.nanmean(radial_velocities):.2f} km/s')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'jupiter_backplane.png'))
    plt.close(fig)


def plot_jupiter_mapped():
    observation = planetmapper.Observation(
        '../data/jupiter.jpg',
        target='jupiter',
        utc='2020-08-25 02:30:40',
        observer='HST',
        show_progress=True,  # so show progress bars for slower functions
    )
    observation.set_disc_params(578, 511, 351, 352.7)

    # observation.run_gui()
    # print(observation.get_disc_params())

    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 8), dpi=200, width_ratios=[1, 2]
    )

    # Do a nice RGB plot of the data in the top left
    rgb_img = np.moveaxis(observation.data, 0, 2)  # imshow needs wavelength index last
    axs[0, 0].imshow(rgb_img, origin='lower')
    observation.plot_wireframe_xy(axs[0, 0])

    # Plot the emission angle backplane in the bottom left
    observation.add_other_bodies_of_interest('Europa')  # mark Europa on this plot
    observation.plot_backplane_img('EMISSION', ax=axs[1, 0])

    # Plot the mapped emission angle backplane in the bottom right
    observation.plot_backplane_map('EMISSION', ax=axs[1, 1])

    # Plot a mapped RGB image of the data in the top right
    _degree_interval = 0.25  # Plot maps with 4 pixels/degree
    emission_cutoff = 80

    mapped_data = observation.get_mapped_data(_degree_interval)  # get the mapped data
    rgb_map = np.moveaxis(mapped_data, 0, 2)  # imshow needs wavelength index last
    rgb_map = planetmapper.utils.normalise(rgb_map)  # normalise to make plot look nicer

    # Only plot areas with emission angles <80deg
    emission_map = observation.get_backplane_map('EMISSION', _degree_interval)
    for idx in range(3):
        rgb_map[:, :, idx][np.where(emission_map > emission_cutoff)] = 1

    # Display mapped image and add a useful annotation
    observation.plot_map(rgb_map, ax=axs[0, 1])
    axs[0, 1].annotate(
        f'Showing emission angles < {emission_cutoff}°',
        (0.005, 0.99),
        xycoords='axes fraction',
        size='small',
        va='top',
    )

    # Add some general formatting
    for ax in axs.ravel():
        ax.set_title('')
    fig.suptitle(observation.get_description(multiline=False))
    fig.tight_layout()

    fig.savefig(os.path.join(PLOT_DIRECTORY, 'jupiter_mapped.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
