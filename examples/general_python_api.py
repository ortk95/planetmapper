#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to generate example plots on 
https://planetmapper.readthedocs.io/en/latest/general_python_api.html
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..')  # Use local dev version of planetmapper
import planetmapper

PLOT_DIRECTORY = '../docs/images'

# Code for each plot is wrapped in an `if True` block for easy organisation and toggling


if False:
    body = planetmapper.Body('saturn', '2020-01-01')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    body.plot_wireframe_radec(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'saturn_wireframe_radec.png'))
    plt.close(fig)


if False:
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


if False:
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


if False:
    observation = planetmapper.Observation('../data/europa.fits.gz')

    # Set the disc position
    observation.set_plate_scale_arcsec(12.25e-3)
    observation.set_disc_params(x0=110, y0=104)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    observation.plot_backplane_img('LON-GRAPHIC', ax=ax)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'europa_backplane.png'))
    plt.close(fig)


if False:
    # Create an object representing how Jupiter would appear in a 50x50 pixel image
    # taken by JWST at a specific time
    body = planetmapper.BodyXY('jupiter', utc='2024-01-01', observer='JWST', sz=50)
    body.set_disc_params(x0=25, y0=25, r0=20)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    body.plot_backplane_img('RADIAL-VELOCITY', ax=ax)

    radial_velocities = body.get_backplane_img('RADIAL-VELOCITY')
    print(f'Average radial velocity: {np.nanmean(radial_velocities):.2f} km/s')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'jupiter_backplane.png'))
    plt.close(fig)

if True:
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
