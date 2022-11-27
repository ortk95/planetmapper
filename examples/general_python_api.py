#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple example script"""
import os
import matplotlib.pyplot as plt
import planetmapper
import planetmapper.data_loader

PLOT_DIRECTORY = '../docs/images'

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

if True:
    fig, [ax_radec, ax_km] = plt.subplots(nrows=2, figsize=(6, 8), dpi=200)

    dates = ['2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00']
    colors = ['r', 'g', 'b']

    for date, c in zip(dates, colors):
        body = planetmapper.Body('jupiter', date)
        body.add_other_bodies_of_interest('Io')
        body.plot_wireframe_radec(ax_radec, color=c)
        body.plot_wireframe_km(ax_km, color=c)

        # Plot some blank data with the correct colour to go on the legend
        ax_radec.scatter(float('nan'), float('nan'), color=c, label=date)
    ax_radec.legend(loc='upper left')

    ax_radec.set_title('plot_wireframe_radec(...)')
    ax_km.set_title('plot_wireframe_km(...)')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'jupiter_wireframes.png'))
    plt.close(fig)
