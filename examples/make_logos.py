#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create various logos for PlanetMapper

Run download_spice_kernels() to download the kernels required to generate the logos.
"""
import os
import sys

import matplotlib.patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from general_python_api import PLOT_DIRECTORY

sys.path.insert(0, '..')  # Use local dev version of planetmapper
import planetmapper
import planetmapper._assets
from planetmapper.kernel_downloader import download_urls

BG_COLOR = 'dodgerblue'
OUTLINE_COLOR = 'royalblue'


def main():
    download_spice_kernels()
    plot_gitub_social_preview()
    plot_logo_wide()
    plot_readthedocs_logo()
    plot_icon()


def download_spice_kernels():
    download_urls(
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat415.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp',
    )


def plot_gitub_social_preview():
    # GitHub recommended image size in px
    w = 1280
    h = 640

    dpi = 200

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    body = planetmapper.BodyXY('saturn', '2020-01-01T05:15', nx=w, ny=h)
    body.set_disc_params(x0=w / 2, y0=h / 2, r0=h * 0.39, rotation=-10)
    body.add_other_bodies_of_interest(
        'pandora', 'aegaeon', 'pan', 'prometheus', 'janus', 'epimetheus', 'enceladus'
    )
    body.plot_wireframe_xy(
        ax,
        add_axis_labels=False,
        add_title=False,
        color='k',
        formatting={
            'pole': dict(
                path_effects=[
                    path_effects.Stroke(linewidth=3, foreground=BG_COLOR),
                    path_effects.Normal(),
                ],
            )
        },
    )
    ax.axis('off')
    ax.annotate(
        'PlanetMapper',
        xy=(0.5, 0.5),
        xycoords='axes fraction',
        ha='center',
        va='center',
        color='w',
        fontsize=57,
        fontweight='bold',
        fontfamily='monospace',
        # rotation=-body.north_pole_angle() - body.get_rotation(),
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground=OUTLINE_COLOR),
            path_effects.Normal(),
        ],
    )

    for obj in (fig, ax):
        obj.set_facecolor(BG_COLOR)
    fig.savefig(os.path.join(PLOT_DIRECTORY, 'logo_github_social_preview.png'))
    plt.show()
    plt.close(fig)


def plot_logo_wide():
    w = 1920
    h = 480
    dpi = 200
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    body = planetmapper.BodyXY('saturn', '2020-01-01T05:15', nx=w, ny=h)
    body.set_disc_params(x0=w / 2, y0=h / 2, r0=h * 0.39, rotation=-10)
    body.add_other_bodies_of_interest(
        'pandora',
        'aegaeon',
        'pan',
        'prometheus',
        'janus',
        'epimetheus',
        'enceladus',
        # 'telesto',
        # 'pallene',
    )
    body.plot_wireframe_xy(
        ax, add_axis_labels=False, add_title=False, color=BG_COLOR, label_poles=False
    )
    ax.annotate(
        'PlanetMapper',
        xy=(0.5, 0.5),
        xycoords='axes fraction',
        ha='center',
        va='center',
        color='w',
        fontsize=65,
        fontweight='bold',
        fontfamily='monospace',
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground=OUTLINE_COLOR),
            path_effects.Normal(),
        ],
    )

    ax.axis('off')
    fig.savefig(
        os.path.join(PLOT_DIRECTORY, 'logo_wide_transparent.png'), transparent=True
    )
    plt.show()
    plt.close(fig)


def plot_readthedocs_logo():
    rtd_color = [x / 255 for x in (41, 128, 185)]
    w = 1280
    h = 480
    dpi = 250
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    body = planetmapper.BodyXY('saturn', '2020-01-01T05:15', nx=w, ny=h)
    body.set_disc_params(x0=w / 2, y0=h / 2, r0=h * 0.45, rotation=-10)
    body.plot_wireframe_xy(
        ax,
        add_axis_labels=False,
        add_title=False,
        color=tuple([(x + 2) / 3 for x in rtd_color]),
        label_poles=False,
    )
    ax.annotate(
        'PlanetMapper',
        xy=(0.5, 0.5),
        xycoords='axes fraction',
        ha='center',
        va='center',
        color='w',
        fontsize=(63 * 200) // dpi,
        fontweight='bold',
        fontfamily='monospace',
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground=rtd_color),
            path_effects.Normal(),
        ],
    )

    ax.axis('off')
    fig.savefig(
        os.path.join(PLOT_DIRECTORY, 'logo_rtd_transparent.png'), transparent=True
    )
    plt.show()
    plt.close(fig)


def plot_icon():
    # The 'standard' Saturn that we use is a bit complicated for a small icon, so choose
    # a time when the rings are more closed, and reduce the number of rings we plot to
    # simplify the image a bit.
    background_color = OUTLINE_COLOR
    wireframe_color = 'w'
    h = 128
    w = 128
    resolution = 100
    fig = plt.figure(figsize=(w / resolution, h / resolution), dpi=1000)
    ax = fig.add_axes([0, 0, 1, 1])
    body = planetmapper.BodyXY('saturn', '2024-04-01T01:00', nx=w, ny=h)
    body.centre_disc()
    body.set_disc_params(r0=h * 0.25, rotation=-30)
    body.ring_radii = {
        74658.0,
        91975.0,
        (117507.0 + 122340.0) / 2,
        136780.0,
    }
    body.plot_wireframe_xy(
        ax,
        add_axis_labels=False,
        add_title=False,
        label_poles=False,
        formatting={
            'grid': dict(linewidth=0.5, linestyle='-', alpha=1),
            'limb': dict(linewidth=2),
            'ring': dict(linewidth=1),
            'terminator': dict(visible=False),
            'limb_illuminated': dict(visible=False),
        },
        color=wireframe_color,
    )
    bg_patch = matplotlib.patches.FancyBboxPatch(
        (-0.5, -0.5),
        w,
        h,
        edgecolor='none',
        facecolor=background_color,
        zorder=0,
        boxstyle=matplotlib.patches.BoxStyle('Round', pad=0, rounding_size=w * 0.2),
    )
    ax.add_patch(bg_patch)
    ax.axis('off')
    paths = [
        os.path.join(PLOT_DIRECTORY, 'icon.png'),
        planetmapper._assets.get_gui_icon_path(),
    ]
    for p in paths:
        fig.savefig(
            p,
            transparent=True,
            dpi=200,
        )
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
