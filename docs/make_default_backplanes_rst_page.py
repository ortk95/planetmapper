import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.split(__file__)[0], '..'))
import numpy as np

import planetmapper

IMAGE_DIR = 'images/generated'
REF_DATE = '2000-01-01T06:00'
BODY_SIZE = 50


def make_page() -> None:
    print('\nGenerating default backplanes page and images...', flush=True)
    body = setup_kernels_and_get_body()

    page_content = get_page_content_from_body(body)
    page_path = Path(__file__).parent / 'default_backplanes.rst'
    with open(page_path, 'w') as f:
        print(f'Writing to {page_path}', flush=True)
        f.write(page_content)
    make_images_for_page(body)

    print('Finished generating default backplanes page and images\n', flush=True)


def get_page_content_from_body(body: planetmapper.BodyXY) -> str:
    msg: list[str] = []
    msg.append('..')
    msg.append('   THIS CONTENT IS AUTOMATICALLY GENERATED')
    msg.append('')

    msg.append('.. _default backplanes:')
    msg.append('')
    msg.append('Default backplanes')
    msg.append('*' * len(msg[-1]))
    msg.append('')

    msg.append(
        'This page lists the backplanes that are automatically registered to '
        'every instance of :class:`planetmapper.BodyXY`.'
    )
    msg.append('')
    msg.append(
        'Backplane images can be generated using :func:`planetmapper.BodyXY.get_backplane_img`'
        ' and the mapped data can be generated using :func:`planetmapper.BodyXY.get_backplane_map`.'
    )
    msg.append('')
    msg.append(
        'When an observation is saved, these backplanes are automatically included as'
        ' extensions to the FITS file. Files can be saved using:'
    )
    msg.append('')
    msg.append(' * :func:`planetmapper.Observation.save_observation`')
    msg.append(' * :func:`planetmapper.Observation.save_mapped_observation`')
    msg.append(
        ' * or the :ref:`PlanetMapper Graphical User Interface (GUI) <gui examples>`.'
    )
    msg.append('')

    msg.append('Backplanes')
    msg.append('=' * len(msg[-1]))
    msg.append('')
    msg.append(
        f'The plots here show the backplanes for an example {BODY_SIZE}x{BODY_SIZE}'
        f' pixel observation of Jupiter observed from the Earth on {REF_DATE}.'
    )
    msg.append('')
    msg.append('------------')
    msg.append('')

    for bp in body.backplanes.values():
        msg.append('`{}` {}'.format(bp.name, bp.description))
        msg.append('-' * len(msg[-1]))
        msg.append('')
        msg.append('.. image:: {}/{}'.format(IMAGE_DIR, get_backplane_img_filename(bp)))
        msg.append('    :width: 100%')
        msg.append('    :alt: Example image of the {} backplane'.format(bp.name))
        msg.append('')
        msg.append(
            '*Functions:* :func:`planetmapper.{}`, :func:`planetmapper.{}`'.format(
                bp.get_img.__qualname__,
                bp.get_map.__qualname__,
            )
        )
        msg.append('')
        if bp.name == 'LON-GRAPHIC':
            msg.append(
                'Note that the positive longitude direction is different for some '
                'bodies (see :attr:`planetmapper.Body.positive_longitude_direction`).'
            )
            msg.append('')
        msg.append('------------')
        msg.append('')
    msg.append('')
    msg.append('Wireframe images')
    msg.append('=' * len(msg[-1]))
    msg.append('')

    msg.append(
        'In addition to the above backplanes, a `WIREFRAME` backplane is also included '
        'by default in saved FITS files. This backplane contains a "wireframe" image '
        'of the body, which shows latitude/longitude gridlines, labels poles, displays '
        'the body\'s limb etc. These wireframe images can be used to help orient the '
        'observations, and can be used as an overlay if you are creating figures from '
        'the FITS files.'
    )
    msg.append('')

    msg.append(
        'The wireframe images are a graphical guide rather than containing any '
        'scientific data, so they are not registered like the other backplanes. '
        'Note that the wireframe images have a fixed size, so they will not be the '
        'same size as the data/mapped data (although the aspect ratio will be the '
        'same).'
    )
    msg.append('')

    msg.append(
        '- Image function: :func:`planetmapper.{}`'.format(
            body.get_wireframe_overlay_img.__qualname__
        )
    )
    msg.append(
        '- Map function: :func:`planetmapper.{}`'.format(
            body.get_wireframe_overlay_map.__qualname__
        )
    )
    msg.append('')
    return '\n'.join(msg)


def make_images_for_page(body: planetmapper.BodyXY) -> None:
    images_root = Path(__file__).parent / IMAGE_DIR
    images_root.mkdir(parents=True, exist_ok=True)

    for backplane in body.backplanes.values():
        image_path = images_root / get_backplane_img_filename(backplane)
        print(f'Generating {image_path}', flush=True)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15, 5),
            dpi=200,
            width_ratios=[1, 2],
            gridspec_kw=dict(
                left=0.06,
                right=0.86,
                top=0.85,
                bottom=0.15,
                hspace=0.075,
            ),
        )
        backplane_img = backplane.get_img()
        backplane_map = backplane.get_map()

        vmin = np.nanmin([np.nanmin(backplane_img), np.nanmin(backplane_map)])
        vmax = np.nanmax([np.nanmax(backplane_img), np.nanmax(backplane_map)])

        body.plot_img(backplane_img, ax=axs[0], vmin=vmin, vmax=vmax, cmap='viridis')
        sm = body.plot_map(
            backplane_map, ax=axs[1], vmin=vmin, vmax=vmax, cmap='viridis'
        )

        pos = axs[1].get_position()
        cax = fig.add_axes((pos.x1 + 0.05, pos.y0, 0.02, pos.height))
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(backplane.description)

        fig.suptitle(
            f'{backplane.name} backplane in PlanetMapper', y=0.95, size='x-large'
        )
        for ax in axs:
            ax.set_title('')

        fig.savefig(str(image_path))
        plt.close(fig)


def get_backplane_img_filename(backplane: planetmapper.Backplane) -> str:
    return f'backplane_{backplane.name.lower()}.png'


def setup_kernels_and_get_body() -> planetmapper.BodyXY:
    planetmapper.base.clear_kernels()
    planetmapper.set_kernel_path(
        Path(__file__).parent.parent / 'tests' / 'data' / 'kernels'
    )
    body = planetmapper.BodyXY('Jupiter', utc=REF_DATE, sz=BODY_SIZE)
    # Ensure the radec and xy axes are not aligned by rotating the body slightly
    body.set_rotation(15)
    return body
