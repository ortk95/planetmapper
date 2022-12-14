"""
Helpful utilities for testing scripts not in planetmapper.

Mainly stuff from `tools` in https://github.com/ortk95/astro-tools and PhD repo.

TODO clean this and remove testing stuff for final version.
"""
import os
import traceback
import subprocess
import numpy as np
import warnings
from datetime import datetime, timedelta


def cprint(*msg, fg=None, bg=None, style=None, skip_print=False, sep=' ', **kwargs):
    """
    Prints coloured and formatted text.

    Parameters
    ----------
    msg
        Message to print.

    fg, bg : {'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w'}
        Foreground and background colours (see code for options).

    style : {'b', 'f', 'i', 'u', 'x', 'y', 'r', 'h', 's'}
        Formatting style to apply to text. Can be multiple values, e.g. 'bi'
        for bold and italic style.

    """
    colcode = {
        'k': 0,  # black
        'r': 1,  # red
        'g': 2,  # green
        'y': 3,  # yellow
        'b': 4,  # blue
        'm': 5,  # magenta
        'c': 6,  # cyan
        'w': 7,  # white
    }

    fmtcode = {
        'b': 1,  # bold
        'f': 2,  # faint
        'i': 3,  # italic
        'u': 4,  # underline
        'x': 5,  # blinking
        'y': 6,  # fast blinking
        'r': 7,  # reverse
        'h': 8,  # hide
        's': 9,  # strikethrough
    }

    # Properties
    props = []
    if isinstance(style, str):
        props = [fmtcode[s] for s in style]
    if isinstance(fg, str):
        props.append(30 + colcode[fg])
    if isinstance(bg, str):
        props.append(40 + colcode[bg])

    # Display
    msg = sep.join(
        str(x) for x in msg
    )  # Reproduce print() behaviour for easy translation
    props = ';'.join(str(x) for x in props)

    if props:
        msg = '\x1b[%sm%s\x1b[0m' % (props, msg)

    if not skip_print:
        print(msg, **kwargs)

    return msg


def print_bar_chart(
    labels,
    bars=None,
    formats=None,
    print_values=True,
    max_label_length=None,
    sort=False,
    **kwargs,
):
    """
    Print bar chart of data

    Parameters
    ----------
    labels : array
        Labels of bars, or bar values if `bars is None`.

    bars : array
        List of bar lengths.

    formats : array
        List of bar formats to be passed to `cprint()`.

    print_values : bool
        Toggle printing bar lengths.

    max_label_length : int
        Set length to trim labels, None for no trimming.

    sort : bool
        Toggle sorting of bars by size.

    **kwargs
        Arguments passed to `cprint()` for every bar.
    """
    if bars is None:
        bars = labels.copy()
        labels = ['' for _ in bars]
    bars = list(bars)
    labels = [str(l) for l in labels]
    if max_label_length is None:
        max_label_length = max([len(l) for l in labels] + [1])
    else:
        labels = clip_string_list(labels, max_label_length)
    labels = [f'{l:{max_label_length}s}' for l in labels]
    if sort:
        if formats is not None and not isinstance(formats, str):
            bars, labels, formats = zip(*sorted(zip(bars, labels, formats)))
        else:
            bars, labels = zip(*sorted(zip(bars, labels)))
    if print_values:
        fmt = '.2e'
        if isinstance(print_values, str):
            fmt = print_values
        value_strs = [f'{v:{fmt}}' for v in bars]
        labels = [f'{l}|{v}' for l, v in zip(labels, value_strs)]
    max_label_length = max([len(l) for l in labels])
    max_length = get_console_width() - max_label_length - 2
    for idx, label in enumerate(labels):
        kw = {**kwargs}
        if formats:
            if formats == 'auto':
                if bars[idx] / sum(bars) > 0.5:
                    kw.update(fg='y', style='b')
                elif bars[idx] / sum(bars) > 0.1:
                    kw.update(fg='g', style='b')
                elif bars[idx] / sum(bars) > 0.01:
                    kw.update(fg='b', style='b')
                else:
                    kw.update(fg='w', style='f')
            elif formats == 'extreme':
                if bars[idx] == max(bars):
                    kw.update(fg='g', style='b')
                elif bars[idx] == min(bars):
                    kw.update(fg='r', style='b')
                else:
                    kw.update(fg='b', style='b')
            else:
                kw.update(formats[idx])

        chrs = ' ????????????????????????'
        length = max_length * bars[idx] / max(bars)
        decimal_idx = (length - int(length)) * (len(chrs) - 1)
        decimal_idx = int(np.round(decimal_idx))
        bar = chrs[-1] * int(length) + chrs[decimal_idx]
        bar = bar.rstrip(' ')
        cprint(f'{label:{max_label_length}s}|{bar}', **kw)


def get_console_width(fallback=75, maximum=98):
    """
    Attempts to find console width, otherwise uses fallback provided.

    Parameters
    ----------
    fallback : int
        Default width value if `stty size` fails.

    Returns
    -------
    width : int
        Console width.
    """
    if test_if_ipython():
        return fallback
    try:
        _, width = subprocess.check_output(
            ['stty', 'size'], stderr=subprocess.PIPE
        ).split()
    except:
        width = fallback
    width = int(width)
    if maximum and width > maximum:
        width = maximum
    return width


def test_if_ipython():
    """Detect if script is running in IPython console"""
    try:
        return __IPYTHON__  # type: ignore
    except NameError:
        return False


def clip_string(s, max_len, continue_str='???'):
    """
    Takes string and clips to certain length if needed.

    Parameters
    ----------
    s : str
        String to clip

    max_len : int
        Maximum allowed string length

    continue_str : str
        String used to indicate string has been clipped

    Returns
    -------
    clipped string
    """
    return s if len(s) <= max_len else s[: max_len - len(continue_str)] + continue_str


def clip_string_list(a, max_len, **kwargs):
    """
    Takes a list of strings and clips them to a certain length if needed.

    Parameters
    ----------
    a : list of str

    Other parameters passed to clip_string()

    Returns
    -------
    clipped list
    """
    return [clip_string(s, max_len, **kwargs) for s in a]
