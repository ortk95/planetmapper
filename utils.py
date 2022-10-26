"""
Helpful utilities

Mainly stuff from `tools` in https://github.com/ortk95/astro-tools and PhD repo.

TODO clean this and remove testing stuff for final version.
"""
import os
import traceback
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


def print_progress(annotation=None, c1='g', c2='k', style='b'):
    """
    Print progress summary of current code.

    Prints summary of code location and execution time for use in optimising and monitoring code.
    Uses traceback to identify the call stack and prints tree-like diagram of stacks where this
    function was called. The call stack is relative to the first time this function is called as it
    uses properties of print_progress to communicate between calls.

    Printed output contains:
        - Seconds elapsed since last call of print_progress.
        - Current time.
        - Traceback for location print_progress is called from, relative to first location
          print_progress was called. This includes file names, function names and line numbers.
        - Optional annotation provided to explain what current line is.

    Arguments
    ---------
    annotation : str
        Optionally provide annotation about current line of code.

    c1, c2, style : str
        Formatting options to pass to cprint()
    """
    now = datetime.now()  # Ignore duration of current code

    # Get timings
    if print_progress.last_dtm is None:
        title = ' seconds'
    else:
        td = now - print_progress.last_dtm
        title = f'{td.total_seconds():8.2f}'
    title += ' @ '
    title += now.strftime('%H:%M:%S')
    title += ' '

    # Get stack
    stack = traceback.extract_stack()[:-1]
    if print_progress.first_stack is None:
        print_progress.first_stack = stack
    first_stack = print_progress.first_stack
    split_idx = len([None for a, b in zip(stack[:-1], first_stack) if a == b])
    stack = stack[split_idx:]
    stack_parts = [(s[2], s[1], os.path.split(s[0])[1]) for s in stack]
    stack_text = [
        f'{a} (line {b} in <ipython>)'
        if c.startswith('<ipython-input-')
        else f'{a} (line {b} in {c})'
        for a, b, c in stack_parts
    ]
    last_stack = print_progress.last_stack
    if last_stack is not None:
        last_idx = len([None for a, b in zip(stack[:-1], last_stack) if a == b])
        stack_text = stack_text[last_idx:]
    else:
        last_idx = 0
    print_progress.last_stack = stack

    # Do actual printing
    for st in stack_text:
        msg = cprint(
            ' ' + '│  ' * last_idx + st + ' ',
            fg=c1,
            bg=c2,
            style=style,
            skip_print=True,
        )
        if st == stack_text[-1]:
            tt = title
            if annotation:
                msg += cprint(
                    f'{annotation} ', fg=c1, bg=c2, style=style + 'i', skip_print=True
                )
        else:
            tt = ' ' * len(title)
        tt = cprint(tt, fg=c2, bg=c1, style=style, skip_print=True)
        print(tt + msg, flush=True)
        last_idx += 1
    print_progress.last_dtm = datetime.now()  # Ignore duration of this code


print_progress.last_dtm = None
print_progress.last_stack = None
print_progress.first_stack = None
