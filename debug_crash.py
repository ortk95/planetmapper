#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import sys
from types import FrameType
from typing import Any

import planetmapper

_DO_QUALNAME = sys.version_info >= (3, 11)  # co_qualname was added in 3.11

IGNORE_NAMES: set[str] = {'_iterate_image'}


def main():
    print('Adding trace for planetmapper functions')
    print('Ignoring: ', IGNORE_NAMES)
    sys.setprofile(tracefunc)

    gui = planetmapper.gui.GUI()
    gui.run()


def tracefunc(frame: FrameType, event: str, arg: Any):
    filename = frame.f_code.co_filename
    if 'planetmapper' not in filename:
        return
    if frame.f_code.co_name in IGNORE_NAMES:
        return
    n = frame.f_code.co_qualname if _DO_QUALNAME else frame.f_code.co_name
    name = f'{n} {os.path.basename(filename)}:{frame.f_lineno}'
    indent = '-' * tracefunc.indent
    if event == 'call':
        print_message(indent + '>', name)
        tracefunc.indent += 1
    elif event == 'return':
        print_message('<' + indent, name)
        tracefunc.indent -= 1
    return tracefunc


tracefunc.indent = 0


def print_message(*msg: Any):
    print(datetime.datetime.now(), *msg, flush=True)


if __name__ == '__main__':
    main()
