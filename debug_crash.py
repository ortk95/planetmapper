#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import sys
from types import FrameType
from typing import Any

import planetmapper


def main():
    sys.setprofile(tracefunc)

    gui = planetmapper.gui.GUI()
    gui.run()


def tracefunc(frame: FrameType, event: str, arg: Any):
    filename = frame.f_code.co_filename
    if 'planetmapper' not in filename:
        return

    name = f'{frame.f_code.co_name} {os.path.basename(filename)}:{frame.f_lineno}'
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
