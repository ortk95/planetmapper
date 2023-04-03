import datetime
import unittest

import common_testing
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, nan
from spiceypy.utils.exceptions import NotFoundError, SpiceSPKINSUFFDATA

import planetmapper
import planetmapper.base
import planetmapper.progress
from planetmapper import Body


class TestObservation(unittest.TestCase):
    def setUp(self) -> None:
        pass