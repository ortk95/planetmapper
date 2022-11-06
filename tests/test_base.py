import unittest
import planetmapper
import datetime
import numpy as np
from numpy import array, nan


class TestSpiceTool(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.SpiceBase()

    def test_speed_of_light(self):
        self.assertEqual(self.obj.speed_of_light(), 299792.458)

    def test_standardise(self):
        self.assertEqual(self.obj.standardise_body_name(' JuPiTeR   '), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name('599'), 'JUPITER')
        self.assertEqual(self.obj.standardise_body_name(599), 'JUPITER')

    def test_time(self):
        pairs = (
            (
                -999999999,
                datetime.datetime(
                    1968, 4, 24, 10, 12, 39, 814453, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                0,
                datetime.datetime(
                    2000, 1, 1, 11, 58, 55, 816073, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                42,
                datetime.datetime(
                    2000, 1, 1, 11, 59, 37, 816073, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                123456789,
                datetime.datetime(
                    2003, 11, 30, 9, 32, 4, 816943, tzinfo=datetime.timezone.utc
                ),
            ),
            (
                0.123456789,
                datetime.datetime(
                    2000, 1, 1, 11, 58, 55, 939530, tzinfo=datetime.timezone.utc
                ),
            ),
        )
        for et, dtm in pairs:
            with self.subTest(f'et = {et}'):
                self.assertEqual(self.obj.et2dtm(et), dtm)

    def test_close_loop(self):
        self.assertTrue(
            np.array_equal(
                self.obj.close_loop(np.array([0, 1, 2, 3, 4, 5])),
                np.array([0, 1, 2, 3, 4, 5, 0]),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.obj.close_loop(np.array([[1.1, 1.2], [2.2, 2.3]])),
                np.array([[1.1, 1.2], [2.2, 2.3], [1.1, 1.2]]),
            )
        )

    def test_unit_vector(self):
        a = np.random.rand(3) * 999
        ahat = self.obj.unit_vector(a)
        self.assertAlmostEqual(np.linalg.norm(ahat), 1)

    def test_angle_conversion(self):
        pair = np.random.rand(2)
        self.assertTrue(
            np.array_equal(self.obj._radian_pair2degrees(*pair), np.rad2deg(pair))
        )
        self.assertTrue(
            np.array_equal(self.obj._degree_pair2radians(*pair), np.deg2rad(pair))
        )
        self.assertTrue(
            np.allclose(
                self.obj._degree_pair2radians(*self.obj._radian_pair2degrees(*pair)),
                pair,
            )
        )


if __name__ == '__main__':
    unittest.main()
