import unittest
import planetmapper
import numpy as np
from test_locations import generate_dtm_str


class TestOptimization(unittest.TestCase):
    def setUp(self):
        body = 'jupiter'
        utc = generate_dtm_str()
        self.fast = planetmapper.BodyXY(body, utc, optimize_speed=True)
        self.slow = planetmapper.BodyXY(body, utc, optimize_speed=False)

        for obj in (self.fast, self.slow):
            obj.set_img_size(10, 11)
            obj.set_r0(4.123)
            obj.set_x0(5.456)
            obj.set_y0(5.789)

    def test_optimization_set_up(self):
        self.assertTrue(self.fast._optimize_speed)
        self.assertFalse(self.slow._optimize_speed)

    def test_string_encoding_different(self):
        """Tests optimization is running as expected"""
        self.assertNotEqual(self.fast._target_encoded, self.slow._target_encoded)

    def test_targvec_img(self):
        self.assertTrue(
            np.array_equal(
                self.fast._get_targvec_img(),
                self.slow._get_targvec_img(),
                equal_nan=True,
            )
        )

    def test_lon_img(self):
        self.assertTrue(
            np.array_equal(
                self.fast.get_lon_img(),
                self.slow.get_lon_img(),
                equal_nan=True,
            )
        )

    def test_doppler_img(self):
        self.assertTrue(
            np.array_equal(
                self.fast.get_doppler_img(),
                self.slow.get_doppler_img(),
                equal_nan=True,
            )
        )


if __name__ == '__main__':
    unittest.main()
