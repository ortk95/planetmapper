import unittest
import planetmapper
import datetime
import numpy as np


class TestBodyXY(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.BodyXY('jupiter', '2000-01-01')

    def test_x0(self):
        v = np.random.rand() * 100
        self.obj.set_x0(v)
        self.assertEqual(self.obj.get_x0(), v)

    def test_y0(self):
        v = np.random.rand() * 100
        self.obj.set_y0(v)
        self.assertEqual(self.obj.get_y0(), v)

    def test_r0(self):
        v = np.random.rand() * 100
        self.obj.set_r0(v)
        self.assertEqual(self.obj.get_r0(), v)

    def test_rotation(self):
        v = np.random.rand() * 359
        self.obj.set_rotation(v)
        self.assertAlmostEqual(self.obj.get_rotation(), v)

    def test_params(self):
        with self.subTest('args'):
            x0, y0, r0, rotation = np.random.rand(4) * 100
            self.obj.set_disc_params(x0, y0, r0, rotation)
            self.assertEqual(self.obj.get_x0(), x0)
            self.assertEqual(self.obj.get_y0(), y0)
            self.assertEqual(self.obj.get_r0(), r0)
            self.assertAlmostEqual(self.obj.get_rotation(), rotation)
        with self.subTest('kwargs'):
            x0, y0, r0, rotation = np.random.rand(4) * 100
            self.obj.set_disc_params(x0=x0, y0=y0, r0=r0, rotation=rotation)
            self.assertEqual(self.obj.get_x0(), x0)
            self.assertEqual(self.obj.get_y0(), y0)
            self.assertEqual(self.obj.get_r0(), r0)
            self.assertAlmostEqual(self.obj.get_rotation(), rotation)

    def test_img_size(self):
        nx = np.random.randint(1, 100)
        ny = np.random.randint(1, 100)
        self.obj.set_img_size(nx=nx, ny=ny)
        self.assertTupleEqual(self.obj.get_img_size(), (nx, ny))

    def test_clear_cache(self):
        self.obj._cache[' test '] = None
        self.obj._clear_cache()
        self.assertEqual(len(self.obj._cache), 0)

        for fn in (
            self.obj.set_x0,
            self.obj.set_r0,
            self.obj.set_y0,
            self.obj.set_rotation,
        ):
            with self.subTest(fn.__name__):
                self.obj._cache[' test '] = None
                fn(np.random.rand())
                self.assertEqual(len(self.obj._cache), 0)

    def test_nx_ny_error(self):
        self.obj.set_img_size(0, 0)
        with self.assertRaises(ValueError):
            self.obj.get_lon_img()


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.Observation(
            data=np.random.rand(1, 5, 5), target='Jupiter', utc='2022-01-01'
        )

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            planetmapper.Observation()

        with self.assertRaises(ValueError):
            planetmapper.Observation(path='test', data=np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
