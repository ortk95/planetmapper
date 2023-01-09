import unittest
import planetmapper
import numpy as np
import datetime


class TestBody(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.Body('jupiter', generate_dtm_str())

    def test_round_trip_conversion(self):
        lon0, lat0 = generate_lonlat(self.obj)
        with self.subTest('radec'):
            lon1, lat1 = self.obj.radec2lonlat(*self.obj.lonlat2radec(lon0, lat0))
            self.assertAlmostEqual(lon0, lon1, delta=0.1)
            self.assertAlmostEqual(lat0, lat1, delta=0.1)
        with self.subTest('targvec'):
            lon1, lat1 = self.obj.targvec2lonlat(self.obj.lonlat2targvec(lon0, lat0))
            self.assertAlmostEqual(lon0, lon1, delta=0.1)
            self.assertAlmostEqual(lat0, lat1, delta=0.1)

    def test_subpoint_visible(self):
        self.assertTrue(
            self.obj.test_if_lonlat_visible(
                self.obj.subpoint_lon, self.obj.subpoint_lat
            )
        )

    def test_missing(self):
        ra = self.obj._subpoint_ra + 180  # this should always miss
        dec = self.obj._subpoint_dec
        self.assertTrue(all(np.isnan(self.obj.radec2lonlat(ra, dec))))

    def test_distance(self):
        self.assertAlmostEqual(
            self.obj.target_distance,
            np.linalg.norm(self.obj._target_obsvec),
            places=3,
        )
        self.assertAlmostEqual(
            self.obj.subpoint_distance,
            np.linalg.norm(self.obj._subpoint_rayvec),
            places=3,
        )

    def test_radians2degrees(self):
        self.assertEqual(self.obj._degree_pair2radians(0, 180), (0, np.pi))

    def test_degrees2radians(self):
        self.assertEqual(self.obj._radian_pair2degrees(0, np.pi), (0, 180))

    def test_poles_to_plot(self):
        self.assertTrue(len(self.obj.get_poles_to_plot()) > 0)

    def test_encoded_strings_for_spice(self):
        for k, v in self.obj.__dict__.items():
            if k.startswith('_') and k.endswith('_encoded'):
                with self.subTest(k):
                    self.assertIsInstance(v, bytes)
                    s = getattr(self.obj, k[1 : -len('_encoded')])
                    self.assertEqual(v, self.obj._encode_str(s))


class TestBodyXY_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.BodyXY('saturn', generate_dtm_str(), nx=0, ny=0)

    def test_round_trip_conversion(self):
        lon, lat = generate_lonlat(self.obj)
        lon_rt, lat_rt = self.obj.xy2lonlat(*self.obj.lonlat2xy(lon, lat))
        self.assertAlmostEqual(lon, lon_rt, delta=0.1)
        self.assertAlmostEqual(lat, lat_rt, delta=0.1)

    def test_missing(self):
        x = self.obj.get_x0()
        y = self.obj.get_x0()
        x = x + self.obj.get_r0() * 10
        self.assertTrue(all(np.isnan(self.obj.xy2lonlat(x, y))))

    def test_backplane_error(self):
        with self.assertRaises(ValueError):
            self.obj.get_ra_img()


class TestBodyXY_Sized(unittest.TestCase):
    def setUp(self):
        self.nx = 5
        self.ny = 10
        self.obj = planetmapper.BodyXY('neptune', generate_dtm_str(), nx=self.nx, ny=self.ny)

    def test_round_trip_conversion(self):
        lon, lat = generate_lonlat(self.obj)
        lon_rt, lat_rt = self.obj.xy2lonlat(*self.obj.lonlat2xy(lon, lat))
        self.assertAlmostEqual(lon, lon_rt, delta=0.1)
        self.assertAlmostEqual(lat, lat_rt, delta=0.1)

    def test_missing(self):
        x = self.obj.get_x0()
        y = self.obj.get_x0()
        x = x + self.obj.get_r0() * 10
        self.assertTrue(all(np.isnan(self.obj.xy2lonlat(x, y))))

    def test_backplane(self):
        img = self.obj.get_ra_img()
        self.assertTupleEqual(img.shape, (self.ny, self.nx))


def generate_dtm_str() -> str:
    """Create datetime string such that tests are reproducable on same day"""
    dtm = datetime.datetime(2000, 1,1)
    return dtm.strftime('%Y-%m-%d 00:00:00')


def generate_lonlat(body: planetmapper.Body) -> tuple[float, float]:
    seed = 0
    rng = np.random.default_rng(seed)
    while True:
        # Avoid lons near meridian where wraparound (e.g. 359.9999<->0.0001) can cause
        # unit tests to incorrectly fail
        lon = rng.random() * 358 + 1

        # Avoid polar latitudes where singularity can break 1-to-1 conversions
        lat = rng.random() * 120 - 60

        if body.test_if_lonlat_visible(lon, lat):
            return lon, lat


if __name__ == '__main__':
    unittest.main()
