import unittest
import mapper
import numpy as np
import datetime


class TestBody(unittest.TestCase):
    def setUp(self):
        dtm = datetime.datetime.now()
        dtm_str = dtm.strftime('%Y-%m-%d %H:%M:%S')
        self.body = mapper.Body('jupiter', dtm_str)

    def test_round_trip_conversion(self):
        """
        Test that conversion lon/lat -> ra/dec -> lon/lat gives consistent result.
        """
        lon, lat = generate_lonlat(self.body)
        lon_rt, lat_rt = self.body.radec2lonlat(*self.body.lonlat2radec(lon, lat))
        self.assertAlmostEqual(lon, lon_rt, places=3)
        self.assertAlmostEqual(lat, lat_rt, places=3)

    def test_missing(self):
        """
        Test ra/dec that should miss the target.
        """
        ra = self.body.subpoint_ra + np.pi  # this should always miss
        dec = self.body.subpoint_dec
        self.assertTrue(all(np.isnan(self.body.radec2lonlat(ra, dec))))


class TestObservation(unittest.TestCase):
    def setUp(self):
        dtm = datetime.datetime.now()
        dtm_str = dtm.strftime('%Y-%m-%d %H:%M:%S')
        self.observation = mapper.Observation('jupiter', dtm_str)

    def test_round_trip_conversion(self):
        """
        Test that conversion lon/lat -> ra/dec -> lon/lat gives consistent result.
        """
        lon, lat = generate_lonlat(self.observation)
        lon_rt, lat_rt = self.observation.xy2lonlat(
            *self.observation.lonlat2xy(lon, lat)
        )
        self.assertAlmostEqual(lon, lon_rt, places=3)
        self.assertAlmostEqual(lat, lat_rt, places=3)

    def test_missing(self):
        """
        Test ra/dec that should miss the target.
        """
        x = self.observation.get_x0()
        y = self.observation.get_x0()
        x = x + self.observation.get_r0() * 10

        self.assertTrue(all(np.isnan(self.observation.xy2lonlat(x, y))))


def generate_lonlat(body: mapper.Body) -> tuple[float, float]:
    """Choose a random point on the surface that's visible"""
    # Use deterministic seed so tests are reproducable (on the same day)
    seed = (datetime.datetime.now() - datetime.datetime(2000, 1, 1)).days
    rng = np.random.default_rng(seed)
    while True:
        lon = np.deg2rad(rng.random() * 360)
        lat = np.deg2rad(rng.random() * 120 - 60)
        # avoid polar latitudes where singularity can break 1-to-1 conversions
        if body.test_if_lonlat_visible(lon, lat):
            return lon, lat


if __name__ == '__main__':
    unittest.main()
