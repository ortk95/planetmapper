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
        lon_rt, lat_rt = self.body.radec2lonlat_radians(
            *self.body.lonlat2radec_radians(lon, lat)
        )
        self.assertAlmostEqual(lon, lon_rt, places=3)
        self.assertAlmostEqual(lat, lat_rt, places=3)

    def test_missing(self):
        """
        Test ra/dec that should miss the target.
        """
        ra = self.body.subpoint_ra + np.pi  # this should always miss
        dec = self.body.subpoint_dec
        self.assertTrue(all(np.isnan(self.body.radec2lonlat_radians(ra, dec))))

    def test_distance(self):
        self.assertAlmostEqual(
            self.body.target_distance,
            np.linalg.norm(self.body._target_obsvec),
            places=3,
        )
        self.assertAlmostEqual(
            self.body.subpoint_distance,
            np.linalg.norm(self.body._subpoint_rayvec),
            places=3,
        )

    def test_radians2degrees(self):
        self.assertEqual(self.body._degree_pair2radians(0, 180), (0, np.pi))

    def test_degrees2radians(self):
        self.assertEqual(self.body._radian_pair2degrees(0, np.pi), (0, 180))


class TestObservation(unittest.TestCase):
    def setUp(self):
        dtm = datetime.datetime.now()
        dtm_str = dtm.strftime('%Y-%m-%d %H:%M:%S')
        self.observation = mapper.Observation('saturn', dtm_str)

    def test_round_trip_conversion(self):
        """
        Test that conversion lon/lat -> ra/dec -> lon/lat gives consistent result.
        """
        lon, lat = generate_lonlat(self.observation)
        lon_rt, lat_rt = self.observation.xy2lonlat_radians(
            *self.observation.lonlat2xy_radians(lon, lat)
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

        self.assertTrue(all(np.isnan(self.observation.xy2lonlat_radians(x, y))))


def generate_lonlat(body: mapper.Body) -> tuple[float, float]:
    """Choose a random point on the surface that's visible"""
    # Use deterministic seed so tests are reproducable (on the same day)
    seed = (datetime.datetime.now() - datetime.datetime(2000, 1, 1)).days
    rng = np.random.default_rng(seed)
    while True:
        # Avoid lons near meridian where wraparound (e.g. 359.9999<->0.0001) can cause
        # unit tests to incorrectly fail
        lon = np.deg2rad(rng.random() * 358 + 1)

        # Avoid polar latitudes where singularity can break 1-to-1 conversions
        lat = np.deg2rad(rng.random() * 120 - 60)

        if body.test_if_lonlat_visible_radians(lon, lat):
            return lon, lat


if __name__ == '__main__':
    unittest.main()
