import unittest
import mapper
import numpy as np
import datetime


class TestBodyRandomLocation(unittest.TestCase):
    def setUp(self):
        self.body = mapper.Body('jupiter', generate_dtm_str())

    def test_round_trip_conversion(self):
        lon0, lat0 = generate_lonlat(self.body)
        lon1, lat1 = self.body.radec2lonlat(*self.body.lonlat2radec(lon0, lat0))
        self.assertAlmostEqual(lon0, lon1, places=2)
        self.assertAlmostEqual(lat0, lat1, places=2)

        lon1, lat1 = self.body.targvec2lonlat(self.body.lonlat2targvec(lon0, lat0))
        self.assertAlmostEqual(lon0, lon1, places=2)
        self.assertAlmostEqual(lat0, lat1, places=2)

    def test_subpoint_visible(self):
        self.assertTrue(
            self.body.test_if_lonlat_visible(
                self.body.subpoint_lon, self.body.subpoint_lat
            )
        )

    def test_missing(self):
        ra = self.body.subpoint_ra + 180  # this should always miss
        dec = self.body.subpoint_dec
        self.assertTrue(all(np.isnan(self.body.radec2lonlat(ra, dec))))

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

    def test_poles_to_plot(self):
        self.assertTrue(len(self.body.get_poles_to_plot()) > 0)


class TestObservationRandomLocation(unittest.TestCase):
    def setUp(self):
        self.observation = mapper.Observation('saturn', generate_dtm_str())

    def test_round_trip_conversion(self):
        lon, lat = generate_lonlat(self.observation)
        lon_rt, lat_rt = self.observation.xy2lonlat(
            *self.observation.lonlat2xy(lon, lat)
        )
        self.assertAlmostEqual(lon, lon_rt, places=2)
        self.assertAlmostEqual(lat, lat_rt, places=2)

    def test_missing(self):
        x = self.observation.get_x0()
        y = self.observation.get_x0()
        x = x + self.observation.get_r0() * 10

        self.assertTrue(all(np.isnan(self.observation.xy2lonlat(x, y))))

def generate_dtm_str() -> str:
    """Create datetime string such that tests are reproducable on same day"""
    dtm = datetime.datetime.now()
    return dtm.strftime('%Y-%m-%d 00:00:00')


def generate_lonlat(body: mapper.Body) -> tuple[float, float]:
    """Choose a random point on the surface that's visible"""
    # Use deterministic seed so tests are reproducable (on the same day)
    seed = (datetime.datetime.now() - datetime.datetime(2000, 1, 1)).days
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
