import unittest
import mapper
import numpy as np
import datetime


class TestSpiceBody(unittest.TestCase):
    def setUp(self):
        dtm = datetime.datetime.now()
        dtm_str = dtm.strftime('%Y-%m-%d %H:%M:%S')
        self.body = mapper.SpiceBody('jupiter', dtm_str)

    def test_round_trip_conversion(self):
        """
        Test that conversion lon/lat -> ra/dec -> lon/lat gives consistent result.
        """
        body = self.body
        while True:
            # Choose random point on surface that's visible
            lon = np.deg2rad(np.random.rand() * 360)
            lat = np.deg2rad(np.random.rand() * 180 - 90)
            if body.test_if_lonlat_visible(lon, lat):
                break
        lon_rt, lat_rt = body.radec_to_lonlat(*body.lonlat_to_radec(lon, lat))
        self.assertAlmostEqual(lon, lon_rt, places=3)
        self.assertAlmostEqual(lat, lat_rt, places=3)

    def test_missing(self):
        """
        Test ra/dec that should miss the target.
        """
        ra = self.body.subpoint_ra + np.pi  # this should always miss
        dec = self.body.subpoint_dec
        self.assertTrue(all(np.isnan(self.body.radec_to_lonlat(ra, dec))))


if __name__ == '__main__':
    unittest.main()
