import unittest
import planetmapper
import datetime
import numpy as np
from numpy import array, nan


class TestBody(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.Body('jupiter', '2000-01-01')

    def test_setup(self):
        self.assertEqual(self.obj.target_body_id, 599)
        self.assertEqual(self.obj.target_frame, 'IAU_JUPITER')
        self.assertEqual(self.obj.et, -43135.816087188054)
        self.assertEqual(
            self.obj.dtm,
            datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        )
        self.assertEqual(self.obj.r_eq, 71492.0)
        self.assertEqual(self.obj.r_polar, 66854.0)
        self.assertEqual(self.obj.flattening, 0.0648743915403122)

    def test_subpoint(self):
        self.assertAlmostEqual(self.obj._subpoint_ra, 23.853442709917307, delta=0.0001)
        self.assertAlmostEqual(self.obj._subpoint_dec, 8.586656737972165, delta=0.0001)
        self.assertAlmostEqual(self.obj.subpoint_lat, 3.3106136262222714, delta=0.01)
        self.assertAlmostEqual(self.obj.subpoint_lon, 340.18465394706067, delta=0.01)
        self.assertAlmostEqual(self.obj.subpoint_distance, 690086737.7625527, delta=10)

    def test_target(self):
        self.assertAlmostEqual(self.obj.target_ra, 23.8534426134383, delta=0.001)
        self.assertAlmostEqual(self.obj.target_dec, 8.586656685345513, delta=0.001)
        self.assertAlmostEqual(self.obj.target_distance, 690158217.238101, delta=10)
        self.assertAlmostEqual(self.obj.target_light_time, 2302.1200127659686, delta=0.01)

    def test_limb(self):
        output = self.obj.limb_radec_by_illumination(npts=5)
        expected = (
            array([23.85126327, nan, nan, 23.8521358, 23.84750193, 23.85126327]),
            array([8.59177249, nan, nan, 8.58113043, 8.58615918, 8.59177249]),
            array([nan, 23.85795285, 23.85835397, nan, nan, nan]),
            array([nan, 8.59051167, 8.5837201, nan, nan, nan]),
        )
        self.assertTrue(np.allclose(output, expected, equal_nan=True))

    def test_illumination(self):
        output = self.obj.illumination_angles_from_lonlat(1, 2)
        expected = (0.0033531475561347644, 0.009694273924622665, 0.006341366070869211)
        self.assertTrue(np.allclose(output, expected))
        self.assertTrue(self.obj.test_if_lonlat_illuminated(5, 6))
        self.assertFalse(self.obj.test_if_lonlat_illuminated(5, 180))

    def test_state(self):
        output = self.obj.radial_velocity_from_lonlat(3, 4)
        expected = 21.95717775230934
        self.assertAlmostEqual(output, expected, delta=0.001)


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.BodyXY('saturn', '2001-02-03', nx=4, ny=6)
        self.obj.set_x0(2)
        self.obj.set_y0(4.5)
        self.obj.set_r0(3.14)
        self.obj.set_rotation(42)

    def test_backplanes(self):
        backplanes_expected = {
            'lon': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 339.74193614, 12.30475516, 9.89347204],
                    [151.97329677, 66.11037854, 50.34885993, 38.85337163],
                    [121.77173032, 87.75626183, 69.94433544, 56.5408394],
                    [125.37706528, 100.08373223, 83.41310382, 69.82429704],
                ]
            ),
            'lat': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, -74.39392437, -57.31096124, -37.77785058],
                    [-82.06411498, -66.20069654, -48.56153971, -30.66041632],
                    [-61.99161334, -49.99414757, -35.16740589, -18.5035111],
                    [-43.52257376, -33.37246192, -19.77289822, -3.46012881],
                ]
            ),
            'ra': array(
                [
                    [52.27753369, 52.27689148, 52.27624926, 52.27560705],
                    [52.27808721, 52.27744499, 52.27680278, 52.27616056],
                    [52.27864072, 52.27799851, 52.27735629, 52.27671408],
                    [52.27919424, 52.27855202, 52.27790981, 52.27726759],
                    [52.27974775, 52.27910554, 52.27846332, 52.27782111],
                    [52.28030127, 52.27965905, 52.27901684, 52.27837462],
                ]
            ),
            'dec': array(
                [
                    [16.81609275, 16.816671, 16.81724925, 16.81782751],
                    [16.81670749, 16.81728574, 16.81786399, 16.81844225],
                    [16.81732223, 16.81790048, 16.81847873, 16.81905699],
                    [16.81793697, 16.81851522, 16.81909347, 16.81967173],
                    [16.81855171, 16.81912996, 16.81970821, 16.82028647],
                    [16.81916645, 16.8197447, 16.82032295, 16.82090121],
                ]
            ),
            'phase': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 6.11683874, 6.11751117, 6.11772943],
                    [6.11681473, 6.11771328, 6.11819194, 6.11840451],
                    [6.11737594, 6.11809818, 6.1185229, 6.11871636],
                    [6.11756135, 6.11823654, 6.11863166, 6.11879554],
                ]
            ),
            'incidence': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 66.7814471, 52.66724318, 52.71184099],
                    [64.6825141, 41.90938948, 28.9241162, 28.68934063],
                    [50.77910064, 28.89261741, 10.79298487, 14.28333334],
                    [48.34688713, 27.27624134, 12.78535275, 20.93894973],
                ]
            ),
            'emission': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 69.70177723, 57.15442836, 58.49943536],
                    [65.01799564, 43.62482383, 33.05624859, 34.73936693],
                    [49.15819839, 28.12086884, 13.53384857, 19.73456747],
                    [44.41928075, 22.62298218, 6.78576636, 20.90012588],
                ]
            ),
            'distance': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 1.32940033e09, 1.32938754e09, 1.32938879e09],
                    [1.32939532e09, 1.32937555e09, 1.32936875e09, 1.32937018e09],
                    [1.32938023e09, 1.32936590e09, 1.32936077e09, 1.32936279e09],
                    [1.32937674e09, 1.32936387e09, 1.32935966e09, 1.32936259e09],
                ]
            ),
            'radial_velocity': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 31.00221482, 33.09921512, 35.19610437],
                    [27.00149613, 29.0985986, 31.19559516, 33.29250901],
                    [25.09779491, 27.19489627, 29.29191017, 31.38884661],
                    [23.19402475, 25.29114448, 27.38817939, 29.48513683],
                ]
            ),
            'doppler': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 1.03412257e-04, 1.10407097e-04, 1.17401567e-04],
                    [9.00672963e-05, 9.70624771e-05, 1.04057305e-04, 1.11051856e-04],
                    [8.37172325e-05, 9.07124097e-05, 9.77072951e-05, 1.04701922e-04],
                    [7.73669388e-05, 8.43621773e-05, 9.13571328e-05, 9.83518299e-05],
                ]
            ),
        }
        for k, img_expected in backplanes_expected.items():
            with self.subTest(k):
                img_output = self.obj.get_backplane_img(k)
                self.assertTrue(np.allclose(img_output, img_expected, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
