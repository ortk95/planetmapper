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
        self.assertAlmostEqual(
            self.obj.target_light_time, 2302.1200127659686, delta=0.01
        )

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
        expected = (11.007734089805565, 31.824423998504553, 20.81747680519847)
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
            'lon-GRAPHIC': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 148.66967045, 144.90154841, 175.05865345],
                    [131.39602155, 116.81352073, 104.00206245, 85.98106893],
                    [113.06425628, 98.42526043, 84.15771131, 65.61452963],
                    [100.01428929, 85.0525584, 70.84384385, 53.82236465],
                ]
            ),
            'lat-GRAPHIC': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, -39.67422868, -58.67991489, -75.33147161],
                    [-13.28322128, -32.80777874, -49.59862555, -65.85078885],
                    [-1.88175101, -20.19873903, -35.50937962, -48.87407433],
                    [14.00307669, -4.60973672, -19.42351885, -31.56501736],
                ]
            ),
            'ra': array(
                [
                    [52.28262665, 52.28198443, 52.28134222, 52.2807],
                    [52.28204839, 52.28140618, 52.28076396, 52.28012175],
                    [52.28147014, 52.28082793, 52.28018571, 52.2795435],
                    [52.28089189, 52.28024967, 52.27960746, 52.27896524],
                    [52.28031364, 52.27967142, 52.27902921, 52.27838699],
                    [52.27973538, 52.27909317, 52.27845095, 52.27780874],
                ]
            ),
            'dec': array(
                [
                    [16.81835628, 16.81780277, 16.81724925, 16.81669574],
                    [16.81897102, 16.81841751, 16.81786399, 16.81731048],
                    [16.81958576, 16.81903225, 16.81847873, 16.81792522],
                    [16.8202005, 16.81964699, 16.81909347, 16.81853996],
                    [16.82081524, 16.82026173, 16.81970821, 16.8191547],
                    [16.82142998, 16.82087647, 16.82032295, 16.81976944],
                ]
            ),
            'phase': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 6.11695125, 6.11699346, 6.11659438],
                    [6.11755054, 6.11788135, 6.11789993, 6.11766551],
                    [6.11811232, 6.1184026, 6.11842833, 6.11823488],
                    [6.1183588, 6.11867888, 6.11872394, 6.11854983],
                ]
            ),
            'incidence': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 65.93764391, 61.29521508, 70.02728715],
                    [58.23141465, 41.16650958, 36.20903989, 42.61658081],
                    [46.81307239, 26.0968229, 16.18554502, 24.79166448],
                    [47.98026373, 24.21240858, 4.97184251, 16.38844203],
                ]
            ),
            'emission': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 61.67686127, 58.92286388, 69.64214404],
                    [52.24125491, 36.16556365, 33.82927665, 43.08960616],
                    [40.83681332, 20.03257064, 13.80783687, 27.28172487],
                    [43.40078595, 20.06912732, 6.9015291, 22.25225907],
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
            'radial-velocity': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 21.44475969, 23.63983289, 25.83481981],
                    [21.12983849, 23.32494596, 25.5199866, 27.71496932],
                    [23.00996199, 25.20503435, 27.40004706, 29.59500695],
                    [24.89000171, 27.08504956, 29.28003617, 31.47496997],
                ]
            ),
            'doppler': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 1.00007153, 1.00007886, 1.00008618],
                    [1.00007048, 1.00007781, 1.00008513, 1.00009245],
                    [1.00007676, 1.00008408, 1.0000914, 1.00009872],
                    [1.00008303, 1.00009035, 1.00009767, 1.00010499],
                ]
            ),
        }
        for k, img_expected in backplanes_expected.items():
            with self.subTest(k):
                img_output = self.obj.get_backplane_img(k)
                self.assertTrue(
                    np.allclose(img_output, img_expected, equal_nan=True),
                    msg='output:\n' + repr(img_output),
                )


if __name__ == '__main__':
    unittest.main()
