import unittest
import planetmapper
import datetime
import numpy as np
from numpy import array, nan


class TestBody(unittest.TestCase):
    def setUp(self):
        self.obj = planetmapper.Body(
            'jupiter', '2000-01-01', aberration_correction='CN+S'
        )

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
        self.obj = planetmapper.BodyXY(
            'saturn', '2001-02-03', nx=4, ny=6, aberration_correction='CN+S'
        )
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
                    [nan, 342.27696346, 9.7262005, 6.58454767],
                    [134.21396498, 64.70493702, 49.1954644, 37.56896844],
                    [119.36634602, 87.26731979, 69.63427605, 56.17053313],
                    [124.82567431, 100.27272446, 83.68908719, 70.06921137],
                ]
            ),
            'lat-GRAPHIC': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, -73.22033306, -56.40984183, -37.12224446],
                    [-81.69039665, -65.39459639, -48.36403575, -30.95793469],
                    [-60.91680203, -49.37988882, -35.14505374, -19.09454569],
                    [-42.31261069, -32.71489956, -19.76675925, -4.19370823],
                ]
            ),
            'ra': array(
                [
                    [52.27742237, 52.27678016, 52.27613794, 52.27549573],
                    [52.27800062, 52.27735841, 52.27671619, 52.27607398],
                    [52.27857888, 52.27793666, 52.27729445, 52.27665223],
                    [52.27915713, 52.27851491, 52.2778727, 52.27723048],
                    [52.27973538, 52.27909317, 52.27845095, 52.27780874],
                    [52.28031364, 52.27967142, 52.27902921, 52.27838699],
                ]
            ),
            'dec': array(
                [
                    [16.81614223, 16.81669574, 16.81724925, 16.81780277],
                    [16.81675697, 16.81731048, 16.81786399, 16.81841751],
                    [16.81737171, 16.81792522, 16.81847873, 16.81903225],
                    [16.81798645, 16.81853996, 16.81909347, 16.81964699],
                    [16.81860119, 16.8191547, 16.81970821, 16.82026173],
                    [16.81921593, 16.81976944, 16.82032295, 16.82087647],
                ]
            ),
            'phase': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 6.11687086, 6.11748104, 6.11765129],
                    [6.11691735, 6.11774093, 6.11818867, 6.11838153],
                    [6.11743282, 6.11811747, 6.11852402, 6.1187095],
                    [6.11758732, 6.11824169, 6.1186287, 6.11879463],
                ]
            ),
            'incidence': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 66.21756434, 53.86676844, 55.38559878],
                    [62.16688538, 41.18023575, 29.28422307, 29.81298738],
                    [49.28838838, 28.20619076, 10.78677825, 14.34554822],
                    [47.85130737, 27.30922095, 13.02629218, 20.1990873],
                ]
            ),
            'emission': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 69.26606098, 58.45069433, 61.18827377],
                    [62.54814228, 43.00535933, 33.51793325, 35.85611047],
                    [47.61296328, 27.43412415, 13.64044678, 19.9055563],
                    [43.79498452, 22.54619372, 7.01282177, 20.1269339],
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
                    [nan, 31.20683773, 33.30596849, 35.40499082],
                    [27.12128602, 29.2205177, 31.31965063, 33.41870303],
                    [25.13487195, 27.23410936, 29.33326274, 31.43234068],
                    [23.14838956, 25.24764906, 27.34682615, 29.4459278],
                ]
            ),
            'doppler': array(
                [
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, 1.0001041, 1.0001111, 1.00011811],
                    [1.00009047, 1.00009747, 1.00010448, 1.00011148],
                    [1.00008384, 1.00009085, 1.00009785, 1.00010485],
                    [1.00007722, 1.00008422, 1.00009122, 1.00009823],
                ]
            ),
            'ring-radius': array(
                [
                    [228957.78029376, 198474.2597485, 170312.78096379, 145823.68508115],
                    [
                        192175.51144006,
                        160987.64747718,
                        132462.60463105,
                        108715.36894146,
                    ],
                    [155908.54014235, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                ]
            ),
            'ring-Lon-GrApHiC': array(
                [
                    [264.99535888, 270.32191921, 277.49094778, 287.27022681],
                    [262.55257487, 268.6402571, 277.49072238, 290.660046],
                    [258.96434659, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                ]
            ),
            '   ring-distance   ': array(
                [
                    [1.32962718e09, 1.32959630e09, 1.32956543e09, 1.32953455e09],
                    [1.32959455e09, 1.32956368e09, 1.32953280e09, 1.32950193e09],
                    [1.32956193e09, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                    [nan, nan, nan, nan],
                ]
            ),
        }
        for k, img_expected in backplanes_expected.items():
            with self.subTest(k):
                img_output = self.obj.get_backplane_img(k)
                self.assertTrue(
                    np.allclose(img_output, img_expected, equal_nan=True),
                    msg=f'output for {k}:\n' + repr(img_output) + ',\n',
                )


if __name__ == '__main__':
    unittest.main()
