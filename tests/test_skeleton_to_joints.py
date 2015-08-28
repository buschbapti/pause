import unittest
from pause import skeleton_to_joints as cv
import math
import numpy as np
import json

class TestSkeletonToJoints(unittest.TestCase):
    def assertAlmostEqual(self,a,b,epsilon=5):
        print "a = {}, b = {}".format(a,b) 
        self.assertTrue(a >= b-epsilon and a <= b+epsilon)

    def test_colinearity(self):
        a = np.array([3,4,-2])
        b = np.array([7,8,-8])
        c = np.array([13,14,-17])
        u = b-a
        v = c-b
        self.assertTrue(cv.is_colinear(u,v))        

    def test_skeleton_to_joints(self):
        filename = '/home/buschbapti/Documents/pause/data_test/pose_data_test.json'
        with open(filename) as data_file:
            data = json.load(data_file)
        ######### neutral posture ###########
        skel_data = cv.convert_skel_to_joints(data["neutral"])
        self.assertAlmostEqual(skel_data["shoulder_L"][0],0.0)
        self.assertAlmostEqual(skel_data["shoulder_L"][1],0.0)
        self.assertAlmostEqual(skel_data["elbow_L"][0],0.0)

        ######### cross posture ###########
        skel_data = cv.convert_skel_to_joints(data["cross"])
        self.assertAlmostEqual(skel_data["shoulder_L"][0],0.0)
        self.assertAlmostEqual(skel_data["shoulder_L"][1],90.0)
        self.assertAlmostEqual(skel_data["elbow_L"][0],0.0)        

if __name__ == '__main__':
    unittest.main()