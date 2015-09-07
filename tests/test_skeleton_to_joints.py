import unittest
from pause import skeleton_to_joints as cv
import math
import numpy as np
import json

class TestSkeletonToJoints(unittest.TestCase):
    def assertAlmostEqual(self, a, b, epsilon=5):
        print "real = {}, desired = {}".format(a,b) 
        self.assertTrue(a >= b-epsilon and a <= b+epsilon)

    def test_colinearity(self):
        a = np.array([3,4,-2])
        b = np.array([7,8,-8])
        c = np.array([13,14,-17])
        u = b-a
        v = c-b
        self.assertTrue(cv.is_colinear(u,v))        

    def write_joint_angles(self, joints_data, name):
        filename = '/home/buschbapti/Documents/pause/data_test/'+name+'_test.json'
        with open(filename, 'w') as outfile:
            json.dump(joints_data, outfile)

    
    def test_skeleton_to_joints(self):
        filename = '/home/buschbapti/Documents/pause/data_test/pose_data_test.json'
        with open(filename) as data_file:
            data = json.load(data_file)
        ######### neutral posture ###########
        print "neutral"
        joints_data = cv.convert_skel_to_joints(data["neutral"])
        self.assertAlmostEqual(joints_data["shoulder_L"][0],0.)
        self.assertAlmostEqual(joints_data["shoulder_L"][1],0.)
        self.assertAlmostEqual(joints_data["shoulder_R"][0],0.)
        self.assertAlmostEqual(joints_data["shoulder_R"][1],0.)
        self.assertAlmostEqual(joints_data["elbow_L"][0],0.)
        self.assertAlmostEqual(joints_data["elbow_R"][0],0.)
        self.assertAlmostEqual(joints_data["wrist_L"][0],0.)
        self.assertAlmostEqual(joints_data["wrist_R"][0],0.)
        self.assertAlmostEqual(joints_data["wrist_L"][1],0.)
        self.assertAlmostEqual(joints_data["wrist_R"][1],0.)
        self.assertAlmostEqual(joints_data["neck"][0],0.)
        self.assertAlmostEqual(joints_data["neck"][0],0.)
        self.write_joint_angles(joints_data, "neutral")
        print "---------"

        ######### shoulder flexion ###########
        print "shoulder_flexion"
        joints_data = cv.convert_skel_to_joints(data["shoulder_flexion"])
        self.assertAlmostEqual(joints_data["shoulder_L"][0],90.)
        self.assertAlmostEqual(joints_data["shoulder_R"][0],90.)
        self.write_joint_angles(joints_data, "shoulder_flexion")
        print "---------"

        ######### shoulder abduction ###########
        print "shoulder_abduction"
        joints_data = cv.convert_skel_to_joints(data["shoulder_abduction"])
        self.assertAlmostEqual(joints_data["shoulder_L"][1],90.)
        self.assertAlmostEqual(joints_data["shoulder_R"][1],90.)
        self.write_joint_angles(joints_data, "shoulder_abduction")
        print "---------"

        ######### elbow flexion ###########
        print "elbow flexion"
        joints_data = cv.convert_skel_to_joints(data["elbow_flexion"])
        self.assertAlmostEqual(joints_data["elbow_L"][0],90.)
        self.assertAlmostEqual(joints_data["elbow_R"][0],90.)
        self.write_joint_angles(joints_data, "elbow_flexion")
        print "---------"

        ######### wrist flexion ###########
        print "wrist flexion"
        joints_data = cv.convert_skel_to_joints(data["wrist_flexion"])
        self.assertAlmostEqual(joints_data["wrist_L"][0],90.)
        self.assertAlmostEqual(joints_data["wrist_R"][0],90.)
        self.write_joint_angles(joints_data, "wrist_flexion")
        print "---------"

        ######### neck flexion ###########
        print "neck flexion"
        joints_data = cv.convert_skel_to_joints(data["neck_flexion"])
        self.assertAlmostEqual(joints_data["neck"][0],45.)
        self.assertAlmostEqual(joints_data["neck"][0],45.)
        self.write_joint_angles(joints_data, "neck_flexion")
        print "---------"

        ######### neck bending ###########
        print "neck bending"
        joints_data = cv.convert_skel_to_joints(data["neck_bending"])
        self.assertAlmostEqual(joints_data["neck"][1],45.)
        self.assertAlmostEqual(joints_data["neck"][1],45.)
        self.write_joint_angles(joints_data, "neck_bending")
        print "---------"

        ######### trunk flexion ###########
        print "trunk flexion"
        joints_data = cv.convert_skel_to_joints(data["trunk_flexion"])
        self.assertAlmostEqual(joints_data["trunk"][0],45.)
        self.assertAlmostEqual(joints_data["trunk"][0],45.)
        self.write_joint_angles(joints_data, "trunk_flexion")
        print "---------"

        ######### trunk bending ###########
        print "trunk bending"
        joints_data = cv.convert_skel_to_joints(data["trunk_bending"])
        self.assertAlmostEqual(joints_data["trunk"][1],45.)
        self.assertAlmostEqual(joints_data["trunk"][1],45.)
        self.write_joint_angles(joints_data, "trunk_bending")
        print "---------"


if __name__ == '__main__':
    unittest.main()