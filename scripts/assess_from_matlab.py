import numpy as np
from pause import matlab_bridge
from pause import assessment
import sys

if __name__ == '__main__':
    # path variables
    dropbox_path = sys.argv[1]
    file_path = 'Inria_TUDa_human_comfort/20150904_partial_kinect_skeleton_fromVREP/func_python_bridge/interface_files/'
    file_name = 'HumanKinematic.json'
    # full path to kinematic file
    fullpath = dropbox_path + file_path + file_name
    # read the kinematic file
    data_angles = matlab_bridge.read_human_kinematic(fullpath)
    # create the REBA object for assessment
    reba = assessment.Assessment(1)
    # assess the given posture
    reba_score = reba.assess(data_angles)
    # return the corresponding score
    sys.stdout.write(str(reba_score))