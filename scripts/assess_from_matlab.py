import numpy as np
from pause import matlab_bridge
from pause import assessment
import sys
import json

if __name__ == '__main__':
    # path variables
    string_data = sys.argv[1]
    matlab_data = json.loads(string_data)
    # read the kinematic file
    data_angles = matlab_bridge.read_human_kinematic(matlab_data)
    # create the REBA object for assessment
    reba = assessment.Assessment(1)
    # assess the given posture
    reba_score = reba.assess(data_angles)
    # return the corresponding score
    sys.stdout.write(str(reba_score))