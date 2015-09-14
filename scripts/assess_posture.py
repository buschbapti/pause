import json
from pause.assessment import Assessment
from pause import skeleton_to_joints as skel
import sys


if __name__ == '__main__':
    name = sys.argv[1]
    filename = '/tmp/'+ name + '.json'
    with open(filename) as data_file:
        skel_data = json.load(data_file)
    # convert skeleton to joints
    joints_data = skel.convert_skel_to_joints(skel_data)

    print joints_data
    
    # assess the posture
    reba = Assessment(25)
    score = reba.assess(joints_data)
    print 'name : {}, score = {}'.format(name,score)