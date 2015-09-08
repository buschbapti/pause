import numpy as np
from pause import skeleton_to_joints

def read_human_kinematic(matlab_data):
    # dict of correspondance notation
    matlab_dict = {}
    matlab_dict['shoulderCenter'] = 'shoulder_C'
    matlab_dict['shoulderRight'] = 'shoulder_R'
    matlab_dict['shoulderLeft'] = 'shoulder_L'
    matlab_dict['elbowRight'] = 'elbow_R'
    matlab_dict['wristRight'] = 'wrist_R'
    matlab_dict['handRight'] = 'hand_R'
    # create skeleton data
    skel_data = {}
    for joint in matlab_data:
        joint_name = matlab_dict[joint['name']]
        joint_pos = joint['xyz']
        skel_data[joint_name] = joint_pos
    # appened fake values for the res of the body
    value = [0,0,0]
    skel_data['head'] = value
    skel_data['elbow_L'] = value
    skel_data['wrist_L'] = value
    skel_data['hand_L'] = value
    skel_data['spine'] = value
    skel_data['hip_C'] = value
    skel_data['hip_L'] = value
    skel_data['hip_R'] = value
    # get the angles from the skeleton
    reba_data = skeleton_to_joints.convert_skel_to_joints(skel_data)
    # put fake values for the fake angles
    reba_data['shoulder_L'] = value
    reba_data['elbow_L'] = value
    reba_data['wrist_L'] = value
    reba_data['neck'] = value
    reba_data['trunk'] = value
    return reba_data
