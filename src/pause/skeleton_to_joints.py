import numpy as np
import math
import json

EPSILON = 10e-5

def points_to_vect(pointA, pointB):
    return (np.array(pointA)-np.array(pointB))
    
def rad_to_deg(angle):
    return angle*180/math.pi

def triad_method(R, r):
    S = R[:,0]
    s = r[:,0]
    M = np.cross(R[:,0],R[:,1])
    m = np.cross(r[:,0],r[:,1])
    A1 = np.concatenate((S.reshape(3,1),M.reshape(3,1),np.cross(S,M).reshape(3,1)),axis=1)
    A2 = np.concatenate((s.reshape(3,1),m.reshape(3,1),np.cross(s,m).reshape(3,1)),axis=1)
    A = np.dot(A1,A2.T)
    return A

def transformation_matrix(R, r, P):
    A = triad_method(R,r)
    # create the transformation matrix
    P_K = -np.dot(A,P)
    P_T_K = np.concatenate((A,P_K),axis=1)
    P_T_K = np.concatenate((P_T_K,[[0,0,0,1]]),axis=0)
    return P_T_K

def vect_to_euler(vectA, vectB):
    # find axis of rotation
    v = np.cross(vectA,vectB)
    if np.linalg.norm(v) == 0.0:
        return [0.0,0.0,0.0]
    v = v/np.linalg.norm(v)
    # calculate rotation angle
    cos_angle = np.dot(vectA,vectB)/(np.linalg.norm(vectA)*np.linalg.norm(vectB))
    angle = math.acos(cos_angle)
    # define variables
    x = v[0]
    y = v[1]
    z = v[2]
    c = math.cos(angle)
    t = 1-math.cos(angle)
    s = math.sin(angle)
    # check singularities
    if (x*y*t + z*s) > 0.998: # north pole singularity detected
        heading = 2*math.atan2(x*math.sin(angle/2),math.cos(angle/2))
        attitude = math.pi/2
        bank = 0.0
        return [rad_to_deg(heading), rad_to_deg(attitude), rad_to_deg(bank)]
    if (x*y*t + z*s) < -0.998: # south pole singularity detected
        heading = -2*math.atan2(x*math.sin(angle/2),math.cos(angle/2))
        attitude = -math.pi/2
        bank = 0.0
        return [rad_to_deg(heading), rad_to_deg(attitude), rad_to_deg(bank)]
    # convert rotation to euler angles
    heading = math.atan2(y*s-x*z*t,1-(y*y+z*z)*t)
    attitude = math.asin(x*y*t+z*s)
    bank = math.atan2(x*s-y*z*t,1-(x*x+z*z)*t)
    return [rad_to_deg(heading), rad_to_deg(bank), rad_to_deg(attitude)]

def is_colinear(u, v, offset=0.):
    sum_colin = 0
    for i in range(len(u)):
        for j in range(i+1,len(u)):
            sum_colin += u[i]*v[j]-u[j]*v[i]
    return (abs(sum_colin) <= EPSILON+offset)

def is_orthogonal(u, v, offset=0.):
    return (abs(np.inner(u.reshape(1,3),v.reshape(1,3))) <= EPSILON+offset)

def normalize(vect):
    if np.linalg.norm(vect) != 0.0:
        return vect/np.linalg.norm(vect)
    else:
        return vect

def get_shoulder_transformation(P_SL, P_SR, P_SC, shoulder="left"):
    # get the axis of the shoulder frame
    if shoulder == "left":
        x = P_SL-P_SR
        P_S = P_SL
    else:
        x = P_SR-P_SL
        P_S = P_SR
    x = normalize(x)
    mid_SC = (P_SL+P_SR)/2
    z = P_SC-mid_SC
    z = normalize(z)
    # calculate the between kinect frame and the shoulder frame
    r = np.concatenate((x,z),axis=1)
    R = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    S_T_K = transformation_matrix(R,r,P_S)
    return S_T_K

def get_elbow_transformation(P_S, P_E, P_W):
    x = P_E-P_S
    v_upper = -x
    v_lower = P_W-P_E
    x = normalize(x)
    # check colinearity between upper and lower arm
    if is_colinear(v_upper,v_lower):
        # find one vector that satisfy the equation
        if x[2] != 0:
            z = np.array([[1.],[1.],[-(x[0]+x[1])/x[2]]])
        elif x[1] != 0:
            z = np.array([[1.],[-x[0]/x[1]],[1.]])
        else:
            z = np.array([[0.],[1.],[1.]])
    else:
        z = np.cross(v_lower.reshape(1,3),v_upper.reshape(1,3))
        z = z.reshape(3,1)
    z = normalize(z)
    # calculate the between kinect frame and the elbow frame
    r = np.concatenate((x,z),axis=1)
    R = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    E_T_K = transformation_matrix(R,r,P_E)
    return E_T_K

def get_wrist_transformation(P_E, P_W, P_H):
    x = P_W-P_E
    v_upper = -x
    v_lower = P_H-P_W
    x = normalize(x)
    # check colinearity between upper and lower arm
    if is_colinear(v_upper,v_lower):
        # find one vector that satisfy the equation
        if x[2] != 0:
            z = np.array([[1.],[1.],[-(x[0]+x[1])/x[2]]])
        elif x[1] != 0:
            z = np.array([[1.],[-x[0]/x[1]],[1.]])
        else:
            z = np.array([[0.],[1.],[1.]])
    else:
        z = np.cross(v_lower.reshape(1,3),v_upper.reshape(1,3))
        z = z.reshape(3,1)
    z = normalize(z)
    # calculate the between kinect frame and the elbow frame
    r = np.concatenate((x,z),axis=1)
    R = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    W_T_K = transformation_matrix(R,r,P_W)
    return W_T_K

def get_neck_transformation(P_SL, P_SR, P_SC):
    # get the axis of the neck frame
    x = P_SL-P_SR
    x = normalize(x)
    mid_SC = (P_SL+P_SR)/2
    z = P_SC-mid_SC
    z = normalize(z)
    # calculate the between kinect frame and the neck frame
    r = np.concatenate((x,z),axis=1)
    R = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    N_T_K = transformation_matrix(R,r,P_SC)
    return N_T_K

def get_trunk_transformation(P_S, P_HC, P_HL, P_HR):
    # get the axis of the trunk frame
    x = P_HL-P_HR
    x = normalize(x)
    z = P_S-P_HC
    z = normalize(z)
    # calculate the between kinect frame and the trunk frame
    r = np.concatenate((x,z),axis=1)
    R = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    T_T_K = transformation_matrix(R,r,P_HC)
    return T_T_K

def calculate_shoulder_angles(S_T_K, K_E, shoulder="left"):
    # first calculate the coordinate of the elbow in the shoulder frame
    P_E = np.concatenate((K_E,[[1]]),axis=0)
    S_E = np.dot(S_T_K,P_E)
    epsilon = 0.01

    # calculate the alpha angle (atan2(y/-z))
    if abs(S_E[0]) <= epsilon and abs(S_E[2]) <= epsilon:
        alpha = math.pi/2
    else:
        alpha = math.atan2(S_E[0],-S_E[2])
    # calculate the rotation beta (atan2(y/x))
    if shoulder == "left":
        beta = math.atan2(-S_E[1],S_E[0])
    else:
        beta = math.atan2(S_E[1],S_E[0])
    # calculate the flexion and the abduction
    flex = math.sin(beta)*alpha
    abd = (1-math.sin(beta))*alpha
    # return the angles converted in degrees
    return [rad_to_deg(flex),rad_to_deg(abd),0.]
    
def calculate_elbow_angles(E_T_K, K_W):
    # first calculate the coordinate of the wrist in the elbow frame
    P_W = np.concatenate((K_W,[[1]]),axis=0)
    E_W = np.dot(E_T_K,P_W)
    # calculate the flexion (atan2(y/x))
    flex = math.atan2(E_W[1],E_W[0])
    return [rad_to_deg(flex),0.,0.]

def calculate_wrist_angles(W_T_K, K_H):
    # first calculate the coordinate of the hand in the wrist frame
    P_H = np.concatenate((K_H,[[1]]),axis=0)
    W_H = np.dot(W_T_K,P_H)
    # calculate the flexion (atan2(y/x))
    flex = math.atan2(W_H[1],W_H[0])
    # calculate the deviation (atan2(z/x))
    dev = math.atan2(W_H[2],W_H[0])
    return [rad_to_deg(flex),0.,0.]

def calculate_neck_angles(N_T_K, K_H):
    # first calculate the coordinate of the head in the neck frame
    P_H = np.concatenate((K_H,[[1]]),axis=0)
    N_H = np.dot(N_T_K,P_H)
    # calculate the flexion (atan2(y/z))
    flex = math.atan2(N_H[1],N_H[2])
    # calculate the deviation (atan2(x/z))
    bend = math.atan2(N_H[0],N_H[2])
    return [rad_to_deg(flex),rad_to_deg(bend),0.]

def calculate_trunk_angles(T_T_K, K_SC):
    # first calculate the coordinate of the head in the trunk frame
    P_SC = np.concatenate((K_SC,[[1]]),axis=0)
    T_SC = np.dot(T_T_K,P_SC)
    # calculate the flexion (atan2(y/z))
    flex = math.atan2(T_SC[1],T_SC[2])
    # calculate the deviation (atan2(x/z))
    bend = math.atan2(T_SC[0],T_SC[2])
    return [rad_to_deg(flex),rad_to_deg(bend),0.]

def save_log(skel_data, joints_data):
    log_data = {}
    log_data['skeleton'] = skel_data
    log_data['angles'] = joints_data
    with open('/tmp/pause_log.json', 'w') as outfile:
        json.dump(log_data, outfile, indent=4, sort_keys=True)

def convert_skel_to_joints(skel_data):
    joints_data = {}
    P_SC = np.array(skel_data['shoulder_C']['position']).reshape((3,1))
    P_SL = np.array(skel_data['shoulder_L']['position']).reshape((3,1))
    P_SR = np.array(skel_data['shoulder_R']['position']).reshape((3,1))
    P_EL = np.array(skel_data['elbow_L']['position']).reshape((3,1))
    P_ER = np.array(skel_data['elbow_R']['position']).reshape((3,1))
    P_WL = np.array(skel_data['wrist_L']['position']).reshape((3,1))
    P_WR = np.array(skel_data['wrist_R']['position']).reshape((3,1))
    P_HL = np.array(skel_data['hand_L']['position']).reshape((3,1))
    P_HR = np.array(skel_data['hand_R']['position']).reshape((3,1))
    P_H = np.array(skel_data['head']['position']).reshape((3,1))
    P_S = np.array(skel_data['spine']['position']).reshape((3,1))
    P_HC = np.array(skel_data['hip_C']['position']).reshape((3,1))
    P_HPL = np.array(skel_data['hip_L']['position']).reshape((3,1))
    P_HPR = np.array(skel_data['hip_R']['position']).reshape((3,1))

    ############ shoulder_L angles ##################
    S_T_K = get_shoulder_transformation(P_SL,P_SR,P_SC,"left")
    joints_data['shoulder_L'] = calculate_shoulder_angles(S_T_K,P_EL,"left")
    ############ shoulder_R angles ##################
    S_T_K = get_shoulder_transformation(P_SL,P_SR,P_SC,"right")
    joints_data['shoulder_R'] = calculate_shoulder_angles(S_T_K,P_ER,"right")
    ############ elbow_L angles ###################
    E_T_K = get_elbow_transformation(P_SL,P_EL,P_WL)
    joints_data['elbow_L'] = calculate_elbow_angles(E_T_K,P_WL)
    ############ elbow_R angles ###################
    E_T_K = get_elbow_transformation(P_SR,P_ER,P_WR)
    joints_data['elbow_R'] = calculate_elbow_angles(E_T_K,P_WR)
    ############ wrist_L angles ###################
    W_T_K = get_wrist_transformation(P_EL,P_WL,P_HL)
    joints_data['wrist_L'] = calculate_wrist_angles(W_T_K,P_HL)
    ############ wrist_R angles ###################
    W_T_K = get_wrist_transformation(P_ER,P_WR,P_HR)
    joints_data['wrist_R'] = calculate_wrist_angles(W_T_K,P_HR)
    ############ neck angles ###################
    N_T_K = get_neck_transformation(P_SL,P_SR,P_SC)
    joints_data['neck'] = calculate_neck_angles(N_T_K,P_H)
    ############ trunk angles ###################
    T_T_K = get_trunk_transformation(P_S,P_HC,P_HPL,P_HPR)
    joints_data['trunk'] = calculate_trunk_angles(T_T_K,P_SC)

    ########### legs angles ####################
    joints_data['leg_L'] = [0,1]
    joints_data['leg_R'] = [0,1]

    ########## load and dynamic ################
    joints_data['load'] = 0
    joints_data['static'] = 0
    joints_data['high_dynamic'] = 0 

    # save the log file
    save_log(skel_data,joints_data)
    return joints_data