import numpy as np
import math

def points_to_vect(pointA, pointB):
    return (np.array(pointA)-np.array(pointB))
    
def rad_to_deg(angle):
    return angle*180/math.pi

def triad_method(R,r):
    S = R[:,0]
    s = r[:,0]
    M = np.cross(R[:,0],R[:,1])
    m = np.cross(r[:,0],r[:,1])
    A1 = np.concatenate((S,M,np.cross(S,M)),axis=1).reshape((3,3))
    A2 = np.concatenate((s,m,np.cross(s,m)),axis=1).reshape((3,3))
    A = np.dot(A1,A2.T)
    return A

def transformation_matrix(R,r,P):
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

def get_shoulder_transformation(P_SL, P_SR, P_SC, shoulder="left"):
    # get the axis of the shoulder frame
    if shoulder == "left":
        x = P_SL-P_SR
        P_S = P_SL
    else:
        x = P_SR-P_SL
        P_S = P_SR
    x = x/np.linalg.norm(x)
    mid_SC = (P_SL+P_SR)/2
    z = P_SC-mid_SC
    z = z/np.linalg.norm(z)
    # calculate the between kinect frame and the shoulder frame
    R = np.concatenate((x,z),axis=1)
    r = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    S_T_K = transformation_matrix(R,r,P_S)
    return S_T_K

def is_colinear(u,v):
    sum_colin = 0
    for i in range(len(u)):
        for j in range(i+1,len(u)):
            sum_colin += u[i]*v[j]-u[j]*v[i]
    return sum_colin == 0

def get_elbow_transformation(P_S,P_E,P_W):
    x = P_E-P_S
    v_upper = -x
    v_lower = P_W-P_E
    x = x/np.linalg.norm(x)
    # check colinearity between upper and lower arm
    if is_colinear(v_upper,v_lower):
        print "yes they are"
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

    print z
    z = z/np.linalg.norm(z)
    # calculate the between kinect frame and the elbow frame
    R = np.concatenate((x,z),axis=1)
    r = np.concatenate(([[1],[0],[0]],[[0],[0],[1]]),axis=1)
    E_T_K = transformation_matrix(R,r,P_E)

    print x
    print "---"
    print z
    print "---"
    print E_T_K

    return E_T_K

def calculate_shoulder_angles(S_T_K,K_E):
    # first calculate the coordinate of the elbow in the shoulder frame
    P_E = np.concatenate((K_E,[[1]]),axis=0)
    S_E = np.dot(S_T_K,P_E)
    # calculate the flexion (atan2(y,-z))
    if S_E[1] == 0 and S_E[2] == 0:
        flex = 0.0
    else:
        flex = math.atan2(S_E[1],-S_E[2])
    # calculate the abduction (atan2(x,-z))
    if S_E[0] == 0 and S_E[2] == 0:
        abd = 0.0
    else:
        abd = math.atan2(S_E[0],-S_E[2])
    return [rad_to_deg(flex),rad_to_deg(abd),0.]
    
def calculate_elbow_angles(E_T_K,K_W):
    # first calculate the coordinate of the wrist in the elbow frame
    P_W = np.concatenate((K_W,[[1]]),axis=0)
    E_W = np.dot(E_T_K,P_W)

    print E_W


    # calculate the flexion (atan2(x,y))
    flex = math.atan2(E_W[1],E_W[0])
    return [rad_to_deg(flex),0.,0.]

def convert_skel_to_joints(skel_data):
    joints_data = {}
    P_SC = np.array(skel_data["shoulder_C"]).reshape((3,1))
    P_SL = np.array(skel_data["shoulder_L"]).reshape((3,1))
    P_SR = np.array(skel_data["shoulder_R"]).reshape((3,1))
    P_EL = np.array(skel_data["elbow_L"]).reshape((3,1))
    P_ER = np.array(skel_data["elbow_R"]).reshape((3,1))
    P_WL = np.array(skel_data["wrist_L"]).reshape((3,1))

    ############ shoulder_L angles ##################
    S_T_K = get_shoulder_transformation(P_SL,P_SR,P_SC,"left")
    angles = calculate_shoulder_angles(S_T_K,P_EL)
    joints_data["shoulder_L"] = [angles[0],angles[1],angles[2]]
    ############ elbow_L angles ###################
    E_T_K = get_elbow_transformation(P_SL,P_EL,P_WL)
    angles = calculate_elbow_angles(E_T_K,P_WL)
    joints_data["elbow_L"] = [angles[0],angles[1],angles[2]]

    # shoulder_R angles
  #  v1 = points_to_vect(skel_data["shoulder_L"],skel_data["shoulder_R"])
   # v2 = points_to_vect(skel_data["shoulder_R"],skel_data["elbow_R"])
   # angles = vect_to_euler(v1,v2)
   # joints_data["shoulder_R"] = [angles[0],angles[2],90+angles[1]]
    # # elbow_L angles
    # v1 = points_to_vect(skel_data["shoulder_L"],skel_data["elbow_L"])
    # v2 = points_to_vect(skel_data["elbow_L"],skel_data["wrist_L"])
    # joints_data["elbow_L"] = vect_to_euler(v1,v2)
    # # elbow_R angles
    # v1 = points_to_vect(skel_data["shoulder_R"],skel_data["elbow_R"])
    # v2 = points_to_vect(skel_data["elbow_R"],skel_data["wrist_R"])
    # joints_data["elbow_R"] = vect_to_euler(v1,v2)
    # # wrist_L angles
    # v1 = points_to_vect(skel_data["elbow_L"],skel_data["wrist_L"])
    # v2 = points_to_vect(skel_data["wrist_L"],skel_data["hand_L"])
    # joints_data["wrist_L"] = vect_to_euler(v1,v2)
    # # wrist_R angles
    # v1 = points_to_vect(skel_data["elbow_R"],skel_data["wrist_R"])
    # v2 = points_to_vect(skel_data["wrist_R"],skel_data["hand_R"])
    # joints_data["wrist_R"] = vect_to_euler(v1,v2)

    print "---------"
    print "shoulder_L : {}".format(joints_data["shoulder_L"])
    # print joints_data["shoulder_R"]
    print "elbow_L : {}".format(joints_data["elbow_L"])
    # print joints_data["wrist_L"]

    return joints_data