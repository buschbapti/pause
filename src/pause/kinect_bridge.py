import numpy as np
import math
from pause import skeleton_to_joints as skel
from pause.assessment import Assessment
import zmq
import numpy
import threading
from collections import namedtuple
import time
import os
import json

torso_joints = ('hip_C', 'spine', 'shoulder_C', 'head')
left_arm_joints = ('shoulder_L', 'elbow_L', 'wrist_L', 'hand_L')
right_arm_joints = ('shoulder_R', 'elbow_R', 'wrist_R', 'hand_R')
left_leg_joints = ('hip_L', 'knee_L', 'ankle_L', 'foot_L')
right_leg_joints = ('hip_R', 'knee_R', 'ankle_R', 'foot_R')
skeleton_joints = torso_joints + left_arm_joints + right_arm_joints + left_leg_joints + right_leg_joints

class Skeleton(object):
    def __init__(self, timestamp, user_id, joints_dict):
        self.timestamp = timestamp
        self.user_id = user_id
        self.joints = joints_dict

class KinectBridge(object):
    def __init__(self, addr, port, window_size=25):
        self._lock = threading.Lock()
        self._skeleton = {}
        # create ZMQ socket for receiving skeleton data
        self.context = zmq.Context()
        self.sub_skel = self.context.socket(zmq.SUB)
        self.sub_skel.connect('tcp://{}:{}'.format(addr, port))
        self.sub_skel.setsockopt(zmq.SUBSCRIBE, '')
        # create thread that receive datas
        t = threading.Thread(target=self.get_skeleton)
        t.daemon = True
        t.start()
        # create reba assessment object
        self.reba = Assessment(window_size)

    def remove_user(self,user_index):
        with self._lock:
            del self._skeleton[user_index]

    def remove_all_users(self):
        with self._lock:
            self._skeleton = {}

    @property
    def tracked_skeleton(self):
        with self._lock:
            return self._skeleton

    @tracked_skeleton.setter
    def tracked_skeleton(self, skeleton):
        with self._lock:
            self._skeleton[skeleton.user_id] = skeleton

    def get_skeleton(self):
        while True:
            md = self.sub_skel.recv_json()
            msg = self.sub_skel.recv()
            skel_array = numpy.fromstring(msg, dtype=float, sep=",")
            skel_array = skel_array.reshape(md['shape'])
            # receive all the joints
            nb_joints = md['shape'][0]
            joints = {}
            for i in range(nb_joints):
                x, y, z, w = skel_array[i][0:4]
                position = [x/w, y/w, z/w]
                xp, yp = skel_array[i][4:6]
                pixel_coord = [xp, yp]
                joints[skeleton_joints[i]] = {}
                joints[skeleton_joints[i]]['position'] = position
                joints[skeleton_joints[i]]['pixel_coordinates'] = pixel_coord
            # create the skeleton object
            self.tracked_skeleton = Skeleton(md['timestamp'], md['user_index'], joints)

    def show_reba_score(self):
        print 'score = {}'.format(self.reba.reba_log['reba'])

    def process_skeleton(self, skeleton):
        # first convert the skeleton data to the joints angles
        joints_data = skel.convert_skel_to_joints(skeleton.joints)

        print joints_data

        # calculate the reba score
        reba_score = self.reba.assess(joints_data)
        self.show_reba_score()

    def record_skeleton(self, filename):
        # get the tracked skeleton
        skeleton = self.tracked_skeleton
        # write the json file
        if skeleton:
            for user,skel in skeleton.iteritems():
                with open(filename+'_'+str(user)+'.json', 'w') as outfile:
                    json.dump(skel.joints, outfile, indent=4, sort_keys=True)


    def run(self):
        #cv2.startWindowThread()
        # while True:
         #   img = numpy.zeros((480, 640, 3))

        time.sleep(5)
        os.system('beep')
        skeleton = self.tracked_skeleton
        if skeleton:
            for user,skel in skeleton.iteritems():
                # assess skeleton with REBA method
                self.process_skeleton(skel)
                # display joints in opencv
                for joint_name in skel.joints:
                    x, y = skel.joints[joint_name]['pixel_coordinates']
                    pt = (int(x),int(y))
      #              cv2.circle(img, pt, 5, (255, 255, 255), thickness=-1)
            # clear skeleton objects
            self.remove_all_users()
        #    cv2.imshow('Skeleton', img)
        #    cv2.waitKey(50)
        # close ZMQ sockets
        self.sub_skel.close()
        self.context.term()
