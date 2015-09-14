from pause.kinect_bridge import KinectBridge
import time
import os

if __name__ == '__main__':
    kinect = KinectBridge('193.50.110.244', 9999)
    time.sleep(5)
    kinect.record_skeleton('/tmp/test_posture')
    os.system('beep')