from pause.kinect_bridge import KinectBridge

if __name__ == '__main__':
    kinect = KinectBridge('193.50.110.244', 9999)
    kinect.run()