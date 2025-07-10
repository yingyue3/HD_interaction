#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, LEDPattern
from std_msgs.msg import Float32MultiArray, Header, ColorRGBA, Int32, String
import dt_apriltags as aptag
from geometry_msgs.msg import Point32
from duckietown_msgs.msg import Twist2DStamped
from select import select

from packages.my_package.learning import DukietownEnv

import sys, tty, termios

env = DukietownEnv

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = True
ENGLISH = False
SAFETY = False
AUSSIE = False

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info = f"/{self._vehicle_name}/camera_node/camera_info"


        # Camera parameters
        self.pub_image = rospy.Publisher("/" + self._vehicle_name + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub_info = rospy.Subscriber(self._camera_info, CameraInfo, self.callback_info)
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback_image)
        # self.sub = rospy.Subscriber("/" + self._vehicle_name + "/camera_node/image/compressed",
        #                             CompressedImage,
        #                             self.callback_image,
        #                             queue_size=1,
        #                             buff_size="20MB")
        
        self.jpeg = TurboJPEG()
        self.undisorted_image = None
        self.apriltag_image = None
        self.K = None
        self.D = None
        self._bridge = CvBridge()

        # LED controller
        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

        #color detection
        self.white_lower = np.array([0, 0, 180], np.uint8) 
        self.white_upper = np.array([180, 40, 255], np.uint8)

        # PID Variables
        self.error = 0
        self.prev_error = 0
        self.error_backwards = 0
        self.prev_error_backwards = 0
        self.history = np.zeros((1,10))
        self.history_backwards = np.zeros((1,5))
        self.integral = 0
        self.integral_backwards = 0
        self.calibration = -95
        self.proportional = None
        # if ENGLISH:
        #     self.offset = -180
        # else:
        #     self.offset = 220
        self.offset = 0
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        
        # Controller parameters
        self.controller_type = rospy.get_param("~controller_type", "pid")
        
        # PID gains (random params)
        self.Kp = rospy.get_param("~Kp", 0.05)
        self.Ki = rospy.get_param("~Ki", 0.005)
        self.Kd = rospy.get_param("~Kd", 0.0005)
        
        # Control variables
        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()
        
        # Movement parameters
        self.linear_velocity = 0.3
        self.max_angular_velocity = 8.0
        
        # Distance tracking
        self.start_time = rospy.get_time()
        self.distance_traveled = 0
        self.target_distance = 1.5
        
        # Velocity publisher
        self.vel_pub = rospy.Publisher(f'/{self._vehicle_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)

        self.tag_id = 0
        
        # Lane subscribers
        # self.yellow_sub = rospy.Subscriber('~/yellow_lane', Float32MultiArray, self.yellow_lane_callback, queue_size=1)
        # self.white_sub = rospy.Subscriber('~/white_lane', Float32MultiArray, self.white_lane_callback, queue_size=1)
    
    def callback_info(self, msg):

        # https://stackoverflow.com/questions/55781120/subscribe-ros-image-and-camerainfo-sensor-msgs-format
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        # https://github.com/IntelRealSense/realsense-ros/issues/709ss
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
    
    def callback_image(self, msg):
        if self.K is None:
            return
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        h,w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 1, (w,h))
        dst = cv2.undistort(image, self.K, self.D, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        self.undisorted_image = self.image_preprocess(dst)
        self.apriltag_image = self.apriltag_image_process(self.undisorted_image)
    
    def apriltag_image_process(self, img):
        h, w, _ = img.shape
        resized_image = img[h//4: , w//4:-w//4, :]

        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return image
    
    def detect_tag(self):
        detector = aptag.Detector(families="tag36h11")
        results = detector.detect(self.apriltag_image)

        if results:

            def area(r):
                # Use corners to compute polygon area
                (ptA, ptB, ptC, ptD) = r.corners
                return 0.5 * abs(
                    ptA[0]*ptB[1] + ptB[0]*ptC[1] + ptC[0]*ptD[1] + ptD[0]*ptA[1]
                    - ptB[0]*ptA[1] - ptC[0]*ptB[1] - ptD[0]*ptC[1] - ptA[0]*ptD[1]
                )

            largest_tag = max(results, key=area)

            tag_id = largest_tag.tag_id
            # rospy.loginfo(tag_id)
            return tag_id
        else:
            return None

    
    def image_preprocess(self, img):

        new_width = 400
        new_height = 300
        resized_image = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        return resized_image
    
    def stop(self):
        self.publish_twisted(v = 0, omega = 0)
    
    def publish_twisted(self, v, omega):

        message = Twist2DStamped(v=v, omega=omega)
        self.vel_pub.publish(message)
    
    def publish_leds(self, x):      
        msg = LEDPattern()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        color_msg = ColorRGBA()
        color_msg.r, color_msg.g, color_msg.b, color_msg.a = x


            # Set LED colors
        msg.rgb_vals = [color_msg] * 5
        self.led_pub.publish(msg) 
    
    def get_char_unix(self):
        tty.setraw(sys.stdin.fileno())
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            rlist, _, _ = select([sys.stdin], [], [], 1.0/3)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # rospy.loginfo("run")
        return key
    
    def wait_for_subscriber(self):
        i = 0
        while not rospy.is_shutdown() and self.vel_pub.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.vel_pub.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")
    
    def keyboard_control(self):
        key = self.get_char_unix() 
        if key == "w":
            self.publish_twisted(self.velocity, 0)
        elif key == "a":
            self.publish_twisted(self.velocity, 5)
        elif key == "d":
            self.publish_twisted(self.velocity, -5)
        elif key == "s":
            self.publish_twisted(-self.velocity, 0)
        else:
            if key == "\x03":
                return "break"
            else:
                self.publish_twisted(0,0)      
        return None



    def run(self):
        # message = Twist2DStamped(v=v, omega=omega)

        self.wait_for_subscriber()
        while(not rospy.is_shutdown()):
            
            flag = self.keyboard_control()
            prev_tag = self.tag_id
            self.tag_id = self.detect_tag()
            # rospy.loginfo(self.tag_id)
            if self.tag_id == 35:
                if prev_tag == 35:
                    continue
                self.publish_leds((0.0, 1.0, 0.0, 0.3))
            elif self.tag_id == 77 or self.tag_id == 78 or self.tag_id == 91:
                rospy.loginfo("Reach Endpoint")
                break
            elif prev_tag != self.tag_id:
                rospy.loginfo("Regular")
                self.publish_leds((1.0, 1.0, 1.0, 1.0))
            if flag == "break":
                # rospy.loginfo("Exit Control Mode")
                break
        self.stop()
        rospy.signal_shutdown("Exiting Control Mode")




if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    node.run()
    rospy.spin()