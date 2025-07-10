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
from std_msgs.msg import Float32MultiArray
from duckietown_msgs.msg import Twist2DStamped

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
        resized_image = img[: , w//4:, :]

        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return image
        

    def calc_error(self, imageFrame):
        if imageFrame is None:
            self.error = self.prev_error
            return 
        height = imageFrame.shape[0]
        imageFrame = imageFrame[height//2:-height//5, :, :]
        kernel = np.ones((5, 5), "uint8")

        hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        res_white = cv2.bitwise_and(imageFrame, imageFrame, mask=white_mask)
        white_mask = cv2.dilate(white_mask, kernel)
        lane_mask = np.zeros_like(white_mask)  
        lane_mask[white_mask > 0] = 255 

        contours, hierarchy = cv2.findContours(lane_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Search for lane in front
        if len(contours)>0:
            # Sort contours by area in descending order and pick the top two
            max_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            x_values = max_contour[:, 0, 0]  # Extracting x-coordinates

            # Compute the average x-coordinate
            avg_x = np.mean(x_values)
        else:
            avg_x = 0


        self.error = avg_x- lane_mask.shape[1]/2.0
    
    def image_preprocess(self, img):

        new_width = 400
        new_height = 300
        resized_image = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        return resized_image

    def calculate_p_control(self, error):
        return self.Kp * error

    def calculate_pd_control(self, error):
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return self.calculate_p_control(error)
        
        derivative = (error - self.last_error) / dt
        self.last_error = error
        self.last_time = current_time
        
        return self.Kp * error + self.Kd * derivative

    def calculate_pid_control(self, error):
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return self.calculate_pd_control(error)
        
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        
        self.last_error = error
        self.last_time = current_time
        
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def get_control_output(self, error):
        if self.controller_type == "p":
            return self.calculate_p_control(error)
        elif self.controller_type == "pd":
            return self.calculate_pd_control(error)
        else:  # pid
            return self.calculate_pid_control(error)
    
    def stop(self):
        self.publish_twisted(v = 0, omega = 0)
    
    def publish_twisted(self, v, omega):

        message = Twist2DStamped(v=v, omega=omega)
        self.vel_pub.publish(message)

    def publish_command(self, v, omega):
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)
        self.vel_pub.publish(msg)
        
        # Update distance traveled
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.distance_traveled += v * dt
        
        # Stop if target distance reached
        if self.distance_traveled >= self.target_distance:
            self.publish_command(0, 0)
            rospy.signal_shutdown("Target distance reached")

    def run(self):
        self.calc_error(self.undisorted_image)
        
        # Get control output
        omega = self.get_control_output(self.error)
        
        # Publish command
        self.publish_command(self.linear_velocity, omega)



if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    while not rospy.is_shutdown():
        node.run()
    rospy.spin()