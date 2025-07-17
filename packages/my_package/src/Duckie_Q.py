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
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, LEDPattern
from std_msgs.msg import Float32MultiArray, Header, ColorRGBA, Int32, String
import dt_apriltags as aptag
from geometry_msgs.msg import Point32
from duckietown_msgs.msg import Twist2DStamped
from select import select

import numpy as np
import random
import csv
import time


from collections import defaultdict

import sys, tty, termios

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = True
ENGLISH = False
SAFETY = False
AUSSIE = False
MODEL_PATH = "checkpoint-0715-test.csv"

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.flatnonzero(Q[observation] == np.max(Q[observation]))
        for action in best_action:
            A[action] += (1.0 - epsilon)/len(best_action)
        return A
    return policy_fn


class QAgent:
    def __init__(self, nA = 3, discount_factor=1.0, alpha=0.5, epsilon=0.1, episode = 25, model_path = ""):
        if model_path == "":
            self.Q = self.reset_Q(3,4)
            self.policy = make_epsilon_greedy_policy(self.Q, epsilon, nA)
            self.prev_episodes = 0
            self.discount_factor = discount_factor
        else:
            self.load_model(model_path)
            self.policy = make_epsilon_greedy_policy(self.Q, self.discount_factor, nA)

        self.grid = np.array([
            [0, 1, -1, -1],  
            [0, -1, 1, -1],
            [0, -1, -1, 1]
        ])
        self.alpha = alpha
        self.episodes = episode

        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        self.action = None
        self.eplore = False
      
    # def load_model(self, path):
    #   with open(path, 'rb') as f:
    #     data = pickle.load(f)
    #   self.Q = data["q_table"]
    #   self.prev_episodes = data["episode"]
    #   self.discount_factor = data["epsilon"]
    #   print(self.Q)
    
    # def save_model(self, path, iteration):
    #   checkpoint = {
    #     'episode': iteration,
    #     'epsilon': self.discount_factor,
    #     'q_table': self.Q  # Or model state_dict if using neural networks
    #   }
    #   with open(path, 'wb') as f:
    #     pickle.dump(checkpoint, f)


    def reset_Q(self, n, m):
        Q = defaultdict(list)
        for i in range(n):
            for j in range(m):
                Q[(i, j)] = np.zeros(3) 
        return Q
            

    def reset(self):
        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        return self.start_state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def tagid_to_state(self, tagid, state = (0,0)):
        next_state = list(state)
        if tagid == 78:  # Move forward
            next_state[1] = min(3, state[1] + 1)
        elif tagid == 77:  # Move right
            next_state[1] = min(3, state[1] + 2)
        elif tagid == 91:  # Move left
            next_state[1] = min(3, state[1] + 3)
        else:
            next_state = [0, 0]
        return tuple(next_state)

    def step(self, tagid):
        next_state = self.tagid_to_state(tagid, self.state)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done
    
    def select_action(self):
        action_probs = self.policy(self.state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        if action_probs[action] != np.max(action_probs):
            self.eplore = True
        else:
            self.eplore = False
        return action       
    
    def update(self, action, tagid):
        state = self.state
        next_state, reward, done = self.step(tagid)
        best_next_action = np.argmax(self.Q[next_state])    
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta
        return reward

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

        self.Q = QAgent()
        
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
    
    def blink_once(self):
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))

    def blink_twice(self):
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
        rospy.sleep(0.2)
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
    
    def blink_three(self):
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
        rospy.sleep(0.2)
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
        rospy.sleep(0.2)
        self.publish_leds((1.0, 1.0, 1.0, 1.0))
        rospy.sleep(0.2)
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
    
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
    
    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    
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
        elif key == "\x03" or key == "q":
            return "break"
        else:
            self.publish_twisted(0,0)      
        return None



    def run(self):
        # message = Twist2DStamped(v=v, omega=omega)

        self.wait_for_subscriber()
        action_selected = False
        action = None
        self.publish_leds((0.0, 0.0, 0.0, 0.0))
        while(not rospy.is_shutdown()):
            
            flag = self.keyboard_control()
            if flag == "break":
                rospy.loginfo("Exit Control Mode")
                timing = None
                break
            self.tag_id = self.detect_tag()
            # rospy.loginfo(self.tag_id)
            if self.tag_id == 35:
                if not action_selected:
                    action_selected = True
                    self.Q.reset()
                    action = self.Q.select_action()
                    if action == 0:
                        self.blink_once()
                    elif action == 1:
                        self.blink_twice()
                    elif action == 2:
                        self.blink_three()
                    rospy.loginfo(f"Action {action} is taken")
                    timing = time.time()
            elif not action_selected:
                continue
            elif self.tag_id == 77 or self.tag_id == 78 or self.tag_id == 91:
                rospy.loginfo("Reach Endpoint")
                reward = self.Q.update(action, self.tag_id)
                if reward == 1:
                    self.publish_leds((0.0, 1.0, 0.0, 0.3))
                else:
                    self.publish_leds((1.0, 0.0, 0.0, 0.3))
                self.stop()
                break
        self.stop()
        return timing, action




if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    data_to_save = []
    fieldnames = ['Totle Time', 
                    'Time from Signal to End', 
                    'Action Taken', 
                    'Termination Location', 
                    'Termination Correct', 
                    'Trial Number',
                    'Q Table',
                    'Explore']
    for i in range(5):
        rospy.loginfo(f"This is round {i}")
        rospy.loginfo("When ready, Press any key to start ...")
        start_time = time.time()
        node._getch()
        rospy.loginfo("Press w, a, s, d for moving")
        timing, action = node.run()
        end_time = time.time()
        if timing == None:
            i -= 1
            continue
        # node.Q.save_model(MODEL_PATH, i)
        rospy.sleep(3)
        data = {
            'Totle Time' : end_time - start_time,
            'Time from Signal to End' : end_time - timing,
            'Action Taken': action,
            'Termination Location': node.Q.state,
            'Termination Correct': node.Q.start_state,
            'Trial Number': i,
            'Q Table': node.Q.Q,
            'Explore': node.Q.eplore
        }
        data_to_save.append(data)
    rospy.loginfo(node.Q.Q)
    csv_filename = MODEL_PATH

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_to_save)
    
    rospy.loginfo("CSV is ready, press any key if pulled out ...")
    node._getch()
    rospy.signal_shutdown("Exiting Control Mode")
    rospy.spin()