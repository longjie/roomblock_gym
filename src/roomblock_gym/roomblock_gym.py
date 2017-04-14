#!/usr/bin/python

from sensor_msgs.msg import LaserScan, Image, Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from create_node.msg import TurtlebotSensorState
import rospy
from gym.spaces import MultiBinary, Box, Tuple, Discrete
import gym
import numpy as np

class RoomblockApi:
    def __init__(self):
        self.scan = np.zeros([10], dtype=np.float32)
        self.image = np.zeros([480,640,3], dtype=np.uint8)
        self.bumper = np.zeros([2], dtype=np.uint8)
        self.twist = np.zeros(4, dtype=np.uint8)

        rospy.loginfo('Subscribing...')
        self.subs = []
        self.subs.append(rospy.Subscriber('scan', LaserScan, self.scan_cb))

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def scan_cb(self, data):
        # fmin replaces nans with 15
        self.scan = np.fmin(np.array(data.ranges), 8.0)

    def set_action_cmd_vel(self, action):
        msg = Twist()
        # 0 1 2
        # 3 4 5
        # 6 7 8
        if action in [0, 1, 2]:
            msg.linear.x = 1.0
        if action in [6, 7, 8]:
            msg.linear.x = -1.0
        if action in [0, 3, 6]:
            msg.angular.z = 1
        if action in [2, 5, 8]:
            msg.angular.z = -1
        self.pub.publish(msg)

class RoomblockEnv:
    def __init__(self, api, obs_bumper=True, obs_lidar=True, obs_twist=True, act_cmd_vel=True):
        self.api = api
        self.obs_bumper = obs_bumper
        self.obs_lidar = obs_lidar
        self.obs_twist = obs_twist
        self.act_cmd_vel = act_cmd_vel

        # scan
        self.observation_space = Box(0, 8.0, self.api.scan.shape)

        # cmd_vel
        # continuous
        # self.action_space = Box(-1.0, +1.0, 2)
        # 0 1 2
        # 3 4 5
        # 6 7 8

        self.action_space = Discrete(9)
        

    def reset(self):
        return self._get_obs()

    def render(self):
        pass

    def _get_obs(self):
        return self.api.scan

    def is_colide(self):
        min_range = 0.2
        for r in self.api.scan:
            if r < min_range:
                return True
        return False

    def step(self, action):
        # set cmd_vel
        self.api.set_action_cmd_vel(action)
        
        obs = self._get_obs()

        reward = 0
        # 0 1 2
        # 3 4 5
        # 6 7 8
        if action in [0, 1, 2]:
            reward += 1
        if action in [6, 7, 8]:
            reward += -1
        if action in [0, 2, 3, 5]:
            reward += -0.1
        if self.is_colide():
            reward = -10
            
        done = False
        info = {}
        return obs, reward, done, info
