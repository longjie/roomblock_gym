#!/usr/bin/python

import rospy
import rosnode
import rostest
from create_node.msg import TurtlebotSensorState
from roomblock_gym.roomblock_gym import RoomblockApi, RoomblockEnv
import unittest
import numpy as np
import time

class RoomblockEnvTest(unittest.TestCase):
    def setUp(self):
        api = RoomblockApi()
        self.env = RoomblockEnv(api)

    def test_one_step(self):
        obs = self.env.reset()
        self.env.render()
        action = self.env.action_space.sample()
        obs, r, done, info = self.env.step(action)

if __name__ == '__main__':
    rospy.init_node('test_roomblock_env')
    rostest.rosrun('roomblock_gym', 'test_roomblock_env', RoomblockEnvTest)

