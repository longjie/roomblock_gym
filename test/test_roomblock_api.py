#!/usr/bin/python

import rospy
import rosnode
import rostest
from create_node.msg import TurtlebotSensorState
from roomblock_gym.roomblock_gym import RoomblockApi
import unittest
import numpy as np
import time

class RoomblockApiTest(unittest.TestCase):
    def setUp(self):
        self.api = RoomblockApi()
        time.sleep(1)
        # wait for publisher
        nodes = rosnode.get_node_names()
        self.assertIn('/publisher', nodes, "Publisher does not exit")

    def test_bumper(self):
        np.testing.assert_array_equal(
            self.api.bumper, np.array([1, 1], dtype=np.uint8))

    def test_scan(self):
        np.testing.assert_array_equal(
            self.api.scan, np.array([1.0]*300, dtype=np.uint8))

    def test_odom(self):
        self.assertEqual(self.api.twist[0], 1.0)
        self.assertEqual(self.api.twist[3], 1.0)

if __name__ == '__main__':
    rospy.init_node('test_roomblock_api')
    rostest.rosrun('roomblock_gym', 'test_roomblock_api', RoomblockApiTest)
