#!/usr/bin/env python

import rospy
from create_node.msg import TurtlebotSensorState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np

if __name__ == '__main__':
    rospy.init_node('publisher')
    pub_sensor = rospy.Publisher(
        '/mobile_base/sensors/core', TurtlebotSensorState, queue_size=10)
    pub_scan = rospy.Publisher(
        '/scan', LaserScan, queue_size=10)
    pub_odom = rospy.Publisher(
        '/odom', Odometry, queue_size=10)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        msg = TurtlebotSensorState()
        msg.bumps_wheeldrops = 3
        pub_sensor.publish(msg)

        msg = LaserScan()
        msg.ranges = [1.0]*300
        pub_scan.publish(msg)

        msg = Odometry()
        msg.twist.twist.linear.x = 1.0
        msg.twist.twist.angular.z = 1.0
        pub_odom.publish(msg)

        rate.sleep()
