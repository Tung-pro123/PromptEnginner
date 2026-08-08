cmake_minimum_required(VERSION 3.0.2)
project(fiveq_racer)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/lane_follow_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
