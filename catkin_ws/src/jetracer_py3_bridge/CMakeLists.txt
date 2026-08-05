cmake_minimum_required(VERSION 3.0.2)
project(jetracer_py3_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  jetracer_smartcity
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs jetracer_smartcity
)

catkin_install_python(PROGRAMS
  scripts/zmq_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
