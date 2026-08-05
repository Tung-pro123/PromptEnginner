cmake_minimum_required(VERSION 3.0.2)
project(jetracer_smartcity)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  IntersectionDecision.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/ai_bridge_node.py
  scripts/perception/detector.py
  scripts/perception/traffic_light_state.py
  scripts/state_machine/route_planner.py
  scripts/state_machine/intersection_state_machine.py
  scripts/control/lane_follower.py
  scripts/control/control_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
