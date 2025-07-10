cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs  # include any message packages you use
)

catkin_package()

catkin_install_python(PROGRAMS
  src/multi_control.py
  learning/DuckietownEnv.py
  learning/Q_learning.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)