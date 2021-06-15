cmake_minimum_required(VERSION 3.0.2)
project(stomp_kinematics)
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  eigen_conversions
  geometry_msgs
  kdl_parser
  moveit_core
  moveit_msgs
  pluginlib
  roscpp
  tf
  trac_ik_lib
  visualization_msgs
)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

add_definitions("-std=c++11")


###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    geometry_msgs
    kdl_parser
    moveit_core
    moveit_msgs
    pluginlib
    roscpp
    tf
    trac_ik_lib
    visualization_msgs
  DEPENDS
    Boost
    EIGEN3
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(include
    ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/kinematics.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############
## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

## Mark executables and/or libraries for installation
install(TARGETS
    ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

