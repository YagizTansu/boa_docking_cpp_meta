cmake_minimum_required(VERSION 2.8.3)
project(pallet_detection_and_docking)

set(CMAKE_CXX_FLAGS "-std=c++14 -fpermissive ${CMAKE_CXX_FLAGS} -Wfatal-errors\ ")

find_package(catkin REQUIRED COMPONENTS visualization_msgs pcl_ros pcl_conversions roscpp roslaunch rviz mbf_msgs std_msgs std_srvs geometry_msgs sensor_msgs tf laser_geometry advobot_base actionlib_msgs)
find_package(Armadillo REQUIRED)
find_package(Boost 1.54.0 REQUIRED system)
find_package(OpenCV REQUIRED)

# add_message_files(FILES CircleObstacle.msg SegmentObstacle.msg Obstacles.msg)
# generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(
  INCLUDE_DIRS include
  # LIBRARIES obstacle_extractor ${PROJECT_NAME}_nodelets ${PROJECT_NAME}_gui
  CATKIN_DEPENDS roscpp rviz std_msgs std_srvs geometry_msgs sensor_msgs tf laser_geometry advobot_base actionlib_msgs mbf_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


# add_library(pallet_docking src/pallet_docking.cpp)
add_executable(pallet_docking src/pallet_docking.cpp)
target_link_libraries(pallet_docking ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
add_dependencies(pallet_docking ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(pallet_detection src/pallet_detection.cpp)
target_link_libraries(pallet_detection ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(pallet_detection ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_definitions(-DQT_NO_KEYWORDS)
