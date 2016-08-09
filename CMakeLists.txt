cmake_minimum_required(VERSION 2.8.3)
project(realsense_simple)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_executable(realsense_simple src/realsense_simple.cpp)
target_link_libraries(realsense_simple ${catkin_LIBRARIES})