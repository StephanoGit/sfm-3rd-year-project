cmake_minimum_required(VERSION 3.10)
project(sfm)
find_package(OpenCV REQUIRED)
add_executable(sfm main.cpp)
target_link_libraries(sfm ${OpenCV_LIBS})