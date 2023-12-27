cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_visualizer_viewports)

find_package(PCL 1.2 REQUIRED)
# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_visualizer_demo pcl_visualizer_demo.cpp)
target_link_libraries(pcl_visualizer_demo ${PCL_LIBRARIES})
