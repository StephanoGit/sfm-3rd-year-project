cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_visualizer)

find_package(PCL 1.2 REQUIRED)
# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Boost
find_package(
  Boost 1.65.0
  COMPONENTS program_options
  REQUIRED)
# Include Boost headers
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_visualizer pcl_visualizer.cpp Segmentation.cpp)
target_link_libraries(pcl_visualizer ${PCL_LIBRARIES} ${Boost_LIBRARIES})
