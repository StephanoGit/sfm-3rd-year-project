cmake_minimum_required(VERSION 3.27)
project(calibrate)

# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find the OpenCV package
find_package(OpenCV REQUIRED)
find_package(pugixml REQUIRED)

# Include the OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable target
add_executable(calibrate CameraCalibration.cpp ../pipeline/src/IOUtil.cpp) # Replace main.cpp with your
# source files

# Link your application with the OpenCV libraries
target_link_libraries(calibrate pugixml ${OpenCV_LIBS})
