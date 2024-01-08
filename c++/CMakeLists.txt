cmake_minimum_required(VERSION 3.27)
project(sfm)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(
  Boost 1.65.0
  COMPONENTS program_options
  REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})

# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find necessary libraries
find_library(
  UNISTRING_LIB
  NAMES unistring
  HINTS /usr/local/opt/libunistring/lib/libunistring.5.dylib)
find_library(
  MBEDCRYPTO_LIB
  NAMES mbedcrypto
  HINTS /usr/local/opt/mbedtls/lib/libmbedcrypto.15.dylib)

if(UNISTRING_LIB AND MBEDCRYPTO_LIB)
  message("Libraries found")
  add_executable(
    sfm
    ./src/SfmReconstruction.cpp
    ./src/FeatureUtil.cpp
    ./src/StereoUtil.cpp
    ./src/main.cpp
    ./src/IOUtil.cpp
    ./src/PlottingUtil.cpp
    ./src/CommonUtil.cpp
    ./src/SfmBundleAdjustment.cpp
    ./calibration/CameraCalibration.cpp)
  target_link_libraries(sfm ${UNISTRING_LIB} ${MBEDCRYPTO_LIB} ${OpenCV_LIBS}
                        ${Boost_LIBRARIES} Ceres::ceres)
else()
  message("Did not find libraries")
endif()
