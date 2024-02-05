cmake_minimum_required(VERSION 3.27)
project(sfm)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(
  Boost 1.65.0
  COMPONENTS program_options filesystem
  REQUIRED)

# Manually create an imported target for JsonCpp to satisfy VTK dependencies
add_library(JsonCpp::JsonCpp SHARED IMPORTED)
set_target_properties(
  JsonCpp::JsonCpp
  PROPERTIES IMPORTED_LOCATION
             "/usr/local/Cellar/jsoncpp/1.9.5/lib/libjsoncpp.dylib"
             INTERFACE_INCLUDE_DIRECTORIES
             "/usr/local/Cellar/jsoncpp/1.9.5/include")

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization features
                                         surface segmentation)

# Include directories
include_directories(${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS} ${Ceres_INCLUDE_DIRS})

# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find necessary libraries
find_library(
  UNISTRING_LIB
  NAMES unistring
  HINTS /usr/local/opt/libunistring/lib)
find_library(
  MBEDCRYPTO_LIB
  NAMES mbedcrypto
  HINTS /usr/local/opt/mbedtls/lib)

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
    ./src/PMVS2Reconstruction.cpp
    ./src/Segmentation.cpp
    ./calibration/CameraCalibration.cpp)
  target_link_libraries(
    sfm
    ${UNISTRING_LIB}
    ${MBEDCRYPTO_LIB}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    Ceres::ceres
    ${PCL_LIBRARIES}
    JsonCpp::JsonCpp # Linking the manually created imported target for JsonCpp
  )
  add_definitions(${PCL_DEFINITIONS})
else()
  message("Did not find libraries")
endif()
