cmake_minimum_required(VERSION 3.27)
project(sfm)
find_package(OpenCV REQUIRED)


find_library(UNISTRING_LIB NAMES unistring HINTS /usr/local/opt/libunistring/lib/libunistring.5.dylib)
find_library(MBEDCRYPTO_LIB NAMES mbedcrypto HINTS /usr/local/opt/mbedtls/lib/libmbedcrypto.15.dylib)

if (UNISTRING_LIB AND MBEDCRYPTO_LIB)
    message("Libraries found")
    add_executable(sfm main.cpp
                   ImageView.cpp
                   ImagePair.cpp
                   util.cpp
                   drawUtil.cpp
                   SfmReconstruction.cpp
                   ./camera_calibration/CameraCalibration.cpp)
    target_link_libraries(sfm ${UNISTRING_LIB} ${MBEDCRYPTO_LIB} ${OpenCV_LIBS})
else()
    message("Did not find libraries")
endif()