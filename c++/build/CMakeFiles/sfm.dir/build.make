# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.8/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.8/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/stephano./github/sfm-3rd-year-project/c++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/stephano./github/sfm-3rd-year-project/c++/build

# Include any dependencies generated for this target.
include CMakeFiles/sfm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sfm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sfm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sfm.dir/flags.make

CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmReconstruction.cpp
CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o -MF CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o.d -o CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmReconstruction.cpp

CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmReconstruction.cpp > CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.i

CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmReconstruction.cpp -o CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.s

CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/FeatureUtil.cpp
CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o -MF CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/FeatureUtil.cpp

CMakeFiles/sfm.dir/src/FeatureUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/FeatureUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/FeatureUtil.cpp > CMakeFiles/sfm.dir/src/FeatureUtil.cpp.i

CMakeFiles/sfm.dir/src/FeatureUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/FeatureUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/FeatureUtil.cpp -o CMakeFiles/sfm.dir/src/FeatureUtil.cpp.s

CMakeFiles/sfm.dir/src/StereoUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/StereoUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/StereoUtil.cpp
CMakeFiles/sfm.dir/src/StereoUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sfm.dir/src/StereoUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/StereoUtil.cpp.o -MF CMakeFiles/sfm.dir/src/StereoUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/StereoUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/StereoUtil.cpp

CMakeFiles/sfm.dir/src/StereoUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/StereoUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/StereoUtil.cpp > CMakeFiles/sfm.dir/src/StereoUtil.cpp.i

CMakeFiles/sfm.dir/src/StereoUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/StereoUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/StereoUtil.cpp -o CMakeFiles/sfm.dir/src/StereoUtil.cpp.s

CMakeFiles/sfm.dir/src/main.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/main.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/main.cpp
CMakeFiles/sfm.dir/src/main.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sfm.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/main.cpp.o -MF CMakeFiles/sfm.dir/src/main.cpp.o.d -o CMakeFiles/sfm.dir/src/main.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/main.cpp

CMakeFiles/sfm.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/main.cpp > CMakeFiles/sfm.dir/src/main.cpp.i

CMakeFiles/sfm.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/main.cpp -o CMakeFiles/sfm.dir/src/main.cpp.s

CMakeFiles/sfm.dir/src/util.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/util.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/util.cpp
CMakeFiles/sfm.dir/src/util.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/sfm.dir/src/util.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/util.cpp.o -MF CMakeFiles/sfm.dir/src/util.cpp.o.d -o CMakeFiles/sfm.dir/src/util.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/util.cpp

CMakeFiles/sfm.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/util.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/util.cpp > CMakeFiles/sfm.dir/src/util.cpp.i

CMakeFiles/sfm.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/util.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/util.cpp -o CMakeFiles/sfm.dir/src/util.cpp.s

CMakeFiles/sfm.dir/src/drawUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/drawUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/drawUtil.cpp
CMakeFiles/sfm.dir/src/drawUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/sfm.dir/src/drawUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/drawUtil.cpp.o -MF CMakeFiles/sfm.dir/src/drawUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/drawUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/drawUtil.cpp

CMakeFiles/sfm.dir/src/drawUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/drawUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/drawUtil.cpp > CMakeFiles/sfm.dir/src/drawUtil.cpp.i

CMakeFiles/sfm.dir/src/drawUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/drawUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/drawUtil.cpp -o CMakeFiles/sfm.dir/src/drawUtil.cpp.s

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp
CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o -MF CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o.d -o CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp > CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp -o CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s

CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/calibration/CameraCalibration.cpp
CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o -MF CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o.d -o CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/calibration/CameraCalibration.cpp

CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/calibration/CameraCalibration.cpp > CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.i

CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/calibration/CameraCalibration.cpp -o CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.s

# Object files for target sfm
sfm_OBJECTS = \
"CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o" \
"CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/StereoUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/main.cpp.o" \
"CMakeFiles/sfm.dir/src/util.cpp.o" \
"CMakeFiles/sfm.dir/src/drawUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o" \
"CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o"

# External object files for target sfm
sfm_EXTERNAL_OBJECTS =

sfm: CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o
sfm: CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/StereoUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/main.cpp.o
sfm: CMakeFiles/sfm.dir/src/util.cpp.o
sfm: CMakeFiles/sfm.dir/src/drawUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o
sfm: CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o
sfm: CMakeFiles/sfm.dir/build.make
sfm: /usr/local/lib/libunistring.dylib
sfm: /usr/local/lib/libmbedcrypto.dylib
sfm: /usr/local/lib/libopencv_gapi.4.8.1.dylib
sfm: /usr/local/lib/libopencv_stitching.4.8.1.dylib
sfm: /usr/local/lib/libopencv_alphamat.4.8.1.dylib
sfm: /usr/local/lib/libopencv_aruco.4.8.1.dylib
sfm: /usr/local/lib/libopencv_bgsegm.4.8.1.dylib
sfm: /usr/local/lib/libopencv_bioinspired.4.8.1.dylib
sfm: /usr/local/lib/libopencv_ccalib.4.8.1.dylib
sfm: /usr/local/lib/libopencv_dnn_objdetect.4.8.1.dylib
sfm: /usr/local/lib/libopencv_dnn_superres.4.8.1.dylib
sfm: /usr/local/lib/libopencv_dpm.4.8.1.dylib
sfm: /usr/local/lib/libopencv_face.4.8.1.dylib
sfm: /usr/local/lib/libopencv_freetype.4.8.1.dylib
sfm: /usr/local/lib/libopencv_fuzzy.4.8.1.dylib
sfm: /usr/local/lib/libopencv_hfs.4.8.1.dylib
sfm: /usr/local/lib/libopencv_img_hash.4.8.1.dylib
sfm: /usr/local/lib/libopencv_intensity_transform.4.8.1.dylib
sfm: /usr/local/lib/libopencv_line_descriptor.4.8.1.dylib
sfm: /usr/local/lib/libopencv_mcc.4.8.1.dylib
sfm: /usr/local/lib/libopencv_quality.4.8.1.dylib
sfm: /usr/local/lib/libopencv_rapid.4.8.1.dylib
sfm: /usr/local/lib/libopencv_reg.4.8.1.dylib
sfm: /usr/local/lib/libopencv_rgbd.4.8.1.dylib
sfm: /usr/local/lib/libopencv_saliency.4.8.1.dylib
sfm: /usr/local/lib/libopencv_sfm.4.8.1.dylib
sfm: /usr/local/lib/libopencv_stereo.4.8.1.dylib
sfm: /usr/local/lib/libopencv_structured_light.4.8.1.dylib
sfm: /usr/local/lib/libopencv_superres.4.8.1.dylib
sfm: /usr/local/lib/libopencv_surface_matching.4.8.1.dylib
sfm: /usr/local/lib/libopencv_tracking.4.8.1.dylib
sfm: /usr/local/lib/libopencv_videostab.4.8.1.dylib
sfm: /usr/local/lib/libopencv_viz.4.8.1.dylib
sfm: /usr/local/lib/libopencv_wechat_qrcode.4.8.1.dylib
sfm: /usr/local/lib/libopencv_xfeatures2d.4.8.1.dylib
sfm: /usr/local/lib/libopencv_xobjdetect.4.8.1.dylib
sfm: /usr/local/lib/libopencv_xphoto.4.8.1.dylib
sfm: /usr/local/lib/libceres.2.2.0.dylib
sfm: /usr/local/lib/libopencv_shape.4.8.1.dylib
sfm: /usr/local/lib/libopencv_highgui.4.8.1.dylib
sfm: /usr/local/lib/libopencv_datasets.4.8.1.dylib
sfm: /usr/local/lib/libopencv_plot.4.8.1.dylib
sfm: /usr/local/lib/libopencv_text.4.8.1.dylib
sfm: /usr/local/lib/libopencv_ml.4.8.1.dylib
sfm: /usr/local/lib/libopencv_phase_unwrapping.4.8.1.dylib
sfm: /usr/local/lib/libopencv_optflow.4.8.1.dylib
sfm: /usr/local/lib/libopencv_ximgproc.4.8.1.dylib
sfm: /usr/local/lib/libopencv_video.4.8.1.dylib
sfm: /usr/local/lib/libopencv_videoio.4.8.1.dylib
sfm: /usr/local/lib/libopencv_imgcodecs.4.8.1.dylib
sfm: /usr/local/lib/libopencv_objdetect.4.8.1.dylib
sfm: /usr/local/lib/libopencv_calib3d.4.8.1.dylib
sfm: /usr/local/lib/libopencv_dnn.4.8.1.dylib
sfm: /usr/local/lib/libopencv_features2d.4.8.1.dylib
sfm: /usr/local/lib/libopencv_flann.4.8.1.dylib
sfm: /usr/local/lib/libopencv_photo.4.8.1.dylib
sfm: /usr/local/lib/libopencv_imgproc.4.8.1.dylib
sfm: /usr/local/lib/libopencv_core.4.8.1.dylib
sfm: /usr/local/lib/libglog.0.6.0.dylib
sfm: /usr/local/lib/libgflags.2.2.2.dylib
sfm: CMakeFiles/sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable sfm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sfm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sfm.dir/build: sfm
.PHONY : CMakeFiles/sfm.dir/build

CMakeFiles/sfm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sfm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sfm.dir/clean

CMakeFiles/sfm.dir/depend:
	cd /Users/stephano./github/sfm-3rd-year-project/c++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/stephano./github/sfm-3rd-year-project/c++ /Users/stephano./github/sfm-3rd-year-project/c++ /Users/stephano./github/sfm-3rd-year-project/c++/build /Users/stephano./github/sfm-3rd-year-project/c++/build /Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles/sfm.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/sfm.dir/depend

