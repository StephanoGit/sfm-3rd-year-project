# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.1/bin/cmake -E rm -f

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

CMakeFiles/sfm.dir/src/IOUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/IOUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/IOUtil.cpp
CMakeFiles/sfm.dir/src/IOUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/sfm.dir/src/IOUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/IOUtil.cpp.o -MF CMakeFiles/sfm.dir/src/IOUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/IOUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/IOUtil.cpp

CMakeFiles/sfm.dir/src/IOUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/IOUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/IOUtil.cpp > CMakeFiles/sfm.dir/src/IOUtil.cpp.i

CMakeFiles/sfm.dir/src/IOUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/IOUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/IOUtil.cpp -o CMakeFiles/sfm.dir/src/IOUtil.cpp.s

CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/PlottingUtil.cpp
CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o -MF CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/PlottingUtil.cpp

CMakeFiles/sfm.dir/src/PlottingUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/PlottingUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/PlottingUtil.cpp > CMakeFiles/sfm.dir/src/PlottingUtil.cpp.i

CMakeFiles/sfm.dir/src/PlottingUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/PlottingUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/PlottingUtil.cpp -o CMakeFiles/sfm.dir/src/PlottingUtil.cpp.s

CMakeFiles/sfm.dir/src/CommonUtil.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/CommonUtil.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/CommonUtil.cpp
CMakeFiles/sfm.dir/src/CommonUtil.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/sfm.dir/src/CommonUtil.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/CommonUtil.cpp.o -MF CMakeFiles/sfm.dir/src/CommonUtil.cpp.o.d -o CMakeFiles/sfm.dir/src/CommonUtil.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/CommonUtil.cpp

CMakeFiles/sfm.dir/src/CommonUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/CommonUtil.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/CommonUtil.cpp > CMakeFiles/sfm.dir/src/CommonUtil.cpp.i

CMakeFiles/sfm.dir/src/CommonUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/CommonUtil.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/CommonUtil.cpp -o CMakeFiles/sfm.dir/src/CommonUtil.cpp.s

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp
CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o -MF CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o.d -o CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp > CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.i

CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/SfmBundleAdjustment.cpp -o CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.s

CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/PMVS2Reconstruction.cpp
CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o -MF CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o.d -o CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/PMVS2Reconstruction.cpp

CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/PMVS2Reconstruction.cpp > CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.i

CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/PMVS2Reconstruction.cpp -o CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.s

CMakeFiles/sfm.dir/src/Segmentation.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/Segmentation.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/src/Segmentation.cpp
CMakeFiles/sfm.dir/src/Segmentation.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/sfm.dir/src/Segmentation.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm.dir/src/Segmentation.cpp.o -MF CMakeFiles/sfm.dir/src/Segmentation.cpp.o.d -o CMakeFiles/sfm.dir/src/Segmentation.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/src/Segmentation.cpp

CMakeFiles/sfm.dir/src/Segmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/Segmentation.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/src/Segmentation.cpp > CMakeFiles/sfm.dir/src/Segmentation.cpp.i

CMakeFiles/sfm.dir/src/Segmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/Segmentation.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/src/Segmentation.cpp -o CMakeFiles/sfm.dir/src/Segmentation.cpp.s

CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/calibration/CameraCalibration.cpp
CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o: CMakeFiles/sfm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o"
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
"CMakeFiles/sfm.dir/src/IOUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/CommonUtil.cpp.o" \
"CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o" \
"CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o" \
"CMakeFiles/sfm.dir/src/Segmentation.cpp.o" \
"CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o"

# External object files for target sfm
sfm_EXTERNAL_OBJECTS =

sfm: CMakeFiles/sfm.dir/src/SfmReconstruction.cpp.o
sfm: CMakeFiles/sfm.dir/src/FeatureUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/StereoUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/main.cpp.o
sfm: CMakeFiles/sfm.dir/src/IOUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/PlottingUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/CommonUtil.cpp.o
sfm: CMakeFiles/sfm.dir/src/SfmBundleAdjustment.cpp.o
sfm: CMakeFiles/sfm.dir/src/PMVS2Reconstruction.cpp.o
sfm: CMakeFiles/sfm.dir/src/Segmentation.cpp.o
sfm: CMakeFiles/sfm.dir/calibration/CameraCalibration.cpp.o
sfm: CMakeFiles/sfm.dir/build.make
sfm: /usr/local/opt/libunistring/lib/libunistring.dylib
sfm: /usr/local/opt/mbedtls/lib/libmbedcrypto.dylib
sfm: /usr/local/lib/libopencv_gapi.4.9.0.dylib
sfm: /usr/local/lib/libopencv_stitching.4.9.0.dylib
sfm: /usr/local/lib/libopencv_alphamat.4.9.0.dylib
sfm: /usr/local/lib/libopencv_aruco.4.9.0.dylib
sfm: /usr/local/lib/libopencv_bgsegm.4.9.0.dylib
sfm: /usr/local/lib/libopencv_bioinspired.4.9.0.dylib
sfm: /usr/local/lib/libopencv_ccalib.4.9.0.dylib
sfm: /usr/local/lib/libopencv_dnn_objdetect.4.9.0.dylib
sfm: /usr/local/lib/libopencv_dnn_superres.4.9.0.dylib
sfm: /usr/local/lib/libopencv_dpm.4.9.0.dylib
sfm: /usr/local/lib/libopencv_face.4.9.0.dylib
sfm: /usr/local/lib/libopencv_freetype.4.9.0.dylib
sfm: /usr/local/lib/libopencv_fuzzy.4.9.0.dylib
sfm: /usr/local/lib/libopencv_hfs.4.9.0.dylib
sfm: /usr/local/lib/libopencv_img_hash.4.9.0.dylib
sfm: /usr/local/lib/libopencv_intensity_transform.4.9.0.dylib
sfm: /usr/local/lib/libopencv_line_descriptor.4.9.0.dylib
sfm: /usr/local/lib/libopencv_mcc.4.9.0.dylib
sfm: /usr/local/lib/libopencv_quality.4.9.0.dylib
sfm: /usr/local/lib/libopencv_rapid.4.9.0.dylib
sfm: /usr/local/lib/libopencv_reg.4.9.0.dylib
sfm: /usr/local/lib/libopencv_rgbd.4.9.0.dylib
sfm: /usr/local/lib/libopencv_saliency.4.9.0.dylib
sfm: /usr/local/lib/libopencv_sfm.4.9.0.dylib
sfm: /usr/local/lib/libopencv_stereo.4.9.0.dylib
sfm: /usr/local/lib/libopencv_structured_light.4.9.0.dylib
sfm: /usr/local/lib/libopencv_superres.4.9.0.dylib
sfm: /usr/local/lib/libopencv_surface_matching.4.9.0.dylib
sfm: /usr/local/lib/libopencv_tracking.4.9.0.dylib
sfm: /usr/local/lib/libopencv_videostab.4.9.0.dylib
sfm: /usr/local/lib/libopencv_viz.4.9.0.dylib
sfm: /usr/local/lib/libopencv_wechat_qrcode.4.9.0.dylib
sfm: /usr/local/lib/libopencv_xfeatures2d.4.9.0.dylib
sfm: /usr/local/lib/libopencv_xobjdetect.4.9.0.dylib
sfm: /usr/local/lib/libopencv_xphoto.4.9.0.dylib
sfm: /usr/local/lib/libboost_program_options-mt.dylib
sfm: /usr/local/lib/libceres.2.2.0.dylib
sfm: /usr/local/lib/libpcl_visualization.dylib
sfm: /usr/local/lib/libpcl_surface.dylib
sfm: /usr/local/lib/libpcl_segmentation.dylib
sfm: /usr/local/lib/libflann_cpp.1.9.2.dylib
sfm: /usr/local/lib/libopencv_shape.4.9.0.dylib
sfm: /usr/local/lib/libopencv_highgui.4.9.0.dylib
sfm: /usr/local/lib/libopencv_datasets.4.9.0.dylib
sfm: /usr/local/lib/libopencv_plot.4.9.0.dylib
sfm: /usr/local/lib/libopencv_text.4.9.0.dylib
sfm: /usr/local/lib/libopencv_ml.4.9.0.dylib
sfm: /usr/local/lib/libopencv_phase_unwrapping.4.9.0.dylib
sfm: /usr/local/Cellar/jsoncpp/1.9.5/lib/libjsoncpp.dylib
sfm: /usr/local/lib/libopencv_optflow.4.9.0.dylib
sfm: /usr/local/lib/libopencv_ximgproc.4.9.0.dylib
sfm: /usr/local/lib/libopencv_video.4.9.0.dylib
sfm: /usr/local/lib/libopencv_videoio.4.9.0.dylib
sfm: /usr/local/lib/libopencv_imgcodecs.4.9.0.dylib
sfm: /usr/local/lib/libopencv_objdetect.4.9.0.dylib
sfm: /usr/local/lib/libopencv_calib3d.4.9.0.dylib
sfm: /usr/local/lib/libopencv_dnn.4.9.0.dylib
sfm: /usr/local/lib/libopencv_features2d.4.9.0.dylib
sfm: /usr/local/lib/libopencv_flann.4.9.0.dylib
sfm: /usr/local/lib/libopencv_photo.4.9.0.dylib
sfm: /usr/local/lib/libopencv_imgproc.4.9.0.dylib
sfm: /usr/local/lib/libopencv_core.4.9.0.dylib
sfm: /usr/local/lib/libglog.0.6.0.dylib
sfm: /usr/local/lib/libgflags.2.2.2.dylib
sfm: /usr/local/lib/libpcl_io.dylib
sfm: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/usr/lib/libpcap.tbd
sfm: /usr/local/lib/libpng.dylib
sfm: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/usr/lib/libz.tbd
sfm: /usr/local/lib/libpcl_features.dylib
sfm: /usr/local/lib/libpcl_filters.dylib
sfm: /usr/local/lib/libpcl_sample_consensus.dylib
sfm: /usr/local/lib/libpcl_search.dylib
sfm: /usr/local/lib/libpcl_octree.dylib
sfm: /usr/local/lib/libpcl_kdtree.dylib
sfm: /usr/local/lib/libvtkChartsCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkInteractionImage-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkIOGeometry-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkIOPLY-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingLOD-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkViewsContext2D-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkViewsCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingContextOpenGL2-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkGUISupportQt-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkInteractionWidgets-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersModeling-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkInteractionStyle-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersExtraction-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkIOLegacy-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkIOCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingAnnotation-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingContext2D-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingFreeType-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkfreetype-9.2.9.2.6.dylib
sfm: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/usr/lib/libz.tbd
sfm: /usr/local/lib/libvtkIOImage-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingOpenGL2-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingHyperTreeGrid-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkImagingSources-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkImagingCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingUI-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkRenderingCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonColor-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersGeometry-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersSources-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersGeneral-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonComputationalGeometry-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkFiltersCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonExecutionModel-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonDataModel-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonMisc-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonTransforms-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkCommonMath-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtkkissfft-9.2.9.2.6.dylib
sfm: /usr/local/lib/libGLEW.dylib
sfm: /usr/local/lib/QtOpenGLWidgets.framework/Versions/A/QtOpenGLWidgets
sfm: /usr/local/lib/QtOpenGL.framework/Versions/A/QtOpenGL
sfm: /usr/local/lib/QtWidgets.framework/Versions/A/QtWidgets
sfm: /usr/local/lib/QtGui.framework/Versions/A/QtGui
sfm: /usr/local/lib/QtCore.framework/Versions/A/QtCore
sfm: /usr/local/lib/libvtkCommonCore-9.2.9.2.6.dylib
sfm: /usr/local/lib/libvtksys-9.2.9.2.6.dylib
sfm: /usr/local/lib/libpcl_ml.dylib
sfm: /usr/local/lib/libomp.dylib
sfm: /usr/local/lib/libpcl_common.dylib
sfm: /usr/local/lib/libboost_filesystem-mt.dylib
sfm: /usr/local/lib/libboost_atomic-mt.dylib
sfm: /usr/local/lib/libboost_system-mt.dylib
sfm: /usr/local/lib/libboost_iostreams-mt.dylib
sfm: /usr/local/lib/libboost_serialization-mt.dylib
sfm: /usr/local/Cellar/lz4/1.9.4/lib/liblz4.dylib
sfm: /usr/local/lib/libqhull_r.8.0.2.dylib
sfm: CMakeFiles/sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable sfm"
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
