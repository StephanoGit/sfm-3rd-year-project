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
CMAKE_SOURCE_DIR = /Users/stephano./github/sfm-3rd-year-project/c++/visualiser

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build

# Include any dependencies generated for this target.
include CMakeFiles/pcl_visualizer_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pcl_visualizer_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pcl_visualizer_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pcl_visualizer_demo.dir/flags.make

CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o: CMakeFiles/pcl_visualizer_demo.dir/flags.make
CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o: /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/pcl_visualizer_demo.cpp
CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o: CMakeFiles/pcl_visualizer_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o -MF CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o.d -o CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o -c /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/pcl_visualizer_demo.cpp

CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/pcl_visualizer_demo.cpp > CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.i

CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/pcl_visualizer_demo.cpp -o CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.s

# Object files for target pcl_visualizer_demo
pcl_visualizer_demo_OBJECTS = \
"CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o"

# External object files for target pcl_visualizer_demo
pcl_visualizer_demo_EXTERNAL_OBJECTS =

pcl_visualizer_demo: CMakeFiles/pcl_visualizer_demo.dir/pcl_visualizer_demo.cpp.o
pcl_visualizer_demo: CMakeFiles/pcl_visualizer_demo.dir/build.make
pcl_visualizer_demo: /usr/local/lib/libpcl_apps.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_outofcore.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_people.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_simulation.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_keypoints.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_tracking.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_recognition.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_registration.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_stereo.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_segmentation.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_ml.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_features.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_filters.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_sample_consensus.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_visualization.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_io.dylib
pcl_visualizer_demo: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/usr/lib/libpcap.tbd
pcl_visualizer_demo: /usr/local/lib/libpng.dylib
pcl_visualizer_demo: /usr/local/lib/libzlibstatic.a
pcl_visualizer_demo: /usr/local/lib/libpcl_surface.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_search.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_kdtree.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_octree.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkChartsCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkInteractionImage-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkIOGeometry-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkIOPLY-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingLOD-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkViewsContext2D-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkViewsCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingContextOpenGL2-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkGUISupportQt-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkInteractionWidgets-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersModeling-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkInteractionStyle-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersExtraction-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkIOLegacy-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkIOCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingAnnotation-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingContext2D-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingFreeType-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkfreetype-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libzlibstatic.a
pcl_visualizer_demo: /usr/local/lib/libvtkIOImage-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingOpenGL2-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingHyperTreeGrid-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkImagingSources-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkImagingCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingUI-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkRenderingCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonColor-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersGeometry-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersSources-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersGeneral-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonComputationalGeometry-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkFiltersCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonExecutionModel-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonDataModel-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonMisc-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonTransforms-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkCommonMath-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtkkissfft-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libGLEW.dylib
pcl_visualizer_demo: /usr/local/lib/QtOpenGLWidgets.framework/Versions/A/QtOpenGLWidgets
pcl_visualizer_demo: /usr/local/lib/QtOpenGL.framework/Versions/A/QtOpenGL
pcl_visualizer_demo: /usr/local/lib/QtWidgets.framework/Versions/A/QtWidgets
pcl_visualizer_demo: /usr/local/lib/QtGui.framework/Versions/A/QtGui
pcl_visualizer_demo: /usr/local/lib/QtCore.framework/Versions/A/QtCore
pcl_visualizer_demo: /usr/local/lib/libvtkCommonCore-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libvtksys-9.2.9.2.6.dylib
pcl_visualizer_demo: /usr/local/lib/libpcl_common.dylib
pcl_visualizer_demo: /usr/local/lib/libboost_system-mt.dylib
pcl_visualizer_demo: /usr/local/lib/libboost_filesystem-mt.dylib
pcl_visualizer_demo: /usr/local/lib/libboost_atomic-mt.dylib
pcl_visualizer_demo: /usr/local/lib/libboost_iostreams-mt.dylib
pcl_visualizer_demo: /usr/local/lib/libboost_serialization-mt.dylib
pcl_visualizer_demo: /usr/local/lib/libGLEW.dylib
pcl_visualizer_demo: /usr/local/lib/libflann_cpp.1.9.2.dylib
pcl_visualizer_demo: /usr/local/Cellar/lz4/1.9.4/lib/liblz4.dylib
pcl_visualizer_demo: /usr/local/lib/libqhull_r.8.0.2.dylib
pcl_visualizer_demo: CMakeFiles/pcl_visualizer_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pcl_visualizer_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcl_visualizer_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pcl_visualizer_demo.dir/build: pcl_visualizer_demo
.PHONY : CMakeFiles/pcl_visualizer_demo.dir/build

CMakeFiles/pcl_visualizer_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pcl_visualizer_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pcl_visualizer_demo.dir/clean

CMakeFiles/pcl_visualizer_demo.dir/depend:
	cd /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/stephano./github/sfm-3rd-year-project/c++/visualiser /Users/stephano./github/sfm-3rd-year-project/c++/visualiser /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build /Users/stephano./github/sfm-3rd-year-project/c++/visualiser/build/CMakeFiles/pcl_visualizer_demo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/pcl_visualizer_demo.dir/depend

