# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/firefly/work/code/ffmedia_release

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/firefly/work/code/ffmedia_release/build

# Include any dependencies generated for this target.
include CMakeFiles/demo_multi_window.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo_multi_window.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo_multi_window.dir/flags.make

CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o: CMakeFiles/demo_multi_window.dir/flags.make
CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o: ../demo/demo_multi_window.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/work/code/ffmedia_release/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o -c /home/firefly/work/code/ffmedia_release/demo/demo_multi_window.cpp

CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/work/code/ffmedia_release/demo/demo_multi_window.cpp > CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.i

CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/work/code/ffmedia_release/demo/demo_multi_window.cpp -o CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.s

# Object files for target demo_multi_window
demo_multi_window_OBJECTS = \
"CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o"

# External object files for target demo_multi_window
demo_multi_window_EXTERNAL_OBJECTS =

demo_multi_window: CMakeFiles/demo_multi_window.dir/demo/demo_multi_window.cpp.o
demo_multi_window: CMakeFiles/demo_multi_window.dir/build.make
demo_multi_window: CMakeFiles/demo_multi_window.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/firefly/work/code/ffmedia_release/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_multi_window"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_multi_window.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo_multi_window.dir/build: demo_multi_window

.PHONY : CMakeFiles/demo_multi_window.dir/build

CMakeFiles/demo_multi_window.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo_multi_window.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo_multi_window.dir/clean

CMakeFiles/demo_multi_window.dir/depend:
	cd /home/firefly/work/code/ffmedia_release/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/firefly/work/code/ffmedia_release /home/firefly/work/code/ffmedia_release /home/firefly/work/code/ffmedia_release/build /home/firefly/work/code/ffmedia_release/build /home/firefly/work/code/ffmedia_release/build/CMakeFiles/demo_multi_window.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo_multi_window.dir/depend
