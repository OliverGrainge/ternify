# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/olivergrainge/Documents/github/ternify/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/olivergrainge/Documents/github/ternify/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/matmul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matmul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matmul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matmul.dir/flags.make

CMakeFiles/matmul.dir/matmul.cpp.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/matmul.cpp.o: /Users/olivergrainge/Documents/github/ternify/examples/matmul.cpp
CMakeFiles/matmul.dir/matmul.cpp.o: CMakeFiles/matmul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/olivergrainge/Documents/github/ternify/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matmul.dir/matmul.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matmul.dir/matmul.cpp.o -MF CMakeFiles/matmul.dir/matmul.cpp.o.d -o CMakeFiles/matmul.dir/matmul.cpp.o -c /Users/olivergrainge/Documents/github/ternify/examples/matmul.cpp

CMakeFiles/matmul.dir/matmul.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/matmul.dir/matmul.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/olivergrainge/Documents/github/ternify/examples/matmul.cpp > CMakeFiles/matmul.dir/matmul.cpp.i

CMakeFiles/matmul.dir/matmul.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/matmul.dir/matmul.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/olivergrainge/Documents/github/ternify/examples/matmul.cpp -o CMakeFiles/matmul.dir/matmul.cpp.s

# Object files for target matmul
matmul_OBJECTS = \
"CMakeFiles/matmul.dir/matmul.cpp.o"

# External object files for target matmul
matmul_EXTERNAL_OBJECTS =

matmul: CMakeFiles/matmul.dir/matmul.cpp.o
matmul: CMakeFiles/matmul.dir/build.make
matmul: libternify.a
matmul: CMakeFiles/matmul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/olivergrainge/Documents/github/ternify/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable matmul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matmul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matmul.dir/build: matmul
.PHONY : CMakeFiles/matmul.dir/build

CMakeFiles/matmul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matmul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matmul.dir/clean

CMakeFiles/matmul.dir/depend:
	cd /Users/olivergrainge/Documents/github/ternify/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/olivergrainge/Documents/github/ternify/examples /Users/olivergrainge/Documents/github/ternify/examples /Users/olivergrainge/Documents/github/ternify/examples/build /Users/olivergrainge/Documents/github/ternify/examples/build /Users/olivergrainge/Documents/github/ternify/examples/build/CMakeFiles/matmul.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/matmul.dir/depend

