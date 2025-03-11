cmake_minimum_required(VERSION 3.10)
project(MyProjectExamples)

# Define the root directory as the parent of the current source directory.
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Create the core library from the src/ files located one directory up.
file(GLOB_RECURSE CORE_SOURCES "${ROOT_DIR}/src/*.cpp")
add_library(ternify ${CORE_SOURCES})

# Include the headers from the include/ folder and its subdirectories
target_include_directories(ternify PUBLIC 
    ${ROOT_DIR}/include
)


# Build an example executable from the example source file.
add_executable(types types.cpp)
add_executable(matmul matmul.cpp)
add_executable(relu relu.cpp)

# Link the example with the core library.
target_link_libraries(types PRIVATE ternify)
target_link_libraries(matmul PRIVATE ternify)
target_link_libraries(relu PRIVATE ternify)