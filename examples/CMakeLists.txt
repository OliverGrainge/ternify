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


# Set default output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# operator functions 
add_executable(types operators/types.cpp)
add_executable(matmul operators/matmul.cpp)
add_executable(relu operators/relu.cpp)
add_executable(layernorm operators/layernorm.cpp)
add_executable(gelu operators/gelu.cpp)
add_executable(softmax operators/softmax.cpp)

# layers 
add_executable(linear layers/linear.cpp)



# Set specific output directory for activation function executables
set_target_properties(types matmul relu layernorm gelu softmax
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ops
)

set_target_properties(linear
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/layers
)

# Link the example with the core library.
target_link_libraries(types PRIVATE ternify)
target_link_libraries(matmul PRIVATE ternify)
target_link_libraries(relu PRIVATE ternify)
target_link_libraries(layernorm PRIVATE ternify)
target_link_libraries(gelu PRIVATE ternify)
target_link_libraries(softmax PRIVATE ternify)

target_link_libraries(linear PRIVATE ternify)
