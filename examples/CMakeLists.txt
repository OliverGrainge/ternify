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
add_executable(layer_linear layers/linear.cpp)
add_executable(layer_relu layers/relu.cpp)
add_executable(layer_gelu layers/gelu.cpp)
add_executable(layer_softmax layers/softmax.cpp)
add_executable(layer_layernorm layers/layernorm.cpp)
add_executable(layer_add layers/add.cpp)
add_executable(mlp integration/mlp.cpp)


# Set specific output directory for activation function executables
set_target_properties(types matmul relu layernorm gelu softmax
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ops
)

set_target_properties(layer_linear layer_relu layer_gelu layer_softmax layer_layernorm layer_add
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/layers
)

set_target_properties(mlp
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/integration
)


# Link the example with the core library.
target_link_libraries(types PRIVATE ternify)
target_link_libraries(matmul PRIVATE ternify)
target_link_libraries(relu PRIVATE ternify)
target_link_libraries(layernorm PRIVATE ternify)
target_link_libraries(gelu PRIVATE ternify)
target_link_libraries(softmax PRIVATE ternify)

target_link_libraries(layer_linear PRIVATE ternify)
target_link_libraries(layer_relu PRIVATE ternify)
target_link_libraries(layer_gelu PRIVATE ternify)
target_link_libraries(layer_softmax PRIVATE ternify)
target_link_libraries(layer_layernorm PRIVATE ternify)
target_link_libraries(layer_add PRIVATE ternify)

target_link_libraries(mlp PRIVATE ternify)