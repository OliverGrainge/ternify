cmake_minimum_required(VERSION 3.10)
project(TernaryGemmBenchmark)

# Use C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add BLAS package
find_package(BLAS REQUIRED)

# Add OpenBLAS include directory
if(APPLE)
    include_directories(/opt/homebrew/opt/openblas/include)
elseif(UNIX)
    include_directories(/usr/include/openblas)
endif()

# Include the header directory.
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ========================== Common executable ==========================
# Base library without NEON
add_library(ternary_gemm_base
    src/backend/common_tgemm.cpp
    src/tgemm.cpp
    src/tgemm_pack_weights.cpp
)

add_executable(benchmark
    benchmarks/benchmark_flops.cpp
)
target_link_libraries(benchmark
    PRIVATE
    ternary_gemm_base
    BLAS::BLAS
)


# ========================== Tests ==========================
enable_testing()

# Add Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(tgemm_tests
    tests/test_tgemm.cpp
    tests/test_tgemm_pack_weights.cpp
)

target_link_libraries(tgemm_tests
    PRIVATE
    ternary_gemm_base
    GTest::gtest_main
    BLAS::BLAS
)

# Register tests
include(GoogleTest)
gtest_discover_tests(tgemm_tests)