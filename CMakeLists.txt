cmake_minimum_required(VERSION 3.10)
project(TernaryGemmBenchmark)

# Use C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the header directory.
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set output directories for all targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add this to ensure proper output paths on multi-configuration generators (like Visual Studio)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# ========================== Common executable ==========================
# Base library without NEON
add_library(ternary_gemm_base
    src/backend/common_tgemm.cpp
    src/tgemm_pack_weights.cpp
    src/tgemm.cpp
)

add_executable(benchmark_baseline
    benchmarks/benchmark_tgemm.cpp
)
target_link_libraries(benchmark_baseline
    PRIVATE
    ternary_gemm_base
)

# ========================== NEON executable ==========================
# NEON-specific library
add_library(ternary_gemm_neon
    src/backend/common_tgemm.cpp
    src/tgemm_pack_weights.cpp
    src/tgemm.cpp
)

# Add NEON-specific flags to the NEON library
target_compile_definitions(ternary_gemm_neon PRIVATE USE_NEON=1)
target_compile_options(ternary_gemm_neon PRIVATE -mfpu=neon)

add_executable(benchmark_neon
    benchmarks/benchmark_tgemm.cpp
)

target_link_libraries(benchmark_neon
    PRIVATE
    ternary_gemm_neon
)

target_compile_definitions(benchmark_neon PRIVATE USE_NEON=1)
target_compile_options(benchmark_neon PRIVATE -mfpu=neon)

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
)

# Register tests
include(GoogleTest)
gtest_discover_tests(tgemm_tests)